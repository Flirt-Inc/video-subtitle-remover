import logging
import os
import sys
import numpy as np
import torch
import cv2
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend import config

logger = logging.getLogger(__name__)

# VAE temporal constraint: (F-1) % 4 == 0, max 81
MINIMAX_MAX_FRAMES = 81
MINIMAX_OVERLAP = 8


class MinimaxInpaint:
    """Video inpainting using MiniMax-Remover (Wan2.1-based diffusion)."""

    def __init__(self):
        self.device = config.device
        model_path = config.MINIMAX_MODEL_PATH
        self.num_inference_steps = getattr(config, 'MINIMAX_INFERENCE_STEPS', 8)
        self.target_short_side = getattr(config, 'MINIMAX_TARGET_SHORT_SIDE', 480)

        logger.info(f"Loading MiniMax-Remover from {model_path}")
        print(f'[MiniMax] Loading model from {model_path}')

        from backend.inpaint.minimax.transformer_minimax_remover import Transformer3DModel
        from backend.inpaint.minimax.pipeline_minimax_remover import Minimax_Remover_Pipeline
        from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler

        transformer = Transformer3DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.float16
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        self.pipe = Minimax_Remover_Pipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        if getattr(config, 'MINIMAX_CPU_OFFLOAD', True) and torch.cuda.is_available():
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        logger.info("MiniMax-Remover loaded")
        print('[MiniMax] Model loaded')

    @staticmethod
    def _snap_frame_count(n: int) -> int:
        """Snap frame count down to satisfy (F-1) % 4 == 0.
        Valid counts: 1, 5, 9, 13, ..., 77, 81."""
        if n <= 1:
            return 1
        n = min(n, MINIMAX_MAX_FRAMES)
        return ((n - 1) // 4) * 4 + 1

    def _compute_target_dims(self, h: int, w: int):
        """Compute target dims: scale short side to target, round to multiple of 8."""
        short_side = min(h, w)
        if short_side > self.target_short_side:
            scale = self.target_short_side / short_side
            new_h = int(h * scale)
            new_w = int(w * scale)
        else:
            new_h, new_w = h, w
        # Round to nearest multiple of 8
        new_h = (new_h // 8) * 8
        new_w = (new_w // 8) * 8
        return max(new_h, 8), max(new_w, 8)

    def __call__(self, input_frames: List[np.ndarray], input_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Inpaint video frames using MiniMax-Remover.

        Args:
            input_frames: list of [H, W, 3] uint8 BGR frames (OpenCV convention)
            input_masks:  list of [H, W] uint8 masks (255=remove, 0=keep)
        Returns:
            list of [H, W, 3] uint8 BGR inpainted frames
        """
        total_frames = len(input_frames)
        if total_frames == 0:
            return []

        H, W = input_frames[0].shape[:2]
        target_h, target_w = self._compute_target_dims(H, W)
        print(f'[MiniMax] {total_frames} frames, {W}x{H} -> {target_w}x{target_h} for inference')

        # Stack to numpy [F, H, W, 3], BGR -> RGB, normalize to [-1, 1]
        frames_rgb = np.stack([f[:, :, ::-1] for f in input_frames])  # BGR->RGB, [F,H,W,3] uint8
        frames_tensor = torch.from_numpy(frames_rgb.copy()).float() / 127.5 - 1.0  # [-1, 1]

        # Stack masks [F, H, W] uint8 -> [F, H, W, 1] float [0, 1]
        masks_np = np.stack(input_masks)  # [F, H, W] uint8
        masks_tensor = torch.from_numpy(masks_np.copy()).float() / 255.0  # [0, 1]
        masks_tensor = masks_tensor.unsqueeze(-1)  # [F, H, W, 1]

        # Process in sliding window chunks
        output_rgb = self._sliding_window_inpaint(
            frames_tensor, masks_tensor, target_h, target_w
        )  # [F, H, W, 3] uint8 RGB at TARGET resolution

        # Upscale back to original resolution if needed
        if target_h != H or target_w != W:
            output_rgb = np.stack([
                cv2.resize(output_rgb[i], (W, H), interpolation=cv2.INTER_LANCZOS4)
                for i in range(total_frames)
            ])

        # RGB -> BGR
        output_bgr = output_rgb[:, :, :, ::-1].copy()
        return [output_bgr[i] for i in range(total_frames)]

    def _sliding_window_inpaint(self, frames: torch.Tensor, masks: torch.Tensor,
                                 target_h: int, target_w: int) -> np.ndarray:
        """Process video via sliding window with overlap blending.

        Args:
            frames: [F, H, W, 3] float tensor [-1, 1]
            masks:  [F, H, W, 1] float tensor [0, 1]
            target_h, target_w: inference resolution
        Returns:
            [F, target_h, target_w, 3] uint8 RGB
        """
        total = frames.shape[0]
        chunk_size = self._snap_frame_count(MINIMAX_MAX_FRAMES)  # 81
        overlap = MINIMAX_OVERLAP
        stride = chunk_size - overlap  # 73

        if total <= chunk_size:
            # Single chunk — snap frame count
            valid_count = self._snap_frame_count(total)
            result = self._inpaint_chunk(frames[:valid_count], masks[:valid_count], target_h, target_w)
            if valid_count < total:
                # Process remaining frames by repeating last result frame
                padding = np.tile(result[-1:], (total - valid_count, 1, 1, 1))
                result = np.concatenate([result, padding], axis=0)
            return result

        # Multiple chunks with overlap blending
        output = np.zeros((total, target_h, target_w, 3), dtype=np.float64)
        weights = np.zeros((total, 1, 1, 1), dtype=np.float64)

        chunk_idx = 0
        pos = 0
        while pos < total:
            end = min(pos + chunk_size, total)
            chunk_frames = frames[pos:end]
            chunk_masks = masks[pos:end]

            # Snap to valid frame count
            valid_count = self._snap_frame_count(len(chunk_frames))
            chunk_frames = chunk_frames[:valid_count]
            chunk_masks = chunk_masks[:valid_count]

            chunk_idx += 1
            print(f'[MiniMax] Processing chunk {chunk_idx}: frames {pos+1}-{pos+valid_count} of {total}')
            result = self._inpaint_chunk(chunk_frames, chunk_masks, target_h, target_w)

            # Build linear blend weights for overlap region
            n = result.shape[0]
            w = np.ones(n, dtype=np.float64)
            if pos > 0 and overlap > 0:
                ramp_len = min(overlap, n)
                w[:ramp_len] = np.linspace(0.0, 1.0, ramp_len)
            if pos + valid_count < total and overlap > 0:
                ramp_len = min(overlap, n)
                w[-ramp_len:] = np.linspace(1.0, 0.0, ramp_len)

            w_4d = w[:, None, None, None]
            output[pos:pos + n] += result.astype(np.float64) * w_4d
            weights[pos:pos + n] += w_4d

            pos += stride
            if pos + valid_count >= total and end >= total:
                break

        # Handle any remaining frames that weren't covered
        uncovered = weights[:, 0, 0, 0] == 0
        if np.any(uncovered):
            # Fill with nearest covered frame
            for i in range(total):
                if uncovered[i]:
                    # Find nearest covered frame
                    for j in range(1, total):
                        if i - j >= 0 and not uncovered[i - j]:
                            output[i] = output[i - j] / max(weights[i - j, 0, 0, 0], 1e-8)
                            weights[i] = 1.0
                            break
                        if i + j < total and not uncovered[i + j]:
                            output[i] = output[i + j] / max(weights[i + j, 0, 0, 0], 1e-8)
                            weights[i] = 1.0
                            break

        weights = np.maximum(weights, 1e-8)
        output = (output / weights).clip(0, 255).astype(np.uint8)
        return output

    def _inpaint_chunk(self, frames: torch.Tensor, masks: torch.Tensor,
                        target_h: int, target_w: int) -> np.ndarray:
        """Run MiniMax inference on a single chunk.

        Args:
            frames: [F, H, W, 3] float tensor [-1, 1]
            masks:  [F, H, W, 1] float tensor [0, 1]
            target_h, target_w: target inference resolution
        Returns:
            [F, target_h, target_w, 3] uint8 RGB
        """
        num_frames = frames.shape[0]

        with torch.no_grad():
            result = self.pipe(
                height=target_h,
                width=target_w,
                num_frames=num_frames,
                num_inference_steps=self.num_inference_steps,
                images=frames,
                masks=masks,
                iterations=0,  # No extra dilation — CRAFT masks already dilated
                output_type="np",
            )

        # result.frames[0] is numpy [F, H, W, 3] in [0, 1]
        video = result.frames[0]
        output = (video * 255.0).clip(0, 255).astype(np.uint8)
        return output
