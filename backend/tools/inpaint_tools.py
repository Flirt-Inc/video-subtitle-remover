import multiprocessing
import cv2
import numpy as np

import config
from backend.inpaint.lama_inpaint import LamaInpaint


def batch_generator(data, max_batch_size):
    """
    根据data大小，生成最大长度不超过max_batch_size的均匀批次数据
    """
    n_samples = len(data)
    # 尝试找到一个比MAX_BATCH_SIZE小的batch_size，以使得所有的批次数量尽量接近
    batch_size = max_batch_size
    num_batches = n_samples // batch_size

    # 处理最后一批可能不足batch_size的情况
    # 如果最后一批少于其他批次，则减小batch_size尝试平衡每批的数量
    while n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1  # 减小批次大小
        num_batches = n_samples // batch_size

    # 生成前num_batches个批次
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    # 将剩余的数据作为最后一个批次
    last_batch_start = num_batches * batch_size
    if last_batch_start < n_samples:
        yield data[last_batch_start:]


def inference_task(batch_data):
    inpainted_frame_dict = dict()
    for data in batch_data:
        index, original_frame, coords_list = data
        mask_size = original_frame.shape[:2]
        mask = create_mask(mask_size, coords_list)
        inpaint_frame = inpaint(original_frame, mask)
        inpainted_frame_dict[index] = inpaint_frame
    return inpainted_frame_dict


def parallel_inference(inputs, batch_size=None, pool_size=None):
    """
    并行推理，同时保持结果顺序
    """
    if pool_size is None:
        pool_size = multiprocessing.cpu_count()
    # 使用上下文管理器自动管理进程池
    with multiprocessing.Pool(processes=pool_size) as pool:
        batched_inputs = list(batch_generator(inputs, batch_size))
        # 使用map函数保证输入输出的顺序是一致的
        batch_results = pool.map(inference_task, batched_inputs)
    # 将批推理结果展平
    index_inpainted_frames = [item for sublist in batch_results for item in sublist]
    return index_inpainted_frames


def inpaint(img, mask):
    lama_inpaint_instance = LamaInpaint()
    img_inpainted = lama_inpaint_instance(img, mask)
    return img_inpainted


def inpaint_with_multiple_masks(censored_img, mask_list):
    inpainted_frame = censored_img
    if mask_list:
        for mask in mask_list:
            inpainted_frame = inpaint(inpainted_frame, mask)
    return inpainted_frame


_craft_net = None
_refine_net = None


def _patch_torchvision_model_urls():
    """Patch missing model_urls for newer torchvision (>= 0.13).

    craft-text-detector imports `model_urls` from `torchvision.models.vgg`
    at module level, but torchvision removed it in favour of the Weights API.
    We inject the dict back so the import succeeds.
    """
    import torchvision.models.vgg as _vgg_mod
    if not hasattr(_vgg_mod, 'model_urls'):
        _vgg_mod.model_urls = {
            'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        }


def _patch_craft_ragged_polys():
    """Patch CRAFT's adjustResultCoordinates for numpy >= 1.24.

    numpy >= 1.24 rejects ``np.array(polys)`` when polygons have different
    numbers of points.  Replace the bulk conversion with per-polygon scaling.
    """
    from craft_text_detector import craft_utils as _cu

    def _adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
        if len(polys) > 0:
            for k in range(len(polys)):
                if polys[k] is not None:
                    polys[k] *= np.array([ratio_w * ratio_net, ratio_h * ratio_net])
        return polys

    _cu.adjustResultCoordinates = _adjustResultCoordinates


def _get_craft_models():
    """Lazy-load CRAFT models (singleton)."""
    global _craft_net, _refine_net
    if _craft_net is None:
        import torch
        _patch_torchvision_model_urls()
        _patch_craft_ragged_polys()
        from craft_text_detector import load_craftnet_model, load_refinenet_model
        use_cuda = torch.cuda.is_available()
        _craft_net = load_craftnet_model(cuda=use_cuda)
        _refine_net = load_refinenet_model(cuda=use_cuda)
    return _craft_net, _refine_net


def detect_characters(frame):
    """Run CRAFT character-level detection on a frame, return list of polygons."""
    import torch
    from craft_text_detector import get_prediction
    craft_net, refine_net = _get_craft_models()
    result = get_prediction(
        image=frame,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=torch.cuda.is_available(),
        long_size=1280,
    )
    return result["boxes"]


def create_mask(size, coords_list, polygons=None, frame=None):
    mask = np.zeros(size, dtype="uint8")
    has_content = False
    mask_type = getattr(config, 'MASK_TYPE', 'rect')
    if mask_type == 'character' and frame is not None:
        boxes = detect_characters(frame)
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        has_content = len(boxes) > 0
    elif mask_type == 'polygon' and polygons is not None:
        for poly in polygons:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        has_content = len(polygons) > 0
    elif coords_list:
        for coords in coords_list:
            xmin, xmax, ymin, ymax = coords
            x1 = xmin - config.SUBTITLE_AREA_DEVIATION_PIXEL
            if x1 < 0:
                x1 = 0
            y1 = ymin - config.SUBTITLE_AREA_DEVIATION_PIXEL
            if y1 < 0:
                y1 = 0
            x2 = xmax + config.SUBTITLE_AREA_DEVIATION_PIXEL
            y2 = ymax + config.SUBTITLE_AREA_DEVIATION_PIXEL
            cv2.rectangle(mask, (x1, y1),
                          (x2, y2), (255, 255, 255), thickness=-1)
        has_content = True
    # Apply morphological dilation for smoother, rounded mask edges
    if has_content:
        dilation = getattr(config, 'MASK_DILATION_ITERATIONS', 0)
        if dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            mask = cv2.dilate(mask, kernel, iterations=dilation)
    return mask


def inpaint_video(video_path, sub_list):
    index = 0
    frame_to_inpaint_list = []
    video_cap = cv2.VideoCapture(video_path)
    while True:
        # 读取视频帧
        ret, frame = video_cap.read()
        if not ret:
            break
        index += 1
        if index in sub_list.keys():
            frame_to_inpaint_list.append((index, frame, sub_list[index]))
        if len(frame_to_inpaint_list) > config.PROPAINTER_MAX_LOAD_NUM:
            batch_results = parallel_inference(frame_to_inpaint_list)
            for index, frame in batch_results:
                file_name = f'/home/yao/Documents/Project/video-subtitle-remover/test/temp/{index}.png'
                cv2.imwrite(file_name, frame)
                print(f"success write: {file_name}")
            frame_to_inpaint_list.clear()
    print(f'finished')


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
