"""Optional PyAV decoding shared by inference loaders and training datasets.

PyAV is loaded only when a video is opened, so image models do not require the
video extra. Frames are decoded in presentation order to preserve frame indices.
"""

from collections import deque
from importlib import import_module
from io import BytesIO

import torch
from PIL import Image


def decode_video_frames(video_path, image_size=None):
    """Yield RGB arrays and original dimensions for inference and dataset readers."""
    try:
        av = import_module("av")
    except ModuleNotFoundError as exc:
        if exc.name != "av":
            raise
        raise ImportError(
            "Video decoding requires PyAV. Install SAM3 with the video extra: "
            'pip install ".[video]" from the SAM3 checkout.'
        ) from exc

    source = BytesIO(video_path) if isinstance(video_path, bytes) else video_path
    with av.open(source) as container:
        for frame in container.decode(video=0):
            height, width = frame.height, frame.width
            if image_size is not None:
                frame = frame.reformat(
                    width=image_size, height=image_size, format="rgb24"
                )
            yield frame.to_ndarray(format="rgb24"), height, width


def read_video_frame(video_path, frame_index):
    """Read an exact frame index for the training dataset, without FPS-based seeking."""
    frames = decode_video_frames(video_path)
    try:
        if frame_index < 0:
            tail = deque(frames, maxlen=-frame_index)
            if len(tail) == -frame_index:
                return Image.fromarray(tail[0][0])
        else:
            for index, (rgb, _, _) in enumerate(frames):
                if index == frame_index:
                    return Image.fromarray(rgb)
    finally:
        frames.close()
    raise IndexError(f"Frame index {frame_index} is out of range for {video_path}")


def load_video_frames_using_pyav(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.5, 0.5, 0.5),
    img_std=(0.5, 0.5, 0.5),
    compute_device=torch.device("cuda"),
):
    """Return normalized TCHW tensors for both SAM3 video and tracker inference."""
    frames = []
    height = width = None
    for rgb, original_height, original_width in decode_video_frames(
        video_path, image_size=image_size
    ):
        if height is None:
            height, width = original_height, original_width
        frames.append(torch.from_numpy(rgb).permute(2, 0, 1))
    if not frames:
        raise ValueError("Video contains no decodable frames")

    images = torch.stack(frames).to(dtype=torch.float32).div_(255.0)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
    mean = images.new_tensor(img_mean).view(1, 3, 1, 1)
    std = images.new_tensor(img_std).view(1, 3, 1, 1)
    images.sub_(mean).div_(std)
    return images, height, width
