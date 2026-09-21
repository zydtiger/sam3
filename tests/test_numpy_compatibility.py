"""Check NumPy interoperability used by SAM3 image, video, and mask workflows."""

import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import cv2
import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch
from PIL import Image

from sam3.agent.helpers.visualizer import GenericMask, _PanopticPrediction
from sam3.agent.viz import visualize
from sam3.model.io_utils import (
    load_resource_as_video_frames,
    load_video_frames_from_video_file,
)
from sam3.model.utils.sam2_utils import (
    load_video_frames_from_video_file as load_tracker_video,
)
from sam3.model.video_io import read_video_frame
from sam3.train.data.sam3_image_dataset import CustomCocoDetectionAPI


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_numpy_tensor_bridges(dtype):
    """Image/video consumers preserve values through the NumPy/Torch bridge."""
    array = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    np.testing.assert_array_equal(torch.from_numpy(array).numpy(), array)


def test_image_frame_normalization():
    """SAM3 image loading keeps RGB geometry and normalized values across NumPy versions."""
    rgb = np.full((12, 16, 3), [0, 128, 255], dtype=np.uint8)
    frames, height, width = load_resource_as_video_frames(
        [Image.fromarray(rgb)], image_size=8, offload_video_to_cpu=True
    )
    assert (height, width) == (12, 16)
    assert frames.shape == (1, 3, 8, 8)
    expected = (torch.tensor([0, 128, 255], dtype=torch.float16) / 255 - 0.5) / 0.5
    torch.testing.assert_close(frames[0, :, 0, 0], expected)


def test_mask_encoding_and_visualization(tmp_path):
    """Exercise the compiled COCO mask extension and SAM3's visualization adapter."""
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:8, 3:10] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    np.testing.assert_array_equal(mask_utils.decode(encoded), mask)
    decoded = GenericMask(encoded, *mask.shape)
    np.testing.assert_array_equal(decoded.mask, mask)
    assert decoded.area() == 42
    image_path = tmp_path / "image.png"
    Image.new("RGB", (16, 12), "white").save(image_path)
    rendered = visualize(
        {
            "orig_img_h": 12,
            "orig_img_w": 16,
            "original_image_path": str(image_path),
            "pred_boxes": [[3, 2, 10, 8]],
            "pred_masks": [encoded["counts"]],
        }
    )
    assert rendered.size == (16, 12)
    assert np.any(np.asarray(rendered) != 255)


def test_panoptic_boolean_masks():
    """Panoptic rendering returns boolean arrays for NumPy mask consumers."""
    labels = torch.tensor([[-1, 1, 1], [-1, 2, 2]])
    prediction = _PanopticPrediction(
        labels,
        [
            {"id": 1, "category_id": 0, "isthing": False},
            {"id": 2, "category_id": 1, "isthing": True},
        ],
    )
    masks = [
        (prediction.non_empty_mask(), labels != -1),
        (next(prediction.semantic_masks())[0], labels == 1),
        (next(prediction.instance_masks())[0], labels == 2),
    ]
    for actual, expected in masks:
        assert actual.dtype == np.bool_
        np.testing.assert_array_equal(actual, expected.numpy())


@pytest.mark.parametrize("backend", ["pyav", "cv2"])
def test_video_frame_numpy_conversion(tmp_path, backend):
    """Decode actual frames through PyAV and SAM3's selectable video loaders."""
    path = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, (32, 24))
    assert writer.isOpened()
    try:
        for value in (40, 160):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    with av.open(str(path)) as reader:
        decoded = [frame.to_ndarray(format="rgb24") for frame in reader.decode(video=0)]
    assert np.stack(decoded).shape == (2, 24, 32, 3)
    for index in (0, 1, -1):
        np.testing.assert_array_equal(
            np.asarray(read_video_frame(str(path), index)), decoded[index]
        )
    with pytest.raises(IndexError):
        read_video_frame(str(path), 2)
    frames, height, width = load_video_frames_from_video_file(
        str(path),
        image_size=16,
        offload_video_to_cpu=True,
        img_mean=(0, 0, 0),
        img_std=(1, 1, 1),
        async_loading_frames=False,
        video_loader_type=backend,
    )
    assert (height, width) == (24, 32)
    assert frames.shape == (2, 3, 16, 16)
    assert torch.isfinite(frames).all()
    torch.testing.assert_close(
        frames.mean(dim=(1, 2, 3)),
        torch.tensor([40.0, 160.0]) / (255 if backend == "pyav" else 1),
        rtol=0,
        atol=2 / (255 if backend == "pyav" else 1),
    )
    if backend == "pyav":
        tracker_frames, tracker_height, tracker_width = load_tracker_video(
            path.read_bytes(),
            image_size=16,
            offload_video_to_cpu=True,
        )
        assert (tracker_height, tracker_width) == (height, width)
        torch.testing.assert_close(tracker_frames, (frames - 0.5) / 0.5)


def test_variable_frame_rate_dataset(tmp_path):
    """Training frame references preserve RGB and index order at irregular timestamps."""
    path = tmp_path / "frames.mp4"
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for pts, color in zip((0, 1, 5), colors):
            rgb = np.full((24, 32, 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = pts, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with av.open(str(path)) as container:
        timestamps = [
            frame.pts * frame.time_base for frame in container.decode(video=0)
        ]
    assert timestamps == [Fraction(0), Fraction(1, 25), Fraction(5, 25)]

    dataset = object.__new__(CustomCocoDetectionAPI)
    dataset.root = str(tmp_path)
    dataset.fix_fname = False
    dataset.blurring_masks_path = None
    dataset.coco = SimpleNamespace(
        loadImagesFromDatapoint=lambda _: [
            {"id": index, "file_name": f"frames.mp4@{index}"}
            for index in range(len(colors))
        ]
    )
    images, _ = dataset._load_images(0)
    for (index, image), color in zip(images, colors):
        np.testing.assert_array_equal(
            np.asarray(image), np.full((24, 32, 3), color, dtype=np.uint8)
        )
    frames, height, width = load_resource_as_video_frames(
        str(path), image_size=16, offload_video_to_cpu=True
    )
    assert (height, width) == (24, 32)
    torch.testing.assert_close(
        frames.mean(dim=(2, 3)), torch.tensor(colors, dtype=torch.float32) / 127.5 - 1
    )


def test_image_import_without_video_dependencies():
    """Base image imports work without decoders or the video process monitor."""
    source_root = Path(__file__).resolve().parents[1]
    code = r"""
import sys
class BlockVideoDependencies:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"av", "decord", "psutil", "cv2", "torchcodec"}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockVideoDependencies())
from PIL import Image
import sam3
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.io_utils import load_resource_as_video_frames
from sam3.model.video_io import read_video_frame
frames, height, width = load_resource_as_video_frames(
    [Image.new("RGB", (16, 12))], image_size=8, offload_video_to_cpu=True,
)
assert frames.shape == (1, 3, 8, 8)
assert (height, width) == (12, 16)
for invoke in (
    lambda: read_video_frame("missing.mp4", 0),
    lambda: sam3.model_builder.build_sam3_video_predictor(),
):
    try:
        invoke()
    except ImportError as exc:
        assert '.[video]' in str(exc), str(exc)
    else:
        raise AssertionError("Expected an actionable missing-extra error")
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=source_root, check=True)
