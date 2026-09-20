"""Check NumPy interoperability used by SAM3 image, video, and mask workflows."""

import cv2
import decord
import numpy as np
import pycocotools.mask as mask_utils
import pytest
import torch
from PIL import Image
from sam3.agent.helpers.visualizer import _PanopticPrediction, GenericMask
from sam3.model.io_utils import (
    load_resource_as_video_frames,
    load_video_frames_from_video_file,
)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_numpy_tensor_bridges(dtype):
    """Image/video consumers preserve values through Torch and Decord bridges."""
    array = np.arange(24, dtype=dtype).reshape(2, 3, 4)
    np.testing.assert_array_equal(torch.from_numpy(array).numpy(), array)
    np.testing.assert_array_equal(decord.ndarray.array(array).asnumpy(), array)


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


def test_mask_encoding_and_visualization():
    """Exercise the compiled COCO mask extension and SAM3's visualization adapter."""
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:8, 3:10] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    np.testing.assert_array_equal(mask_utils.decode(encoded), mask)
    decoded = GenericMask(encoded, *mask.shape)
    np.testing.assert_array_equal(decoded.mask, mask)
    assert decoded.area() == 42


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


def test_video_frame_numpy_conversion(tmp_path):
    """Decode actual frames through both Decord and SAM3's OpenCV-to-Torch loader."""
    path = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, (32, 24))
    assert writer.isOpened()
    try:
        for value in (40, 160):
            writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    reader = decord.VideoReader(str(path), num_threads=1)
    assert reader.get_batch([0, 1]).asnumpy().shape == (2, 24, 32, 3)
    frames, height, width = load_video_frames_from_video_file(
        str(path),
        image_size=16,
        offload_video_to_cpu=True,
        img_mean=(0, 0, 0),
        img_std=(1, 1, 1),
        async_loading_frames=False,
    )
    assert (height, width) == (24, 32)
    assert frames.shape == (2, 3, 16, 16)
    assert torch.isfinite(frames).all()
    torch.testing.assert_close(
        frames.mean(dim=(1, 2, 3)), torch.tensor([40.0, 160.0]), rtol=0, atol=2
    )
