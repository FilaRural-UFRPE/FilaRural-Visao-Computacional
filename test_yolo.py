import unittest
from unittest.mock import Mock, patch
import sys

import numpy as np

sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("onnxruntime", Mock())

import yolo
from yolo import YoloONNX


class YoloTests(unittest.TestCase):
    def test_preprocess_converts_opencv_bgr_to_rgb(self):
        detector = object.__new__(YoloONNX)
        bgr_red = np.zeros((1, 1, 3), dtype=np.uint8)
        bgr_red[0, 0] = [0, 0, 255]
        rgb_red = bgr_red[:, :, ::-1]

        with patch("yolo.cv2.cvtColor", return_value=rgb_red) as convert, \
                patch("yolo.cv2.resize", return_value=np.tile(rgb_red, (640, 640, 1))):
            blob, _, _ = detector._preprocess(bgr_red)

        convert.assert_called_once_with(bgr_red, yolo.cv2.COLOR_BGR2RGB)
        self.assertEqual(blob[0, 0, 0, 0], 1.0)
        self.assertEqual(blob[0, 2, 0, 0], 0.0)

    def test_read_falls_back_to_full_frame_when_roi_is_empty(self):
        detector = object.__new__(YoloONNX)
        detector.filepath = None
        detector.image = None
        detector.roi_offset = (0, 0)
        detector.detections = []
        detector.input_name = "images"
        detector.session = Mock()
        detector.session.run.side_effect = ["roi-output", "full-output"]
        detector._crop_roi = Mock(return_value=(np.zeros((5, 10, 3)), (0, 5)))
        detector._preprocess = Mock(side_effect=[("roi-blob", 1, (10, 5)), ("full-blob", 1, (10, 10))])
        full_detections = [(1, 2, 3, 4, 0.8)]
        detector._postprocess = Mock(side_effect=[[], full_detections])

        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with patch("yolo.cv2.imread", return_value=image):
            result = detector.read("frame.jpg")

        self.assertEqual(result, 0)
        self.assertEqual(detector.detections, full_detections)
        self.assertEqual(detector.roi_offset, (0, 0))
        self.assertEqual(detector.session.run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
