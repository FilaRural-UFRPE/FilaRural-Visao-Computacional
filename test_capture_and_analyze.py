import importlib
import os
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np


# A suíte unitária não precisa dos binários pesados de OpenCV/ONNX. O módulo
# falso deixa os limites externos sob mock e mantém os testes rápidos no CI.
sys.modules.setdefault("cv2", Mock())

os.environ.setdefault("RTSP_PASSWORD", "test-password")
os.environ.setdefault("RTSP_IP", "127.0.0.1")
capture = importlib.import_module("capture_and_analyze")


class CaptureFrameTests(unittest.TestCase):
    @patch.object(capture.cv2, "VideoCapture")
    def test_retries_transient_reads_and_releases_capture(self, video_capture):
        camera = video_capture.return_value
        camera.isOpened.return_value = True
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        camera.read.side_effect = [(False, None)] + [(True, frame)] * 5

        result = capture.capture_frame("rtsp://camera")

        self.assertIs(result, frame)
        self.assertEqual(camera.read.call_count, 6)
        camera.release.assert_called_once_with()

    @patch.object(capture.cv2, "VideoCapture")
    def test_releases_capture_when_stream_cannot_be_opened(self, video_capture):
        camera = video_capture.return_value
        camera.isOpened.return_value = False

        self.assertIsNone(capture.capture_frame("rtsp://camera"))
        camera.release.assert_called_once_with()

    @patch.object(capture.requests, "post")
    @patch.object(capture.cv2, "imencode")
    def test_rejects_non_object_api_response(self, imencode, post):
        encoded = Mock()
        encoded.tobytes.return_value = b"jpeg"
        imencode.return_value = (True, encoded)
        post.return_value.json.return_value = []

        self.assertIsNone(capture.send_to_api(Mock(), "https://api.test/analyze"))


if __name__ == "__main__":
    unittest.main()
