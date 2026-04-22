"""
Unified interface for 2D human pose estimation models.
Currently supports:
- YOLO11-Pose (Ultralytics)
- RTMPose (OpenMMLab/mmpose)
- ViTPose (OpenMMLab/mmpose)
"""

import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple

# Keypoint 17 format (COCO):
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

class BasePoseEstimator(ABC):
    def __init__(self, model_path: str, config_path: str = None, device: str = 'cpu'):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        pass

    @abstractmethod
    def infer(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Run pose estimation on a single RGB or BGR image (H, W, 3).
        Returns:
            A list of persons, where each person is a numpy array of shape (17, 3),
            representing [x, y, confidence] for the 17 COCO keypoints.
        """
        pass

    def __call__(self, image: np.ndarray) -> List[np.ndarray]:
        return self.infer(image)


class YoloPoseEstimator(BasePoseEstimator):
    def _load_model(self):
        from ultralytics import YOLO
        # Ultralytics API automatically handles downloading the .pt if it doesn't exist locally
        return YOLO(self.model_path)

    def infer(self, image: np.ndarray) -> List[np.ndarray]:
        # YOLO requires confidence threshold to prune bad detections.
        results = self.model.predict(image, device=self.device, verbose=False, conf=0.3)

        persons = []
        if len(results) == 0:
            return persons

        r = results[0]
        if r.keypoints is None or r.keypoints.data is None:
            return persons

        # r.keypoints.data signature: (num_persons, 17, 3)
        # The 3 values are (x, y, conf)
        kp_data = r.keypoints.data.cpu().numpy()

        for person_kps in kp_data:
            persons.append(person_kps)

        return persons


class MMPoseEstimator(BasePoseEstimator):
    """
    Wrapper for both RTMPose and ViTPose (or any openmmlab top-down model).
    Since these are usually top-down, they technically need a human bounding box first.
    MMPose's Inferencer encapsulates the (Detector -> Pose) pipeline natively!
    """
    def _load_model(self):
        from mmpose.apis import MMPoseInferencer

        # MMPoseInferencer natively wires up a default object detector (e.g. rtmdet)
        # to feed bounding boxes to the top-down pose estimator.
        inferencer = MMPoseInferencer(
            pose2d=self.config_path,
            pose2d_weights=self.model_path,
            device=self.device
        )
        return inferencer

    def infer(self, image: np.ndarray) -> List[np.ndarray]:
        # Perform inference
        # next() is required because inferencer acts as a generator when batching
        result_gen = self.model(image, return_vis=False)
        result = next(result_gen)

        persons = []

        if 'predictions' not in result or len(result['predictions']) == 0:
            return persons

        preds = result['predictions'][0]

        # Iterate over all detected persons
        for person in preds:
            # keypoints is a list of [x, y], keypoint_scores is a list of [conf]
            kpts = np.array(person['keypoints'])          # (17, 2)
            scores = np.array(person['keypoint_scores'])  # (17,)

            # Combine into (17, 3)
            # Add new axis to scores to make it (17, 1), then concatenate
            person_kp = np.concatenate((kpts, scores[:, np.newaxis]), axis=1)
            persons.append(person_kp)

        return persons


def create_estimator(model_type: str, weights_path: str, config_path: str = None, device: str = 'cpu') -> BasePoseEstimator:
    """Factory function to build the correct pose estimator instance."""
    if model_type.lower() == 'yolo':
        return YoloPoseEstimator(weights_path, None, device)
    elif model_type.lower() == 'mmpose':
        return MMPoseEstimator(weights_path, config_path, device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
