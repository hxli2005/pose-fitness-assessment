import numpy as np

class FitnessScorer:
    """
    Evaluates the quality of a completed repetition based on a list of its frame-by-frame angles.
    This module uses specific biomechanical rules (e.g., knee > 90deg, body straight < 15deg)
    to classify common mistakes in fitness actions.
    """

    def __init__(self, action_type="squat"):
        self.action_type = action_type.lower()
        # Define the acceptable threshold bounds per action
        self.rules = {
            "squat": {
                "min_knee_angle": (70, 90), # Ideal deep squat knee angle range at lowest point
                "max_valgus_diff": 15       # Max expected asymmetry between left/right knee in deg
            },
            "pushup": {
                "min_elbow_angle": (70, 90), # Ideal pushup depth
                "max_body_deviation": 20     # Max deviation from straight 180 degrees
            },
            "crunch": {
                "min_hip_angle": (45, 70),   # Range typical for crunches compression
            }
        }

        if self.action_type not in self.rules:
            raise ValueError(f"No scoring rules available for: {self.action_type}")

    def score_rep(self, trajectory_angles: list) -> dict:
        """
        Takes a list of angle dictionaries collected during one repetition (from entry to exit).
        evaluates if it meets the criteria for a "standard" rep, and logs any identified errors.

        Returns:
            {"standard": True/False, "errors": list_of_strings}
        """
        if not trajectory_angles:
            return {"standard": False, "errors": ["No data available"]}

        eval_result = {"standard": True, "errors": []}
        rule = self.rules[self.action_type]

        if self.action_type == "squat":
            # Extract knee angles sequence
            left_knees = [a.get('left_knee', 180) for a in trajectory_angles if a.get('left_knee')]
            right_knees = [a.get('right_knee', 180) for a in trajectory_angles if a.get('right_knee')]

            # 1. Depth Check (Did they go low enough?)
            min_kl = min(left_knees) if left_knees else 180
            min_kr = min(right_knees) if right_knees else 180
            overall_min_knee_angle = min(min_kl, min_kr)

            if overall_min_knee_angle > rule["min_knee_angle"][1]: # > 90
                eval_result["standard"] = False
                eval_result["errors"].append("Half-rep (Didn't squat low enough)")

            # 2. Symmetry Check (Valgus / Knee cave in)
            if left_knees and right_knees:
                # Compare symmetry at the bottom of the squat
                idx_lowest = min(len(left_knees), len(right_knees))
                # Grab angles when they are around the minimum
                diffs = [abs(lk - rk) for lk, rk in zip(left_knees[:idx_lowest], right_knees[:idx_lowest])]
                # If peak difference during squat phase is very high
                if max(diffs) > rule["max_valgus_diff"]:
                    eval_result["standard"] = False
                    eval_result["errors"].append("Knee valgus (Asymmetrical knee movement)")


        elif self.action_type == "pushup":
            # 1. Depth check
            left_el = [a.get('left_elbow', 180) for a in trajectory_angles if a.get('left_elbow')]
            right_el = [a.get('right_elbow', 180) for a in trajectory_angles if a.get('right_elbow')]

            min_el = min(left_el) if left_el else 180
            min_er = min(right_el) if right_el else 180
            overall_min_elbow_angle = min(min_el, min_er)

            if overall_min_elbow_angle > rule["min_elbow_angle"][1]: # > 90
                eval_result["standard"] = False
                eval_result["errors"].append("Half-rep (Didn't go low enough)")

            # 2. Body straightness check (Sagging/Piked hips)
            # Normally requires evaluating the shoulder-hip-ankle alignment calculated outside.
            # But the angles dict given here depends on implementation of joint_angles.py
            body_st_l = [a.get('left_torso', 180) for a in trajectory_angles if a.get('left_torso')]
            body_st_r = [a.get('right_torso', 180) for a in trajectory_angles if a.get('right_torso')]

            min_tl = min(body_st_l) if body_st_l else 180
            min_tr = min(body_st_r) if body_st_r else 180

            # If the torso bends (away from 180) by more than max_body_deviation
            if (180 - min_tl) > rule["max_body_deviation"] or (180 - min_tr) > rule["max_body_deviation"]:
                eval_result["standard"] = False
                eval_result["errors"].append("Sagging hips / Bad core alignment")

        elif self.action_type == "crunch":
            # 1. Compression depth
            left_hip = [a.get('left_hip', 180) for a in trajectory_angles if a.get('left_hip')]
            min_lh = min(left_hip) if left_hip else 180

            if min_lh > rule["min_hip_angle"][1]: # > 70 implies torso didn't lift up sufficiently
                eval_result["standard"] = False
                eval_result["errors"].append("Insufficient magnitude (Didn't rise high enough)")

        return eval_result
