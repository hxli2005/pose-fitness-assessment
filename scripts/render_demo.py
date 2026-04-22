import os
import cv2
import yaml
import argparse
import numpy as np
from tqdm import tqdm

from pose_estimators import create_estimator
from joint_angles import extract_key_angles, KP
from rep_counter import RepCounter
from fitness_scorer import FitnessScorer

# COCO 17-keypoint skeleton connections
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Arms & Shoulders
    (5, 11), (6, 12), (11, 12),              # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)   # Legs
]

def draw_skeleton(img, kps, conf_threshold=0.3):
    """Draw keypoints and skeleton connections on the image."""
    # Draw connections
    for u, v in SKELETON:
        if kps[u, 2] > conf_threshold and kps[v, 2] > conf_threshold:
            pt1 = (int(kps[u, 0]), int(kps[u, 1]))
            pt2 = (int(kps[v, 0]), int(kps[v, 1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw points
    for i, (x, y, conf) in enumerate(kps):
        if conf > conf_threshold:
            cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)
    return img

def parse_args():
    parser = argparse.ArgumentParser(description="Render Fitness Assessment Demo Video")
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='outputs/demos/demo.mp4', help='Path to output video')
    parser.add_argument('--model', type=str, default='rtmpose_s', help='Model key from models.yaml')
    parser.add_argument('--action', type=str, default='squat', choices=['squat', 'pushup', 'crunch'])
    parser.add_argument('--config', type=str, default='configs/models.yaml', help='Path to models.yaml')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load Model Configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    model_info = config['models'][args.model]

    print(f"Loading {args.model} estimator...")
    estimator = create_estimator(
        model_type=model_info['type'],
        weights_path=model_info['weights'],
        config_path=model_info.get('config'),
        device=args.device
    )

    # Initialize Logic Modules
    counter = RepCounter(action_type=args.action)
    scorer = FitnessScorer(action_type=args.action)

    # Open Video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.input}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Processing {total_frames} frames ({width}x{height} @ {fps}fps)")

    last_score = {"standard": True, "errors": []}
    rep_history = []

    for _ in tqdm(range(total_frames), desc="Rendering..."):
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Pose Estimation
        # (Assuming only 1 person in frame for this strict fitness demo)
        persons = estimator(frame)
        if len(persons) == 0:
            # If no person detected, just write the original frame
            out.write(frame)
            continue

        kps = persons[0] # Get the first person

        # 2. Extract angles
        angles = extract_key_angles(kps)

        # 3. Update Counter State Machine
        counter_status = counter.update(angles)

        # 4. If a rep just finished, Score it!
        if counter_status["rep_finished"]:
            last_score = scorer.score_rep(counter_status["trajectory"])
            rep_history.append({
                "count": counter_status["count"],
                "score": last_score
            })

        # --- VISUALIZATION --- #
        # Draw Skeleton
        frame = draw_skeleton(frame, kps)

        # Draw specific angle depending on action
        if args.action == "squat" and angles.get('left_knee') is not None:
            lk = angles['left_knee']
            lx, ly = kps[KP['left_knee']][:2]
            cv2.putText(frame, f"{int(lk)} deg", (int(lx)-50, int(ly)-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # UI Overlay (Top Left Panel)
        cv2.rectangle(frame, (10, 10), (450, 160), (0, 0, 0), -1)

        # Display Counter
        cv2.putText(frame, f"Action: {args.action.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"REPS: {counter.count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        # Display Last Score
        if last_score["standard"]:
            color = (0, 255, 0)
            msg = "Feedback: PERFECT"
        else:
            color = (0, 0, 255)
            msg = "Feedback: " + " | ".join(last_score["errors"])

        # Display truncated message if too long
        if len(msg) > 35: msg = msg[:32] + "..."
        cv2.putText(frame, msg, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # State machine indicator
        phase = "GOING DOWN" if counter.in_rep else "UP"
        cv2.putText(frame, f"Phase: {phase}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Demo video saved to: {args.output}")

if __name__ == '__main__':
    main()
