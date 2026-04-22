import os
import time
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from pose_estimators import create_estimator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pose Estimators on COCO Val2017")
    parser.add_argument('--config', type=str, default='../configs/models.yaml', help='Path to models.yaml')
    parser.add_argument('--data-dir', type=str, default='../data/coco', help='COCO data directory')
    parser.add_argument('--output-dir', type=str, default='../outputs/e1', help='Where to save results')
    parser.add_argument('--num-samples', type=int, default=None, help='Limit evaluation to N samples (default: All)')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use: cpu, cuda:0, mps')
    return parser.parse_args()

def get_model_params(model):
    """Attempt to count parameters of the underlying torch model"""
    try:
        total_params = 0
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            total_params = sum(p.numel() for p in model.model.parameters())
        elif hasattr(model, 'inferencer') and hasattr(model.inferencer, 'model'): # For mmpose
             total_params = sum(p.numel() for p in model.inferencer.model.parameters())
        elif hasattr(model, 'parameters'):
             total_params = sum(p.numel() for p in model.parameters())
        return total_params / 1e6 # Return in Millions (M)
    except Exception as e:
        print(f"Warning: Could not count parameters: {e}")
        return 0.0

def evaluate_model(model_name: str, model_info: dict, args, coco_gt: COCO, img_ids: list):
    print(f"\n{'='*50}\nEvaluating: {model_name}\n{'='*50}")

    # 1. Load model
    print(f"Loading {model_info['type']} model...")
    st = time.time()
    estimator = create_estimator(
        model_type=model_info['type'],
        weights_path=model_info['weights'],
        config_path=model_info.get('config'),
        device=args.device
    )
    print(f"Loaded in {time.time() - st:.2f}s")

    params_m = get_model_params(estimator.model)
    print(f"Est. Parameters: {params_m:.2f} M")

    # 2. Inference Loop
    results = []
    fps_measurements = []

    print("Running inference...")
    # Warmup
    warmimg_info = coco_gt.loadImgs(img_ids[0])[0]
    warmup_img = cv2.imread(os.path.join(args.data_dir, 'val2017', warmimg_info['file_name']))
    estimator(warmup_img)

    for img_id in tqdm(img_ids, desc=f"{model_name} Inferencing"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(args.data_dir, 'val2017', img_info['file_name'])

        import cv2 # imported here just for safety if not in top
        img = cv2.imread(img_path)
        if img is None:
            continue

        # Timing start
        t0 = time.perf_counter()

        # Predict
        # Output: List of (17, 3) arrays
        persons = estimator(img)

        # Timing end
        t1 = time.perf_counter()
        fps_measurements.append(1.0 / (t1 - t0 + 1e-9))

        # Format for COCOeval
        for kps in persons:
            # Flatten to [x1, y1, v1, x2, y2, v2, ...]
            # v=2 (labeled and visible), 1 (labeled but not visible), 0 (not labeled).
            # We output conf, so we set v = conf
            flat_kps = kps.flatten().tolist()

            # Simple heuristic: compute score as mean confidence
            score = float(np.mean(kps[:, 2]))

            results.append({
                'image_id': img_id,
                'category_id': 1, # person
                'keypoints': flat_kps,
                'score': score
            })

    avg_fps = np.mean(fps_measurements)
    print(f"\nAverage FPS: {avg_fps:.2f} fps")

    # 3. Evaluation using pycocotools
    if len(results) == 0:
        print("No predictions generated!")
        return {'model': model_name, 'AP': 0, 'AR': 0, 'FPS': avg_fps, 'Params_M': params_m}

    # Save results to temp json (required by COCO api)
    temp_json = os.path.join(args.output_dir, f"temp_{model_name}_results.json")
    with open(temp_json, 'w') as f:
        json.dump(results, f)

    try:
        coco_dt = coco_gt.loadRes(temp_json)
        coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        map_score = coco_eval.stats[0] # AP @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]
        mar_score = coco_eval.stats[5] # AR @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]
    except Exception as e:
        print(f"COCO evaluation failed: {e}")
        map_score, mar_score = 0.0, 0.0

    # Cleanup temp file
    if os.path.exists(temp_json):
        os.remove(temp_json)

    return {
        'model': model_name,
        'AP': map_score * 100, # Convert to percentage
        'AR': mar_score * 100,
        'FPS': avg_fps,
        'Params_M': params_m
    }

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Models Config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    models_cfg = config.get('models', {})

    # Load COCO GT
    ann_file = os.path.join(args.data_dir, 'annotations/person_keypoints_val2017.json')
    print(f"Loading COCO Ground Truth from: {ann_file}")
    coco_gt = COCO(ann_file)

    # Get images containing persons
    img_ids = coco_gt.getImgIds(catIds=coco_gt.getCatIds(catNms=['person']))
    if args.num_samples:
        img_ids = img_ids[:args.num_samples]
        print(f"Evaluating on subset of {args.num_samples} images for testing.")
    else:
        print(f"Evaluating on total {len(img_ids)} images.")

    # Loop models
    summary_results = []
    for model_name, model_info in models_cfg.items():
        res = evaluate_model(model_name, model_info, args, coco_gt, img_ids)
        summary_results.append(res)

    # Save Final Table
    df = pd.DataFrame(summary_results)
    csv_path = os.path.join(args.output_dir, 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nFinal Results saved to {csv_path}")
    print(df.to_markdown(index=False))

if __name__ == '__main__':
    main()
