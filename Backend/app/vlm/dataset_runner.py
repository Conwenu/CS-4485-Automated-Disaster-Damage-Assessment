import json
import os
import time
from collections import Counter

from app.services.house_crop_adapter import load_normalized_building_pairs_with_issues
from app.vlm.building_service import assess_building_damage_with_retry

RESULTS_PATH = os.path.join("data", "santa_rosa", "building_results.json")
EVALUATION_PATH = os.path.join("data", "santa_rosa", "building_evaluation.json")
SUMMARY_PATH = os.path.join("data", "santa_rosa", "building_summary.json")
CROP_ROOT = os.path.join("data", "santa_rosa", "building_crops")


def load_existing_results(results_path):
    if not os.path.exists(results_path):
        return []

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: invalid JSON in {results_path}")
        return []

    if isinstance(data, list):
        return data

    return []


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp"

    def write_temp_file():
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    write_temp_file()

    for _ in range(3):
        try:
            os.replace(temp_path, path)
            return
        except (PermissionError, FileNotFoundError):
            # Some Windows setups briefly lose the temp file during replace.
            if not os.path.exists(temp_path):
                write_temp_file()
            time.sleep(0.2)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    if os.path.exists(temp_path):
        os.remove(temp_path)


def upsert_result(results, item):
    item_id = item.get("id")
    if not item_id:
        results.append(item)
        return

    for index, existing in enumerate(results):
        if isinstance(existing, dict) and existing.get("id") == item_id:
            results[index] = item
            return

    results.append(item)


def build_evaluation_records(results):
    evaluation = []

    for item in results:
        if item.get("status") != "ok":
            continue

        model = item.get("model", {})
        eval_data = item.get("evaluation", {})
        evaluation.append({
            "id": item.get("id"),
            "scene_id": item.get("scene_id"),
            "city": item.get("city"),
            "ground_truth": eval_data.get("ground_truth"),
            "prediction": eval_data.get("prediction"),
            "match": eval_data.get("match"),
            "confidence": model.get("confidence"),
            "reasoning": model.get("reasoning"),
        })

    return evaluation


def build_summary(evaluation):
    total = len(evaluation)
    matched = 0
    ground_truth_counts = Counter()
    prediction_counts = Counter()
    confusion_counts = Counter()
    mismatches = []

    for item in evaluation:
        ground_truth = item.get("ground_truth")
        prediction = item.get("prediction")
        match = item.get("match")

        if match is True:
            matched += 1

        if ground_truth:
            ground_truth_counts[ground_truth] += 1
        if prediction:
            prediction_counts[prediction] += 1
        if ground_truth and prediction:
            confusion_counts[f"{ground_truth} -> {prediction}"] += 1

        if match is False:
            mismatches.append({
                "id": item.get("id"),
                "scene_id": item.get("scene_id"),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "confidence": item.get("confidence"),
                "reasoning": item.get("reasoning"),
            })

    accuracy = 0.0
    if total:
        accuracy = matched / total

    return {
        "total": total,
        "matched": matched,
        "accuracy": round(accuracy, 4),
        "ground_truth_counts": dict(ground_truth_counts),
        "prediction_counts": dict(prediction_counts),
        "confusion_counts": dict(confusion_counts),
        "mismatches": mismatches,
    }


def run_dataset(
    image_root="data/santa_rosa/images",
    crop_root=CROP_ROOT,
    label_root=None,
    results_path=RESULTS_PATH,
    evaluation_path=EVALUATION_PATH,
    summary_path=SUMMARY_PATH,
    scene_id=None,
    limit=None,
    progress_every=25,
):
    valid_pairs, invalid_pairs = load_normalized_building_pairs_with_issues(
        image_root=image_root,
        crop_root=crop_root,
        scene_id=scene_id,
        label_root=label_root,
    )

    results = load_existing_results(results_path)
    processed_ids = set()
    for item in results:
        if isinstance(item, dict) and item.get("id") and item.get("status") == "ok":
            processed_ids.add(item.get("id"))

    skipped_done = 0
    failed = 0
    completed_new = 0

    selected_pairs = valid_pairs
    if isinstance(limit, int) and limit > 0:
        selected_pairs = valid_pairs[:limit]

    total_selected = len(selected_pairs)

    for index, pair in enumerate(selected_pairs, start=1):
        pair_id = pair.get("id")
        if not pair_id:
            continue

        if progress_every and ((index == 1) or (index % progress_every == 0) or (index == total_selected)):
            print(f"Progress {index}/{total_selected} current={pair_id}")

        if pair_id in processed_ids:
            skipped_done += 1
            continue

        pre_image_path = pair.get("pre_image_path")
        post_image_path = pair.get("post_image_path")
        ground_truth = pair.get("ground_truth")

        if not isinstance(pre_image_path, str) or not isinstance(post_image_path, str):
            failed += 1
            continue

        try:
            prediction = assess_building_damage_with_retry(
                pre_image_path,
                post_image_path,
                ground_truth=ground_truth,
            )
            upsert_result(results, {
                "id": pair_id,
                "scene_id": pair.get("scene_id"),
                "city": pair.get("city"),
                "status": "ok",
                "pre_image_path": pre_image_path,
                "post_image_path": post_image_path,
                **prediction,
            })
            completed_new += 1
        except Exception as error:
            failed += 1
            upsert_result(results, {
                "id": pair_id,
                "scene_id": pair.get("scene_id"),
                "city": pair.get("city"),
                "status": "failed",
                "error": str(error),
                "pre_image_path": pre_image_path,
                "post_image_path": post_image_path,
                "ground_truth": ground_truth,
            })

        processed_ids.add(pair_id)
        save_json(results_path, results)

    for item in invalid_pairs:
        pair = item.get("pair", {})
        pair_id = pair.get("id")

        if pair_id and pair_id in processed_ids:
            continue

        upsert_result(results, {
            "id": pair_id,
            "scene_id": pair.get("scene_id"),
            "status": "invalid_pair",
            "issues": item.get("issues", []),
            "pair": pair,
        })

        if pair_id:
            processed_ids.add(pair_id)

        save_json(results_path, results)

    evaluation = build_evaluation_records(results)
    summary = build_summary(evaluation)
    save_json(evaluation_path, evaluation)
    save_json(summary_path, summary)

    print(
        "Done "
        f"new_ok={completed_new} "
        f"already_done={skipped_done} "
        f"failed={failed} "
        f"invalid_pairs={len(invalid_pairs)} "
        f"selected={total_selected}"
    )
    print(f"Saved results: {results_path}")
    print(f"Saved evaluation: {evaluation_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Accuracy: {summary['matched']}/{summary['total']}")


if __name__ == "__main__":
    run_dataset()
