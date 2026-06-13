import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.geometry.geometry_utils import bbox_iou


DETECTION_TYPES = {"door", "window"}
QUANTITY_CATEGORIES = {
    "room_area",
    "room_perimeter",
    "wall_length",
    "wall_length_total",
    "door_count",
    "window_count",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pipeline predictions against ground truth.")
    parser.add_argument("--pred", required=True, type=Path, help="Predicted merged_objects.json or combined prediction JSON.")
    parser.add_argument("--gt", required=True, type=Path, help="Ground-truth objects JSON or combined ground-truth JSON.")
    parser.add_argument("--pred-quantities", type=Path, default=None, help="Optional predicted quantities.json.")
    parser.add_argument("--gt-quantities", type=Path, default=None, help="Optional ground-truth quantities.json.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for evaluation_report.json/md.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred_payload = _read_json(args.pred)
    gt_payload = _read_json(args.gt)
    pred_objects = _extract_objects(pred_payload)
    gt_objects = _extract_objects(gt_payload)
    pred_quantities = _extract_quantities(pred_payload)
    gt_quantities = _extract_quantities(gt_payload)

    if args.pred_quantities is not None:
        pred_quantities = _extract_quantities(_read_json(args.pred_quantities))
    else:
        sibling = args.pred.with_name("quantities.json")
        if sibling.exists():
            pred_quantities = _extract_quantities(_read_json(sibling))
    if args.gt_quantities is not None:
        gt_quantities = _extract_quantities(_read_json(args.gt_quantities))
    else:
        sibling = args.gt.with_name("example_quantities.json")
        if sibling.exists():
            gt_quantities = _extract_quantities(_read_json(sibling))

    report = evaluate(pred_objects, gt_objects, pred_quantities, gt_quantities, args.iou_threshold)
    output_dir = args.output or args.pred.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_report.json"
    md_path = output_dir / "evaluation_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


def evaluate(
    pred_objects: list[dict[str, Any]],
    gt_objects: list[dict[str, Any]],
    pred_quantities: list[dict[str, Any]] | None = None,
    gt_quantities: list[dict[str, Any]] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    detection_metrics = _detection_metrics(pred_objects, gt_objects, iou_threshold)
    quantity_metrics = _quantity_metrics(pred_quantities or [], gt_quantities or [])
    return {
        "iou_threshold": iou_threshold,
        "detection": detection_metrics,
        "quantities": quantity_metrics,
        "summary": {
            "predicted_objects": len(pred_objects),
            "ground_truth_objects": len(gt_objects),
            "predicted_quantities": len(pred_quantities or []),
            "ground_truth_quantities": len(gt_quantities or []),
        },
    }


def _detection_metrics(pred_objects: list[dict[str, Any]], gt_objects: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    by_class: dict[str, Any] = {}
    all_matches = []
    for object_type in sorted(DETECTION_TYPES):
        preds = [item for item in pred_objects if item.get("type") == object_type and item.get("geometry_type") == "bbox"]
        gts = [item for item in gt_objects if item.get("type") == object_type and item.get("geometry_type") == "bbox"]
        matches = _match_bboxes(preds, gts, iou_threshold)
        true_positive = len(matches)
        false_positive = len(preds) - true_positive
        false_negative = len(gts) - true_positive
        precision = _safe_div(true_positive, true_positive + false_positive)
        recall = _safe_div(true_positive, true_positive + false_negative)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        avg_iou = _safe_div(sum(match["iou"] for match in matches), len(matches))
        by_class[object_type] = {
            "predicted": len(preds),
            "ground_truth": len(gts),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "count_error": len(preds) - len(gts),
            "average_iou": round(avg_iou, 6),
            "matches": matches,
        }
        all_matches.extend(matches)

    total_tp = sum(item["true_positive"] for item in by_class.values())
    total_fp = sum(item["false_positive"] for item in by_class.values())
    total_fn = sum(item["false_negative"] for item in by_class.values())
    precision = _safe_div(total_tp, total_tp + total_fp)
    recall = _safe_div(total_tp, total_tp + total_fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "overall": {
            "true_positive": total_tp,
            "false_positive": total_fp,
            "false_negative": total_fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "average_iou": round(_safe_div(sum(match["iou"] for match in all_matches), len(all_matches)), 6),
        },
        "by_class": by_class,
    }


def _match_bboxes(preds: list[dict[str, Any]], gts: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    candidates = []
    for pred_index, pred in enumerate(preds):
        for gt_index, gt in enumerate(gts):
            iou = bbox_iou(pred["geometry"], gt["geometry"])
            if iou >= iou_threshold:
                candidates.append((iou, pred_index, gt_index))
    candidates.sort(reverse=True)
    used_preds = set()
    used_gts = set()
    matches = []
    for iou, pred_index, gt_index in candidates:
        if pred_index in used_preds or gt_index in used_gts:
            continue
        used_preds.add(pred_index)
        used_gts.add(gt_index)
        matches.append(
            {
                "pred_id": preds[pred_index].get("id"),
                "gt_id": gts[gt_index].get("id"),
                "iou": round(iou, 6),
            }
        )
    return matches


def _quantity_metrics(pred_quantities: list[dict[str, Any]], gt_quantities: list[dict[str, Any]]) -> dict[str, Any]:
    pred_lookup = _quantity_lookup(pred_quantities)
    gt_lookup = _quantity_lookup(gt_quantities)
    keys = sorted(set(pred_lookup) | set(gt_lookup))
    items = []
    by_category: dict[str, list[float]] = {}
    for key in keys:
        category, name = key
        pred = pred_lookup.get(key)
        gt = gt_lookup.get(key)
        pred_value = float(pred["quantity"]) if pred else None
        gt_value = float(gt["quantity"]) if gt else None
        absolute_error = abs(pred_value - gt_value) if pred_value is not None and gt_value is not None else None
        percentage_error = _safe_div(absolute_error, abs(gt_value)) * 100 if absolute_error is not None and gt_value not in (None, 0.0) else None
        if absolute_error is not None:
            by_category.setdefault(category, []).append(absolute_error)
        items.append(
            {
                "category": category,
                "name": name,
                "predicted": pred_value,
                "ground_truth": gt_value,
                "unit": (pred or gt or {}).get("unit"),
                "absolute_error": round(absolute_error, 6) if absolute_error is not None else None,
                "percentage_error": round(percentage_error, 6) if percentage_error is not None else None,
                "missing_prediction": pred is None,
                "missing_ground_truth": gt is None,
            }
        )
    return {
        "items": items,
        "by_category": {
            category: {
                "count": len(errors),
                "mean_absolute_error": round(_safe_div(sum(errors), len(errors)), 6),
            }
            for category, errors in sorted(by_category.items())
        },
    }


def _quantity_lookup(items: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup = {}
    category_counts: dict[str, int] = {}
    for item in items:
        category = str(item.get("category") or "")
        if category not in QUANTITY_CATEGORIES:
            continue
        name = str(item.get("name") or category)
        if category in {"door_count", "window_count", "wall_length_total"}:
            name = category
        elif category in {"room_area", "room_perimeter"}:
            name = _normalize_room_quantity_name(name, category)
        elif category == "wall_length":
            category_counts[category] = category_counts.get(category, 0) + 1
            name = name or f"Wall {category_counts[category]}"
        lookup[(category, name)] = item
    return lookup


def _normalize_room_quantity_name(name: str, category: str) -> str:
    lowered = name.lower()
    suffix = " area" if category == "room_area" else " perimeter"
    if lowered.endswith(suffix):
        return name[: -len(suffix)]
    return name


def _extract_objects(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict) and "geometry_type" in item]
    if isinstance(payload, dict):
        objects = payload.get("objects") or payload.get("merged_objects") or payload.get("detections") or []
        return [item for item in objects if isinstance(item, dict)]
    return []


def _extract_quantities(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict) and "category" in item and "quantity" in item]
    if isinstance(payload, dict):
        quantities = payload.get("quantities") or []
        return [item for item in quantities if isinstance(item, dict)]
    return []


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- IoU threshold: `{report['iou_threshold']}`",
        f"- Predicted objects: `{report['summary']['predicted_objects']}`",
        f"- Ground-truth objects: `{report['summary']['ground_truth_objects']}`",
        "",
        "## Detection",
    ]
    overall = report["detection"]["overall"]
    lines.extend(
        [
            f"- Precision: `{overall['precision']}`",
            f"- Recall: `{overall['recall']}`",
            f"- F1: `{overall['f1']}`",
            f"- Average IoU: `{overall['average_iou']}`",
            f"- False positives: `{overall['false_positive']}`",
            f"- False negatives: `{overall['false_negative']}`",
            "",
            "### By Class",
        ]
    )
    for object_type, metrics in report["detection"]["by_class"].items():
        lines.append(
            f"- {object_type}: precision `{metrics['precision']}`, recall `{metrics['recall']}`, F1 `{metrics['f1']}`, count error `{metrics['count_error']}`"
        )
    lines.extend(["", "## Quantities"])
    if report["quantities"]["items"]:
        for item in report["quantities"]["items"]:
            lines.append(
                f"- {item['category']} / {item['name']}: pred `{item['predicted']}`, gt `{item['ground_truth']}`, abs error `{item['absolute_error']}`, pct error `{item['percentage_error']}`"
            )
    else:
        lines.append("- No quantity ground truth provided.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
