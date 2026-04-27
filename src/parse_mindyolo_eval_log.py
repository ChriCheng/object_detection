import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--out_file", required=True)
    return parser.parse_args()


def find_metric(text, metric_type, iou, area="all", max_dets=None):
    """
    Example:
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
    """
    if max_dets is None:
        pattern = (
            rf"Average {metric_type}\s+\((AP|AR)\)\s+@\[\s*"
            rf"IoU={re.escape(iou)}\s*\|\s*area=\s*{area}\s*\|\s*maxDets=\s*\d+\s*\]\s*=\s*([0-9.]+)"
        )
    else:
        pattern = (
            rf"Average {metric_type}\s+\((AP|AR)\)\s+@\[\s*"
            rf"IoU={re.escape(iou)}\s*\|\s*area=\s*{area}\s*\|\s*maxDets=\s*{max_dets}\s*\]\s*=\s*([0-9.]+)"
        )

    m = re.search(pattern, text)
    return float(m.group(2)) if m else None


def main():
    args = parse_args()
    log_path = Path(args.log_file)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = log_path.read_text(encoding="utf-8", errors="ignore")

    metrics = {
        "source_log": str(log_path),
        "map_50_95": find_metric(text, "Precision", "0.50:0.95", "all", 100),
        "map_50": find_metric(text, "Precision", "0.50", "all", 100),
        "map_75": find_metric(text, "Precision", "0.75", "all", 100),
        "map_small": find_metric(text, "Precision", "0.50:0.95", "small", 100),
        "map_medium": find_metric(text, "Precision", "0.50:0.95", "medium", 100),
        "map_large": find_metric(text, "Precision", "0.50:0.95", "large", 100),
        "ar_1": find_metric(text, "Recall", "0.50:0.95", "all", 1),
        "ar_10": find_metric(text, "Recall", "0.50:0.95", "all", 10),
        "ar_100": find_metric(text, "Recall", "0.50:0.95", "all", 100),
        "ar_small": find_metric(text, "Recall", "0.50:0.95", "small", 100),
        "ar_medium": find_metric(text, "Recall", "0.50:0.95", "medium", 100),
        "ar_large": find_metric(text, "Recall", "0.50:0.95", "large", 100),
        "speed_inference_ms": None,
        "speed_nms_ms": None,
        "speed_total_ms": None,
        "fps_total": None,
        "fps_inference_only": None,
    }

    speed_match = re.search(
        r"Speed:\s*([0-9.]+)/([0-9.]+)/([0-9.]+)\s*ms inference/NMS/total",
        text
    )
    if speed_match:
        inf_ms = float(speed_match.group(1))
        nms_ms = float(speed_match.group(2))
        total_ms = float(speed_match.group(3))
        metrics["speed_inference_ms"] = inf_ms
        metrics["speed_nms_ms"] = nms_ms
        metrics["speed_total_ms"] = total_ms
        metrics["fps_total"] = round(1000.0 / total_ms, 2)
        metrics["fps_inference_only"] = round(1000.0 / inf_ms, 2)

    out_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
