from pathlib import Path
from collections import namedtuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data structure for storing detection pairs
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])


def get_output_layers(net: cv2.dnn.Net) -> list[str]:
    """Retrieve output layer names in a way compatible with all OpenCV versions."""
    layer_names = net.getLayerNames()
    unconnected = net.getUnconnectedOutLayers()
    if len(unconnected.shape) == 1:
        return [layer_names[i - 1] for i in unconnected]
    return [layer_names[i[0] - 1] for i in unconnected]


def calculate_iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute Intersection over Union (IoU) for two bounding boxes [x1, y1, x2, y2]."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    union_area = float(box_a_area + box_b_area - inter_area)
    return inter_area / union_area if union_area > 0 else 0.0


def run_inference(
    config_path: Path,
    weights_path: Path,
    images_dir: Path,
    conf_threshold: float = 0.25,
    nms_threshold: float = 0.5,
) -> pd.DataFrame:
    """Run YOLOv3 inference over a directory of images and collect metric outputs."""
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError("YOLO configuration or weights file not found.")

    net = cv2.dnn.readNet(str(config_path), str(weights_path))
    output_layers = get_output_layers(net)
    image_paths = list(images_dir.glob("*.jpg"))

    results = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        height, width, _ = img.shape

        # Create input blob and forward pass
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

        # Parse output detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])

                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, conf_threshold, nms_threshold
        )

        # Dummy ground truth bounding box (center region: [25%w, 25%h, 75%w, 75%h])
        gt_box = [
            int(0.25 * width),
            int(0.25 * height),
            int(0.75 * width),
            int(0.75 * height),
        ]

        if len(indices) > 0:
            # Take top detection post-NMS
            idx = indices.flatten()[0]
            x, y, w, h = boxes[idx]
            pred_box = [x, y, x + w, y + h]
            iou = calculate_iou(gt_box, pred_box)
            max_conf = confidences[idx]

            results.append(
                [img_path.stem] + pred_box + gt_box + [max_conf, iou]
            )
        else:
            # Negative detection fallback
            results.append([img_path.stem] + [0] * 8 + [0.0, 0.0])

    columns = [
        "name",
        "plwa",
        "plha",
        "plwb",
        "plhb",
        "tlwa",
        "tlha",
        "tlwb",
        "tlhb",
        "con",
        "iou",
    ]
    df = pd.DataFrame(results, columns=columns)
    return df.drop_duplicates(subset=["name"], keep="first")


def compute_metrics(df: pd.DataFrame, iou_threshold: float = 0.5) -> pd.DataFrame:
    """Vectorized calculation of Precision-Recall metrics across sorted confidences."""
    df = df.sort_values(by="con", ascending=False).reset_index(drop=True)

    # Vectorized True Positive / False Positive assignment
    df["TP"] = np.where(df["iou"] >= iou_threshold, 1, 0)
    df["FP"] = np.where(df["iou"] < iou_threshold, 1, 0)

    # Cumulative sums for precision-recall curves
    df["Acc TP"] = df["TP"].cumsum()
    df["Acc FP"] = df["FP"].cumsum()

    total_instances = len(df)
    df["Precision"] = df["Acc TP"] / (df["Acc TP"] + df["Acc FP"])
    df["Recall"] = df["Acc TP"] / total_instances

    return df


def main():
    base_dir = Path(__file__).parent
    config_path = base_dir / "yolo_custom_detection" / "yolov3_testing.cfg"
    weights_path = base_dir / "yolo_custom_detection" / "yolov3_training_last.weights"
    images_dir = base_dir / "valid_b"

    try:
        df_results = run_inference(config_path, weights_path, images_dir)
        df_metrics = compute_metrics(df_results)

        # Compute Precision-Recall Area Under Curve (PR-AUC) using NumPy trapz
        pr_auc = np.trapz(df_metrics["Precision"], df_metrics["Recall"])
        print(f"Calculated PR-AUC: {abs(pr_auc):.4f}")

        # Export metrics
        output_csv = base_dir / "output_combined.csv"
        df_metrics.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv.resolve()}")

        # Plot Precision-Recall Curve
        plt.figure(figsize=(8, 5))
        plt.plot(
            df_metrics["Recall"],
            df_metrics["Precision"],
            color="red",
            label=f"AUC = {abs(pr_auc):.3f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

    except FileNotFoundError as e:
        print(f"Error running pipeline: {e}")


if __name__ == "__main__":
    main()
