from pathlib import Path
import numpy as np


def compute_batch_iou(
    boxes_a: np.ndarray, boxes_b: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    """Calculate Intersection over Union (IoU) scores for pairs of bounding boxes.

    Args:
        boxes_a (np.ndarray): Array of shape (N, 4) with bounding boxes [x1, y1, x2, y2].
        boxes_b (np.ndarray): Array of shape (N, 4) with bounding boxes [x1, y1, x2, y2].
        epsilon (float, optional): Small constant to avoid division by zero. Defaults to 1e-5.

    Returns:
        np.ndarray: Vector of shape (N,) containing IoU scores ranging from 0.0 to 1.0.
    """
    # 1. Coordinates of intersection rectangles
    x1 = np.maximum(boxes_a[:, 0], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3], boxes_b[:, 3])

    # 2. Compute intersection areas (clip negative values where boxes do not overlap)
    intersection_width = np.maximum(0.0, x2 - x1)
    intersection_height = np.maximum(0.0, y2 - y1)
    area_intersection = intersection_width * intersection_height

    # 3. Compute individual box areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # 4. Compute union area
    area_union = area_a + area_b - area_intersection

    # 5. Calculate IoU ratio
    iou = area_intersection / (area_union + epsilon)
    return iou


def main():
    # Define relative paths for data portability
    data_dir = Path(__file__).parent / "data"
    labels_path = data_dir / "labels.csv"
    predictions_path = data_dir / "predictions.csv"

    if not labels_path.exists() or not predictions_path.exists():
        print(f"Data files not found in: {data_dir.resolve()}")
        return

    # Load arrays directly as float/int
    labels = np.loadtxt(labels_path, delimiter=",", dtype=float)
    predictions = np.loadtxt(predictions_path, delimiter=",", dtype=float)

    # Compute batch IoUs
    iou_scores = compute_batch_iou(labels, predictions)

    print("Batch IoU Calculation Complete:")
    print(f"Mean IoU: {np.mean(iou_scores):.4f}")
    print(f"Sample IoUs: {iou_scores[:5]}")


if __name__ == "__main__":
    main()
