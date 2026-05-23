from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

CLASS_NAMES: list[str] = [
    "01_palm",
    "02_l",
    "03_fist",
    "04_fist_moved",
    "05_thumb",
    "06_index",
    "07_ok",
    "08_palm_moved",
    "09_c",
    "10_down",
]


@dataclass(frozen=True)
class PredictionResult:
    image_path: Path
    predicted_label: str
    confidence: float


class GestureModelTester:
    def __init__(self, model_path: Path, image_size: int = 224) -> None:
        self.model_path = model_path
        self.image_size = image_size
        self.model = self._load_model()
        self.class_names = CLASS_NAMES
        self._validate_output_classes()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        return load_model(self.model_path)

    def _validate_output_classes(self) -> None:
        output_shape = self.model.output_shape
        if not isinstance(output_shape, tuple) or len(output_shape) < 2:
            raise ValueError(f"Unexpected model output shape: {output_shape}")

        class_count = output_shape[-1]
        if class_count != len(self.class_names):
            raise ValueError(
                "The model output classes do not match CLASS_NAMES. "
                f"Model has {class_count} outputs, but CLASS_NAMES has {len(self.class_names)} labels."
            )

    def collect_samples(
        self,
        image_dir: Path,
        sample_count: int,
        seed: int,
    ) -> list[Path]:
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        image_paths = self._list_image_files(image_dir)
        if not image_paths:
            raise ValueError(f"No images were found in {image_dir}")

        rng = random.Random(seed)
        take_count = min(sample_count, len(image_paths))
        return rng.sample(image_paths, take_count)

    def predict_image(self, image_path: Path) -> PredictionResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, (self.image_size, self.image_size))
        normalized = resized.astype(np.float32) / 255.0
        batch = np.expand_dims(normalized, axis=0)

        probabilities = self.model.predict(batch, verbose=0)[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

        return PredictionResult(
            image_path=image_path,
            predicted_label=self.class_names[predicted_index],
            confidence=confidence,
        )

    def predict_samples(self, samples: Sequence[Path]) -> list[PredictionResult]:
        return [self.predict_image(image_path) for image_path in samples]

    @staticmethod
    def _list_image_files(class_dir: Path) -> list[Path]:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in image_extensions
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the trained hand gesture model on a few images from a folder.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("hand_gesture_model.h5"),
        help="Path to the trained Keras model file.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("09"),
        help="Path to a folder containing sample images. Defaults to the local 09 folder.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=4,
        help="Number of images to sample from the folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when selecting sample images.",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        default=None,
        help="Optional path to save the results grid as an image.",
    )
    return parser.parse_args()


def print_results(results: Iterable[PredictionResult]) -> float:
    results = list(results)

    print("\nPrediction results")
    print("-" * 80)
    for result in results:
        print(
            f"pred={result.predicted_label:15} | confidence={result.confidence:.3f} | {result.image_path.name}"
        )

    print("-" * 80)
    print(f"Evaluated {len(results)} images")
    return 0.0


def plot_results(results: Sequence[PredictionResult], save_path: Path | None = None) -> None:
    if not results:
        return

    columns = 3
    rows = int(np.ceil(len(results) / columns))
    plt.figure(figsize=(5 * columns, 5 * rows))

    for index, result in enumerate(results, start=1):
        image = cv2.imread(str(result.image_path))
        if image is None:
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        title = (
            f"P: {result.predicted_label} ({result.confidence:.2f})\n"
            f"{result.image_path.name}"
        )

        plt.subplot(rows, columns, index)
        plt.imshow(rgb_image)
        plt.title(title, fontsize=10)
        plt.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def main() -> None:
    args = parse_args()
    tester = GestureModelTester(model_path=args.model)
    samples = tester.collect_samples(
        image_dir=args.image_dir,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    results = tester.predict_samples(samples)
    print_results(results)
    plot_results(results, save_path=args.save_figure)


if __name__ == "__main__":
    main()
