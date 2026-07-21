from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "headword-model.json"
TOKEN = re.compile(r"[^a-zåäöàáé-]+", re.IGNORECASE)


@dataclass(frozen=True)
class WordObservation:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float
    ink_density: float
    line_left: float
    relative_height: float
    ocr_tesseract: str = ""
    ocr_runeberg: str = ""
    ocr_conflict: bool = False

    @property
    def features(self) -> list[float]:
        return [
            1.0,
            self.ink_density,
            self.line_left,
            self.relative_height,
            max(0.0, min(1.0, self.confidence / 100.0)),
            min(len(self.text), 24) / 24.0,
        ]


def normalize_token(text: str) -> str:
    return TOKEN.sub("", text.casefold().strip())


def sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


@dataclass
class HeadwordModel:
    weights: list[float]
    means: list[float]
    scales: list[float]
    samples: int
    positive_samples: int

    def probability(self, observation: WordObservation) -> float:
        values = observation.features
        standardized = [
            1.0 if index == 0 else (value - self.means[index]) / self.scales[index]
            for index, value in enumerate(values)
        ]
        return sigmoid(sum(weight * value for weight, value in zip(self.weights, standardized)))

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "HeadwordModel | None":
        if not MODEL_PATH.exists():
            return None
        return cls(**json.loads(MODEL_PATH.read_text(encoding="utf-8")))


def train_model(samples: list[tuple[WordObservation, int]]) -> HeadwordModel:
    if len(samples) < 40:
        raise ValueError("För få träningsord. Godkänn fler av sidorna 19–28 först.")
    positives = sum(label for _, label in samples)
    if positives < 10:
        raise ValueError("För få markerade uppslagsord i träningsmaterialet.")

    raw = [observation.features for observation, _ in samples]
    columns = list(zip(*raw))
    means = [0.0] + [sum(column) / len(column) for column in columns[1:]]
    scales = [1.0]
    for index, column in enumerate(columns[1:], start=1):
        variance = sum((value - means[index]) ** 2 for value in column) / len(column)
        scales.append(max(math.sqrt(variance), 1e-6))

    weights = [0.0] * len(raw[0])
    rate = 0.08
    for _ in range(1200):
        gradient = [0.0] * len(weights)
        for values, (_, label) in zip(raw, samples):
            standardized = [
                1.0 if index == 0 else (value - means[index]) / scales[index]
                for index, value in enumerate(values)
            ]
            error = sigmoid(sum(w * v for w, v in zip(weights, standardized))) - label
            for index, value in enumerate(standardized):
                gradient[index] += error * value
        for index in range(len(weights)):
            weights[index] -= rate * gradient[index] / len(samples)

    return HeadwordModel(weights, means, scales, len(samples), positives)
