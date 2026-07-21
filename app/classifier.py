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
    feature_count = len(raw[0])
    means = [0.0] * feature_count
    scales = [1.0] * feature_count
    for index in range(1, feature_count):
        column = [row[index] for row in raw]
        means[index] = sum(column) / len(column)
        variance = sum((value - means[index]) ** 2 for value in column) / len(column)
        scales[index] = max(math.sqrt(variance), 1e-6)

    vectors = [
        [1.0 if index == 0 else (value - means[index]) / scales[index] for index, value in enumerate(row)]
        for row in raw
    ]
    labels = [label for _, label in samples]
    weights = [0.0] * feature_count
    positive_weight = max(1.0, (len(labels) - positives) / positives)

    for step in range(1400):
        gradients = [0.0] * feature_count
        learning_rate = 0.12 / (1.0 + step / 500.0)
        for vector, label in zip(vectors, labels):
            prediction = sigmoid(sum(weight * value for weight, value in zip(weights, vector)))
            sample_weight = positive_weight if label else 1.0
            error = (prediction - label) * sample_weight
            for index, value in enumerate(vector):
                gradients[index] += error * value
        for index in range(feature_count):
            regularization = 0.0 if index == 0 else 0.002 * weights[index]
            weights[index] -= learning_rate * (gradients[index] / len(samples) + regularization)

    return HeadwordModel(weights, means, scales, len(samples), positives)
