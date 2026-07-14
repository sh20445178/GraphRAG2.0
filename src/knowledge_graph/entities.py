"""
Entity and relation types used to build the ML knowledge graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Entity:
    id: str
    label: str               # e.g. "Algorithm", "Dataset", "Framework", "Metric"
    name: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "name": self.name,
            "description": self.description,
            **self.properties,
        }


@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: str       # e.g. "USES", "TRAINED_ON", "OUTPERFORMS", "PART_OF"
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type,
            "weight": self.weight,
            **self.properties,
        }


# ── Canonical entity labels ───────────────────────────────────────────────

ENTITY_LABELS = {
    "Algorithm",
    "Dataset",
    "Framework",
    "Metric",
    "Task",
    "Paper",
    "Architecture",
    "Optimizer",
    "LossFunction",
    "HyperParameter",
}

RELATION_TYPES = {
    "USES",
    "TRAINED_ON",
    "EVALUATED_ON",
    "OUTPERFORMS",
    "VARIANT_OF",
    "PART_OF",
    "PROPOSED_IN",
    "REQUIRES",
    "SUITABLE_FOR",
}
