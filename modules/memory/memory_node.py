"""Memory Node data structures for Long-Term Personality Memory Layer.

Supports typed memories, multi-factor score calculation, versioning,
recall reinforcement, and lifecycle state management.

Follows Google Python Style Guide.
"""

from datetime import datetime
from enum import Enum
import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Supported memory classification types."""
    FACT = "fact"                # 客观事实
    PREFERENCE = "preference"      # 用户偏好
    EVENT = "event"              # 共同经历事件
    EMOTION = "emotion"          # 情绪节点
    RELATIONSHIP = "relationship"# 关系节点
    CORE = "core"                # 核心人格印象
    INSTRUCTION = "instruction"  # 指令型 (严禁持久化)


class MemoryState(str, Enum):
    """Memory lifecycle states."""
    ACTIVE = "active"            # 活跃，参与常规检索
    WEAK = "weak"                # 弱化，低频访问
    ARCHIVED = "archived"        # 归档，仅在高相关搜索时读取


class MemoryNode(BaseModel):
    """Represents a single atomic memory unit with rich metadata."""

    node_id: str
    user_id: str
    node_type: MemoryType = MemoryType.FACT
    content: str
    tags: List[str] = Field(default_factory=list)
    base_importance: float = Field(default=0.5, ge=0.0, le=1.0)
    emotional_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    decayable: bool = True
    is_latest: bool = True
    superseded_by: Optional[str] = None
    state: MemoryState = MemoryState.ACTIVE
    timeline_entry: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    last_accessed_at: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def calculate_effective_weight(
        self, decay_rate: float = 0.05, max_weight_cap: float = 0.95
    ) -> float:
        """Calculates current dynamic weight applying time decay and emotional boost.

        Args:
            decay_rate: Time decay coefficient lambda.
            max_weight_cap: Upper bound saturation cap to prevent mono-topic dominance.

        Returns:
            Float value representing current dynamic weight between 0.0 and max_weight_cap.
        """
        if not self.is_latest or self.state == MemoryState.ARCHIVED:
            return 0.05

        try:
            last_time = datetime.strptime(self.last_accessed_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            last_time = datetime.now()

        elapsed_seconds = (datetime.now() - last_time).total_seconds()
        elapsed_days = max(0.0, elapsed_seconds / 86400.0)

        # Exponential decay: e^(-lambda * delta_t) if decayable
        time_decay = math.exp(-decay_rate * elapsed_days) if self.decayable else 1.0

        # Frequency boost (up to +0.2)
        frequency_boost = min(0.2, self.access_count * 0.03)

        # Emotional boost factor
        emotional_boost = abs(self.emotional_score) * 0.15

        # Relationship weight boost
        relationship_boost = 0.2 if self.node_type == MemoryType.RELATIONSHIP else 0.0

        raw_score = (
            (self.base_importance * time_decay)
            + frequency_boost
            + emotional_boost
            + relationship_boost
        )

        final_weight = max(0.0, min(max_weight_cap, round(raw_score, 4)))

        # Update lifecycle state based on weight
        if final_weight < 0.15 and self.state == MemoryState.ACTIVE:
            self.state = MemoryState.WEAK

        return final_weight

    def reinforce_recall(self, boost: float = 0.1):
        """Reinforces memory importance when successfully recalled."""
        self.access_count += 1
        self.base_importance = min(0.95, round(self.base_importance + boost, 4))
        self.last_accessed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.state == MemoryState.WEAK:
            self.state = MemoryState.ACTIVE

    def to_dict(self) -> Dict[str, Any]:
        """Serializes memory node to dictionary."""
        data = self.model_dump()
        data["node_type"] = self.node_type.value
        data["state"] = self.state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryNode":
        """Instantiates a MemoryNode from dictionary representation."""
        if "node_type" in data and isinstance(data["node_type"], str):
            try:
                data["node_type"] = MemoryType(data["node_type"])
            except ValueError:
                data["node_type"] = MemoryType.FACT

        if "state" in data and isinstance(data["state"], str):
            try:
                data["state"] = MemoryState(data["state"])
            except ValueError:
                data["state"] = MemoryState.ACTIVE

        return cls(**data)
