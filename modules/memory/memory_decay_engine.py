"""Memory Decay and Analysis Engine.

Performs post-dialogue LLM analysis to extract structured, typed memory nodes,
evaluate emotional valence and confidence, execute Single-Pass Timeline generation,
and filter out prompt injection threats via MemorySecurity.

Follows Google Python Style Guide.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from modules.memory.memory_node import MemoryNode, MemoryType, MemoryState
from modules.memory.memory_security import MemorySecurity
from src.services.ai.llm_service import LLMService

logger = logging.getLogger('main')


class MemoryDecayEngine:
    """Engine responsible for post-dialogue structured analysis and decay calculations."""

    def __init__(self, root_dir: str, llm_service: LLMService):
        self.root_dir = root_dir
        self.llm = llm_service
        self.security = MemorySecurity()
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        prompt_path = os.path.join(self.root_dir, "src", "base", "memory_weight.md")
        if os.path.exists(prompt_path):
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read memory_weight.md: {e}")

        return (
            "分析以下对话，提取客观事实，打分情绪(-1.0到1.0)和重要性(0.0到1.0)。"
            "输出JSON格式：{\"should_save\": true, \"user_state\":\"\", \"emotional_score\":0.0, "
            "\"importance_score\":0.5, \"confidence\":0.8, \"timeline_entry\":\"\", \"extracted_nodes\":[]}"
        )

    def analyze_dialogue_round(
        self, avatar_name: str, user_id: str, user_msg: str, bot_reply: str
    ) -> Tuple[Optional[str], float, float, Optional[str], List[MemoryNode]]:
        """Analyzes a dialogue round in a single LLM pass.

        Args:
            avatar_name: Avatar character name.
            user_id: User identifier.
            user_msg: User message.
            bot_reply: Bot reply.

        Returns:
            Tuple of (user_state, emotional_score, importance_score, timeline_entry, extracted_nodes).
        """
        dialogue_text = f"用户({user_id}): {user_msg}\n{avatar_name}: {bot_reply}"
        full_prompt = f"{self.prompt_template}\n\n对话内容:\n{dialogue_text}"

        try:
            raw_response = self.llm.get_response(
                message=full_prompt,
                user_id=f"analyzer_{user_id}",
                system_prompt="你是一个精准的数据分析和结构化 JSON 转化组件。只输出符合 Schema 要求的 JSON。"
            )

            json_str = raw_response.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0].strip()

            parsed = json.loads(json_str)

            if not parsed.get("should_save", True):
                logger.debug(f"Memory analysis skipped saving for user {user_id} (should_save=false)")
                return None, 0.0, 0.2, None, []

            user_state = parsed.get("user_state", "")
            emotional_score = float(parsed.get("emotional_score", 0.0))
            importance_score = float(parsed.get("importance_score", 0.5))
            timeline_entry = parsed.get("timeline_entry", "")
            raw_nodes = parsed.get("extracted_nodes", [])

            validated_nodes: List[MemoryNode] = []

            if isinstance(raw_nodes, list):
                for idx_data in raw_nodes:
                    if not isinstance(idx_data, dict):
                        continue

                    content = idx_data.get("content", "").strip()
                    if not content:
                        continue

                    # Parse node_type
                    type_str = idx_data.get("node_type", "fact").lower()
                    try:
                        node_type = MemoryType(type_str)
                    except ValueError:
                        node_type = MemoryType.FACT

                    tags = idx_data.get("tags", [])
                    node_imp = float(idx_data.get("importance", importance_score))
                    node_conf = float(idx_data.get("confidence", 0.8))

                    # CORE Memory escalation rule: High importance + High confidence
                    if node_imp >= 0.85 and node_conf >= 0.90 and node_type != MemoryType.CORE:
                        logger.info(f"Escalating node to CORE memory: '{content[:30]}...'")
                        node_type = MemoryType.CORE

                    temp_node = MemoryNode(
                        node_id=f"n_{os.urandom(4).hex()}",
                        user_id=user_id,
                        node_type=node_type,
                        content=content,
                        tags=tags if isinstance(tags, list) else [],
                        base_importance=node_imp,
                        emotional_score=emotional_score,
                        confidence=node_conf,
                        timeline_entry=timeline_entry if timeline_entry else None
                    )

                    # Security inspection
                    is_safe, reason = self.security.validate_node(temp_node)
                    if is_safe:
                        validated_nodes.append(temp_node)
                    else:
                        logger.warning(f"Security validation rejected memory node: {reason}")

            logger.info(
                f"Memory Analysis Pass Complete (User: {user_id}): State='{user_state}', "
                f"Emotion={emotional_score}, ValidatedNodes={len(validated_nodes)}, TimelineEntry={bool(timeline_entry)}"
            )

            return user_state, emotional_score, importance_score, timeline_entry, validated_nodes

        except Exception as e:
            logger.error(f"Failed to analyze dialogue round for memory: {e}")
            return None, 0.0, 0.3, None, []
