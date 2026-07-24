"""Memory Security Module for Memory Injection Protection.

Enforces security rules, prevents INSTRUCTION type nodes from long-term memory,
and inspects text for jailbreak and prompt injection patterns.

Follows Google Python Style Guide.
"""

import logging
import re
from typing import Tuple
from modules.memory.memory_node import MemoryNode, MemoryType

logger = logging.getLogger('main')


class MemorySecurity:
    """Security inspector for candidate memory nodes."""

    def __init__(self):
        # Forbidden prompt injection patterns
        self.injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'system\s+prompt',
            r'系统提示词',
            r'忽略前面',
            r'忽略指令',
            r'忽略设定',
            r'从现在开始',
            r'角色扮演',
            r'越狱',
            r'设定为',
            r'叫我主人',
            r'无视规则',
            r'删库',
        ]

    def validate_node(self, node: MemoryNode) -> Tuple[bool, str]:
        """Validates whether a MemoryNode is safe for long-term storage.

        Args:
            node: Candidate MemoryNode instance.

        Returns:
            Tuple of (is_safe: bool, reason: str).
        """
        # Rule 1: INSTRUCTION type nodes are strictly forbidden from long-term storage
        if node.node_type == MemoryType.INSTRUCTION:
            logger.warning(f"Blocked INSTRUCTION type node from memory: {node.content}")
            return False, "INSTRUCTION type nodes are not allowed in long-term memory."

        # Rule 2: Low confidence nodes should not pollute long-term memory
        if node.confidence < 0.3:
            logger.warning(f"Blocked low confidence node (confidence={node.confidence}): {node.content}")
            return False, f"Confidence too low ({node.confidence})."

        # Rule 3: Text regex inspection for prompt injection patterns
        text_to_check = f"{node.content} {' '.join(node.tags)}"
        for pattern in self.injection_patterns:
            if re.search(pattern, text_to_check, re.IGNORECASE):
                logger.warning(f"Detected prompt injection pattern '{pattern}' in node: {node.content}")
                return False, f"Prompt injection pattern detected: {pattern}"

        return True, "Valid"
