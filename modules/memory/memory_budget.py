"""Token Budget Manager for Long-Term Personality Memory Layer.

Controls and allocates token limits across Core Memory, Timeline Context,
Dynamic Memory Nodes, and Dialogue History to prevent context overflow and
minimize 'Lost in the Middle' LLM attention degradation.

Follows Google Python Style Guide.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger('main')


class MemoryBudgetManager:
    """Manages token budgets for prompt construction across memory categories."""

    def __init__(
        self,
        core_budget: int = 300,
        timeline_budget: int = 500,
        dynamic_budget: int = 800,
        history_budget: int = 1000
    ):
        """Initializes budget allocations in estimated tokens.

        Args:
            core_budget: Maximum tokens allocated for Core Memory.
            timeline_budget: Maximum tokens allocated for Timeline Journal.
            dynamic_budget: Maximum tokens allocated for Dynamic Memory Nodes.
            history_budget: Maximum tokens allocated for Recent Message History.
        """
        self.core_budget = core_budget
        self.timeline_budget = timeline_budget
        self.dynamic_budget = dynamic_budget
        self.history_budget = history_budget

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count for a given text string.

        Uses standard heuristic: ~1 token per 1.5 CJK characters or 4 ASCII chars.

        Args:
            text: Input text string.

        Returns:
            Estimated integer token count.
        """
        if not text:
            return 0
        
        cjk_chars = 0
        ascii_chars = 0
        for char in text:
            if ord(char) > 127:
                cjk_chars += 1
            else:
                ascii_chars += 1

        estimated = int((cjk_chars / 1.5) + (ascii_chars / 4.0))
        return max(1, estimated)

    def trim_text_to_budget(self, text: str, max_tokens: int) -> str:
        """Truncates text to fit within a given token limit.

        Args:
            text: Content string.
            max_tokens: Maximum allowed tokens.

        Returns:
            Truncated string within budget.
        """
        if not text or self.estimate_tokens(text) <= max_tokens:
            return text

        lines = text.splitlines()
        trimmed_lines: List[str] = []
        current_tokens = 0

        for line in lines:
            line_tokens = self.estimate_tokens(line)
            if current_tokens + line_tokens > max_tokens:
                break
            trimmed_lines.append(line)
            current_tokens += line_tokens

        return "\n".join(trimmed_lines)

    def allocate_memory_context(
        self,
        core_memory: str,
        timeline_entries: List[str],
        dynamic_nodes_formatted: str
    ) -> Dict[str, str]:
        """Trims and formats memory sections based on their respective token budgets.

        Args:
            core_memory: Core personality memory string.
            timeline_entries: List of first-person timeline entries.
            dynamic_nodes_formatted: Formatted dynamic memory node string.

        Returns:
            Dictionary containing token-budgeted memory strings.
        """
        # Trim Core Memory
        trimmed_core = self.trim_text_to_budget(core_memory, self.core_budget)

        # Trim Timeline
        timeline_text = "\n".join(timeline_entries) if timeline_entries else ""
        trimmed_timeline = self.trim_text_to_budget(timeline_text, self.timeline_budget)

        # Trim Dynamic Memory
        trimmed_dynamic = self.trim_text_to_budget(dynamic_nodes_formatted, self.dynamic_budget)

        total_tokens = (
            self.estimate_tokens(trimmed_core) +
            self.estimate_tokens(trimmed_timeline) +
            self.estimate_tokens(trimmed_dynamic)
        )

        logger.debug(
            f"Budget allocated -> Core: {self.estimate_tokens(trimmed_core)}t, "
            f"Timeline: {self.estimate_tokens(trimmed_timeline)}t, "
            f"Dynamic: {self.estimate_tokens(trimmed_dynamic)}t | Total: {total_tokens}t"
        )

        return {
            "core_memory": trimmed_core,
            "timeline_context": trimmed_timeline,
            "dynamic_memory": trimmed_dynamic,
            "total_estimated_tokens": str(total_tokens)
        }
