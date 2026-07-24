"""Selective Memory Retriever with Diversity Cap Enforcement.

Filters, ranks, and retrieves Top-K memory nodes while enforcing category diversity
(Diversity Cap) to prevent mono-topic saturation, and triggers recall reinforcement.

Follows Google Python Style Guide.
"""

import logging
import re
from typing import List, Set
from modules.memory.memory_node import MemoryNode, MemoryType, MemoryState

logger = logging.getLogger('main')


class MemoryRetriever:
    """Retrieves relevant and high-weighted memory nodes with category diversity."""

    def __init__(self, decay_rate: float = 0.05):
        self.decay_rate = decay_rate

    def _tokenize(self, text: str) -> set:
        if not text:
            return set()
        clean_text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text.lower())
        words = set(clean_text.split())
        chinese_chars = [c for c in clean_text if '\u4e00' <= c <= '\u9fff']
        for i in range(len(chinese_chars) - 1):
            words.add(chinese_chars[i] + chinese_chars[i + 1])
        return words

    def calculate_relevance(self, query: str, node: MemoryNode) -> float:
        """Calculates textual relevance score between query and memory node content/tags."""
        query_tokens = self._tokenize(query)
        node_text = f"{node.content} {' '.join(node.tags)}"
        node_tokens = self._tokenize(node_text)

        if not query_tokens or not node_tokens:
            return 0.0

        intersection = query_tokens.intersection(node_tokens)
        union = query_tokens.union(node_tokens)
        jaccard_score = len(intersection) / len(union) if union else 0.0

        overlap_count = sum(1 for token in query_tokens if token in node_text.lower())
        containment_boost = min(0.5, overlap_count * 0.15)

        return min(1.0, jaccard_score + containment_boost)

    def retrieve_relevant_nodes(
        self,
        query: str,
        nodes: List[MemoryNode],
        top_k: int = 5,
        min_weight_threshold: float = 0.10
    ) -> List[MemoryNode]:
        """Filters, ranks, and returns Top-K nodes with Diversity Cap enforcement.

        Args:
            query: Current incoming user message text.
            nodes: Candidate MemoryNode list.
            top_k: Maximum number of nodes to return.
            min_weight_threshold: Floor threshold for effective weight.

        Returns:
            List of selected MemoryNode instances (reinforced upon selection).
        """
        if not nodes or not query.strip():
            return []

        scored_nodes = []
        for node in nodes:
            # Skip non-latest versions or archived nodes unless query explicitly matches
            if not node.is_latest or node.state == MemoryState.ARCHIVED:
                continue

            effective_weight = node.calculate_effective_weight(self.decay_rate)
            if effective_weight < min_weight_threshold:
                continue

            relevance = self.calculate_relevance(query, node)
            
            # Composite score: Dynamic weight (60%) + Context relevance (40%)
            composite_score = (effective_weight * 0.6) + (relevance * 0.4)
            scored_nodes.append((composite_score, node))

        scored_nodes.sort(key=lambda item: item[0], reverse=True)

        # Enforce Diversity Cap: Avoid filling Top-K with a single MemoryType
        selected_nodes: List[MemoryNode] = []
        type_counts = {}

        for score, node in scored_nodes:
            if len(selected_nodes) >= top_k:
                break

            node_type = node.node_type
            count = type_counts.get(node_type, 0)

            # Max 2 nodes of the same type in Top-5 unless candidates are limited
            if count < 2 or len(scored_nodes) <= top_k:
                selected_nodes.append(node)
                type_counts[node_type] = count + 1

        # Fill remaining slots if diversity constraint was too restrictive
        if len(selected_nodes) < top_k and len(scored_nodes) > len(selected_nodes):
            for score, node in scored_nodes:
                if len(selected_nodes) >= top_k:
                    break
                if node not in selected_nodes:
                    selected_nodes.append(node)

        # Apply Recall Reinforcement (Hebbian learning / Spaced repetition boost)
        for node in selected_nodes:
            node.reinforce_recall(boost=0.05)

        logger.debug(
            f"Selective memory retrieval (Diversity Cap): Query='{query[:20]}...', "
            f"Candidates={len(scored_nodes)}, Selected={len(selected_nodes)}"
        )

        return selected_nodes
