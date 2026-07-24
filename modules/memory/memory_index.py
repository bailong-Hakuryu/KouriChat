"""Inverted & Tag Index for High-Performance Memory Retrieval.

Maintains a lightweight keyword/tag-to-node_id inverted index (memory_index.json)
to eliminate full O(N) scanning over memory node storage.

Follows Google Python Style Guide.
"""

import json
import logging
import os
import re
from typing import Dict, List, Set

logger = logging.getLogger('main')


class MemoryIndex:
    """Inverted and Tag Index mapper (Token/Tag -> Set of Node IDs)."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def tokenize(self, text: str) -> Set[str]:
        """Extracts unique keywords, tags, and bi-grams from text."""
        if not text:
            return set()
        clean_text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text.lower())
        tokens = set(clean_text.split())

        # Add Chinese bi-grams for Chinese character matching
        chinese_chars = [c for c in clean_text if '\u4e00' <= c <= '\u9fff']
        for i in range(len(chinese_chars) - 1):
            tokens.add(chinese_chars[i] + chinese_chars[i + 1])

        return tokens

    def _get_index_path(self, avatar_name: str, user_id: str) -> str:
        """Returns file path for memory_index.json."""
        avatar_memory_dir = os.path.join(self.root_dir, "data", "avatars", avatar_name, "memory", user_id)
        os.makedirs(avatar_memory_dir, exist_ok=True)
        return os.path.join(avatar_memory_dir, "memory_index.json")

    def load_index(self, avatar_name: str, user_id: str) -> Dict[str, List[str]]:
        """Loads inverted index mapping from JSON."""
        index_path = self._get_index_path(avatar_name, user_id)
        if not os.path.exists(index_path):
            return {}
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memory index: {e}")
            return {}

    def save_index(self, avatar_name: str, user_id: str, inverted_index: Dict[str, List[str]]):
        """Saves inverted index mapping to JSON."""
        index_path = self._get_index_path(avatar_name, user_id)
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(inverted_index, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory index: {e}")

    def update_node_index(
        self, avatar_name: str, user_id: str, node_id: str, content: str, tags: List[str] = None
    ):
        """Adds or updates keywords and tags for a memory node in inverted index."""
        inverted_index = self.load_index(avatar_name, user_id)
        tokens = self.tokenize(content)

        if tags:
            for tag in tags:
                tokens.update(self.tokenize(tag))

        for token in tokens:
            if token not in inverted_index:
                inverted_index[token] = []
            if node_id not in inverted_index[token]:
                inverted_index[token].append(node_id)

        self.save_index(avatar_name, user_id, inverted_index)

    def lookup_candidate_node_ids(self, avatar_name: str, user_id: str, query: str) -> Set[str]:
        """Looks up candidate node_ids matching query tokens/tags from inverted index."""
        inverted_index = self.load_index(avatar_name, user_id)
        if not inverted_index:
            return set()

        query_tokens = self.tokenize(query)
        candidate_ids: Set[str] = set()

        for token in query_tokens:
            if token in inverted_index:
                candidate_ids.update(inverted_index[token])

        return candidate_ids
