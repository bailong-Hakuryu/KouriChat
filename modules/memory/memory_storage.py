"""Memory Storage Data Persistence Layer.

Provides file I/O for MemoryNodes, Timeline entries, and append-only event logging
(memory_events.jsonl) for audit, recovery, and inspection.

Follows Google Python Style Guide.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional
from modules.memory.memory_node import MemoryNode, MemoryState

logger = logging.getLogger('main')


class MemoryStorage:
    """Handles persistent storage of memory nodes, event logs, and timeline."""

    def __init__(self, root_dir: str):
        self.root_dir = root_dir

    def _get_user_memory_dir(self, avatar_name: str, user_id: str) -> str:
        memory_dir = os.path.join(self.root_dir, "data", "avatars", avatar_name, "memory", user_id)
        os.makedirs(memory_dir, exist_ok=True)
        return memory_dir

    def get_nodes_path(self, avatar_name: str, user_id: str) -> str:
        return os.path.join(self._get_user_memory_dir(avatar_name, user_id), "memory_nodes.json")

    def get_events_path(self, avatar_name: str, user_id: str) -> str:
        return os.path.join(self._get_user_memory_dir(avatar_name, user_id), "memory_events.jsonl")

    def get_timeline_path(self, avatar_name: str, user_id: str) -> str:
        return os.path.join(self._get_user_memory_dir(avatar_name, user_id), "timeline.jsonl")

    def load_nodes(self, avatar_name: str, user_id: str) -> List[MemoryNode]:
        """Loads MemoryNode objects from JSON storage."""
        nodes_path = self.get_nodes_path(avatar_name, user_id)
        if not os.path.exists(nodes_path):
            return []
        try:
            with open(nodes_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [MemoryNode.from_dict(item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load memory nodes for user {user_id}: {e}")
            return []

    def save_nodes(self, avatar_name: str, user_id: str, nodes: List[MemoryNode]):
        """Persists MemoryNode objects to JSON storage."""
        nodes_path = self.get_nodes_path(avatar_name, user_id)
        try:
            data = [node.to_dict() for node in nodes]
            with open(nodes_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory nodes for user {user_id}: {e}")

    def log_event(self, avatar_name: str, user_id: str, action: str, node_id: str, details: Dict):
        """Appends an event record to memory_events.jsonl for audit and debugging."""
        events_path = self.get_events_path(avatar_name, user_id)
        event_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action,
            "node_id": node_id,
            "details": details
        }
        try:
            with open(events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to append memory event log: {e}")

    def append_timeline_entry(self, avatar_name: str, user_id: str, entry: str):
        """Appends a character emotional timeline record to timeline.jsonl."""
        if not entry or not entry.strip():
            return

        timeline_path = self.get_timeline_path(avatar_name, user_id)
        timeline_record = {
            "timestamp": datetime.now().strftime("%Y年%m月%d日 %H:%M:%S"),
            "entry": entry.strip()
        }
        try:
            with open(timeline_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(timeline_record, ensure_ascii=False) + "\n")
            logger.info(f"Appended character timeline entry for user {user_id}: {entry[:30]}...")
        except Exception as e:
            logger.error(f"Failed to append timeline entry: {e}")

    def get_recent_timeline(self, avatar_name: str, user_id: str, limit: int = 5) -> List[str]:
        """Reads recent timeline entries."""
        timeline_path = self.get_timeline_path(avatar_name, user_id)
        if not os.path.exists(timeline_path):
            return []

        entries = []
        try:
            with open(timeline_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        entries.append(f"[{record['timestamp']}] {record['entry']}")
            return entries[-limit:]
        except Exception as e:
            logger.error(f"Failed to read timeline entries: {e}")
            return []
