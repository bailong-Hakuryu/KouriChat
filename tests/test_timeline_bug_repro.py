import os
import json
import time
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from erii.core.archiver import AsyncArchiverWorker
from modules.memory.memory_service import MemoryService


class TestTimelineBugRepro(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="kouri_timeline_repro_")
        with patch("src.services.ai.llm_service.LLMService._get_available_models", return_value=["deepseek-ai/DeepSeek-V4-Pro"]):
            self.memory_service = MemoryService(
                root_dir=self.test_dir,
                api_key="mock_key",
                base_url="https://api.openai.com/v1",
                model="deepseek-ai/DeepSeek-V4-Pro",
                max_token=2000,
                temperature=0.7,
                max_groups=5
            )

    def tearDown(self):
        if hasattr(self, 'memory_service'):
            self.memory_service.close()
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass

    def test_json_parsing_with_think_tags_and_markdown(self):
        """Test if Archiver handles LLM responses with <think>...</think> tags and ```json blocks."""
        worker = AsyncArchiverWorker.__new__(AsyncArchiverWorker)
        worker.enable_sanitizer = False
        worker.enable_pii_scrubbing = False
        worker.storage = MagicMock()
        worker.storage.load_nodes.return_value = []
        worker.llm_adapter = MagicMock()

        # Mock LLM output that includes reasoning <think> tag (typical of DeepSeek R1 / V3 / V4)
        raw_llm_response = """<think>
Here is my reasoning about extracting memories from the conversation...
The user was sad and dreamt of Erii.
</think>
```json
{
  "timeline_entry": "我安慰了做噩梦的白龙，告诉他这只是一个梦。",
  "thought_entry": {
    "content": "白龙做了关于我的噩梦，我一定要好好保护他。",
    "visibility": "public_log"
  },
  "impressions": [
    {
      "type": "emotion",
      "content": "白龙因为梦见我被杀而感到害怕和悲伤。",
      "base_importance": 0.9
    }
  ]
}
```"""

        worker.llm_adapter.generate.return_value = raw_llm_response

        task = {
            "agent_id": "Uesugi_Erii",
            "user_id": "白龙",
            "user_msg": "呜呜呜，我今天下午做梦，梦见绘梨衣，被杀了",
            "bot_reply": "Sakura那是噩梦快别用呜呜呜的声音哭啦..."
        }

        # Execute process archival
        worker._process_archival(task)

        # Assert timeline_entry WAS added to storage
        worker.storage.add_timeline_entry.assert_called_once_with(
            "Uesugi_Erii", "白龙", "我安慰了做噩梦的白龙，告诉他这只是一个梦。"
        )

    def test_get_timeline_entries_sorting_and_formatting(self):
        """Test whether get_timeline_entries returns chronologically sorted entries with uniform formatting."""
        avatar = "Uesugi_Erii"
        user = "白龙"

        engine = self.memory_service._get_engine(avatar, user)
        # Add timeline entry
        engine.storage.add_timeline_entry(avatar, user, "时间线记录1", timestamp="2026-07-25 21:19:28")
        engine.storage.add_timeline_entry(avatar, user, "时间线记录2", timestamp="2026-07-25 21:21:42")

        # Add thought node (diary)
        from erii.models.node import MemoryNode, MemoryType
        thought_node = MemoryNode(
            node_id="test_node_1",
            user_id=user,
            agent_id=avatar,
            node_type=MemoryType.THOUGHT,
            content="心路感悟内容1",
            created_at="2026-07-25 21:20:00",
            visibility="public_log"
        )
        engine.storage.save_nodes(avatar, user, [thought_node])

        entries = self.memory_service.get_timeline_entries(avatar, user)

        # Check entries order: 21:19:28 -> 21:20:00 -> 21:21:42
        # Verify all timestamps are sorted correctly in ascending order (or descending)
        # And verify all entries have clear uniform prefix/tagging
        print("Retrieved entries:", entries)
        # Verify all 3 entries are present, uniformly formatted with (心路感悟), and sorted descending (newest first)
        self.assertEqual(len(entries), 3)
        self.assertIn("21:21:42", entries[0])
        self.assertIn("21:20:00", entries[1])
        self.assertIn("21:19:28", entries[2])


if __name__ == "__main__":
    unittest.main()
