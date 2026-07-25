import os
import time
import json
import shutil
import tempfile
import unittest
import datetime
from unittest.mock import MagicMock, patch

from modules.memory.memory_service import MemoryService
from src.services.ai.llm_service import LLMService
from erii.core.archiver import AsyncArchiverWorker
from erii.storage.sqlite_storage import SQLiteStorage
from erii.core.queue.persistent_queue import PersistentTaskQueue

class TestMemoryAndPerceptionRobustness(unittest.TestCase):
    """
    Comprehensive tests ensuring:
    1. Time perception & temporal continuity across short/long intervals & midnight boundary.
    2. Character first-person perspective (agent_id = '我', user_id = '用户/Sakura').
    3. Auto DB dir creation and background archival thread reliability.
    4. End-to-end add_conversation -> archiver -> timeline & nodes.
    """

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="kouri_robustness_test_")
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

    def test_time_perception_across_midnight_boundary(self):
        """Verify time perception correctly identifies short intervals even when crossing midnight."""
        with patch("src.services.ai.llm_service.LLMService._get_available_models", return_value=["deepseek-ai/DeepSeek-V4-Pro"]):
            llm = LLMService(
                api_key="mock_key",
                base_url="https://api.openai.com/v1",
                model="deepseek-ai/DeepSeek-V4-Pro",
                max_token=2000,
                temperature=0.7,
                max_groups=5
            )

        user_id = "白龙"
        t1 = datetime.datetime(2026, 7, 24, 23, 59, 50)
        t2 = datetime.datetime(2026, 7, 25, 0, 0, 30)  # 40 seconds later, next day

        llm.chat_contexts[user_id] = [
            {"role": "user", "content": "到家了", "timestamp": t1.isoformat()},
            {"role": "assistant", "content": "晚安啦", "timestamp": t1.isoformat()},
            {"role": "user", "content": "好的", "timestamp": t2.isoformat()}
        ]

        time_ctx = llm._build_time_context(user_id)
        self.assertTrue("40秒" in time_ctx or "秒" in time_ctx)
        self.assertTrue("实时连续进行中" in time_ctx or "连续" in time_ctx)

    def test_message_timestamp_formatting_for_api(self):
        """Verify timestamp [HH:MM] is prepended to USER history messages only (not assistant)."""
        with patch("src.services.ai.llm_service.LLMService._get_available_models", return_value=["deepseek-ai/DeepSeek-V4-Pro"]):
            llm = LLMService(
                api_key="mock_key",
                base_url="https://api.openai.com/v1",
                model="deepseek-ai/DeepSeek-V4-Pro",
                max_token=2000,
                temperature=0.7,
                max_groups=5
            )

        user_id = "白龙"
        t1 = datetime.datetime(2026, 7, 24, 21, 38, 41)
        llm.chat_contexts[user_id] = [
            {"role": "user", "content": "到家了，绘梨衣", "timestamp": t1.isoformat()},
            {"role": "assistant", "content": "晚安啦Sakura", "timestamp": t1.isoformat()}
        ]

        mock_resp = MagicMock()
        mock_resp.model_dump.return_value = {"choices": [{"message": {"content": "收到"}}]}
        with patch.object(llm.client.chat.completions, "create", return_value=mock_resp) as mock_create:
            llm.get_response("明天见", user_id, "人设提示")
            sent = mock_create.call_args[1]["messages"]
            history = sent[1:]
            # user 消息应带时间前缀
            self.assertEqual(history[0]["content"], "[21:38] 到家了，绘梨衣")
            # assistant 消息不应带时间前缀（避免模型学习并在输出中复现时间戳格式）
            self.assertEqual(history[1]["content"], "晚安啦Sakura")

    def test_archiver_perspective_and_node_extraction(self):
        """Verify archiver prompt formats agent_id & user_id and extracts nodes with character perspective."""
        worker = AsyncArchiverWorker.__new__(AsyncArchiverWorker)
        worker.enable_sanitizer = False
        worker.enable_pii_scrubbing = False
        worker.storage = MagicMock()
        worker.storage.load_nodes.return_value = []
        worker.llm_adapter = MagicMock()

        mock_json = json.dumps({
            "timeline_entry": "我与 白龙 确认了他已经安全到家。",
            "thought_entry": {
                "content": "今晚星星很亮，希望 白龙 能做个好梦。",
                "visibility": "public_log"
            },
            "impressions": [
                {
                    "type": "event",
                    "content": "我向 白龙 道晚安，并祝福他在梦里见到圆圆的饭团。",
                    "base_importance": 0.85
                }
            ]
        })
        worker.llm_adapter.generate.return_value = mock_json

        task = {
            "agent_id": "Uesugi_Erii",
            "user_id": "白龙",
            "user_msg": "到家了，绘梨衣",
            "bot_reply": "晚安啦Sakura"
        }

        worker._process_archival(task)

        # Assert perspective guidelines were included in LLM prompt
        prompt = worker.llm_adapter.generate.call_args[0][0]
        self.assertIn("Uesugi_Erii", prompt)
        self.assertIn("白龙", prompt)
        self.assertIn("STRICT FIRST-PERSON PERSPECTIVE", prompt)

        # Assert nodes saved to storage contain correct agent_id and content
        save_nodes_call = worker.storage.save_nodes.call_args
        self.assertIsNotNone(save_nodes_call)
        agent_id_arg, user_id_arg, saved_nodes = save_nodes_call[0]
        self.assertEqual(agent_id_arg, "Uesugi_Erii")
        self.assertTrue(len(saved_nodes) > 0)
        self.assertIn("白龙", saved_nodes[0].content)

    def test_end_to_end_add_conversation_and_timeline(self):
        """End-to-end test: add_conversation -> background archival -> timeline entries."""
        avatar = "Uesugi_Erii"
        user = "白龙"

        # Mock LLM memory extraction response
        archival_json = json.dumps({
            "timeline_entry": "我得知白龙已经安全到家，心里感到踏实。",
            "thought_entry": {
                "content": "今晚星星特别亮，希望白龙睡个好觉。",
                "visibility": "public_log"
            },
            "impressions": [
                {
                    "type": "event",
                    "content": "我叮嘱白龙早点休息。",
                    "base_importance": 0.8
                }
            ]
        })
        self.memory_service._call_llm_chat = MagicMock(return_value=archival_json)

        # Initialize memory files
        self.memory_service.initialize_memory_files(avatar, user)

        # Add conversation turn
        self.memory_service.add_conversation(avatar, "到家了，绘梨衣", "晚安啦Sakura", user)

        # Wait briefly for background archiver thread to consume persistent queue
        time.sleep(0.5)

        # Verify timeline entries can be retrieved without error
        entries = self.memory_service.get_timeline_entries(avatar, user)
        self.assertTrue(len(entries) > 0, "Timeline entries should be created and retrieved without database errors")
        self.assertTrue(any("白龙" in entry or "心路感悟" in entry for entry in entries))

if __name__ == "__main__":
    unittest.main()
