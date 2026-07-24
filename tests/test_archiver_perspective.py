import os
import tempfile
import unittest
from unittest.mock import MagicMock
from erii.core.archiver import AsyncArchiverWorker
from erii.storage.sqlite_storage import SQLiteStorage
from erii.core.queue.persistent_queue import PersistentTaskQueue

class TestArchiverPerspectiveAndDatabase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="kouri_archiver_test_")

    def test_database_directory_auto_creation(self):
        """Verify SQLiteStorage and PersistentTaskQueue auto-create parent directories."""
        db_dir = os.path.join(self.test_dir, "nested", "deep", "dir")
        db_path = os.path.join(db_dir, "erii_memory.db")
        
        # Folder does not exist yet
        self.assertFalse(os.path.exists(db_dir))

        # Initializing SQLiteStorage and PersistentTaskQueue should not crash with "unable to open database file"
        try:
            storage = SQLiteStorage(db_path=db_path)
            queue = PersistentTaskQueue(db_path=db_path)
            self.assertTrue(os.path.exists(db_dir))
        except Exception as e:
            self.fail(f"Failed to auto-create DB directory: {e}")

    def test_extraction_prompt_formatting_and_perspective(self):
        """Verify EXTRACTION_PROMPT includes agent_id, user_id, and perspective rules."""
        worker = AsyncArchiverWorker.__new__(AsyncArchiverWorker)
        worker.enable_sanitizer = False
        worker.enable_pii_scrubbing = False
        worker.storage = MagicMock()
        worker.llm_adapter = MagicMock()

        # Mock LLM response with proper perspective
        mock_llm_json = '''{
            "timeline_entry": "我与 白龙 约定明天早上告诉他一些事情。",
            "thought_entry": {
                "content": "希望白龙今晚做个好梦。",
                "visibility": "public_log"
            },
            "impressions": [
                {
                    "type": "event",
                    "content": "我向 白龙 道晚安，祝福他能在梦里见到圆圆的饭团。",
                    "base_importance": 0.8
                }
            ]
        }'''
        worker.llm_adapter.generate.return_value = mock_llm_json

        task = {
            "agent_id": "Uesugi_Erii",
            "user_id": "白龙",
            "user_msg": "到家了，绘梨衣",
            "bot_reply": "晚安啦Sakura"
        }

        worker._process_archival(task)

        # Check prompt sent to LLM contains agent_id and user_id perspective
        call_args = worker.llm_adapter.generate.call_args[0][0]
        self.assertIn("Uesugi_Erii", call_args, "Prompt must include agent_id to ground character perspective")
        self.assertIn("白龙", call_args, "Prompt must include user_id to prevent mistaking user for '我'")

if __name__ == "__main__":
    unittest.main()
