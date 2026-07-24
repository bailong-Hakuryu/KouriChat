import os
import shutil
import json
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from modules.memory.memory_service import MemoryService


class TestEriiIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="kouri_erii_test_")
        with patch("src.services.ai.llm_service.LLMService._get_available_models", return_value=["gpt-3.5-turbo"]):
            self.memory_service = MemoryService(
                root_dir=self.test_dir,
                api_key="mock_key",
                base_url="https://api.openai.com/v1",
                model="gpt-3.5-turbo",
                max_token=2000,
                temperature=0.7,
                max_groups=5
            )
        # Mock LLM API call to avoid network calls during test execution
        self.memory_service._call_llm_chat = MagicMock(return_value='{"timeline_entry": "Interaction logged", "impressions": []}')

    def tearDown(self):
        if hasattr(self, 'memory_service'):
            self.memory_service.close()
        try:
            shutil.rmtree(self.test_dir, ignore_errors=True)
        except Exception:
            pass


    def test_engine_initialization_and_node_ops(self):
        avatar = "sakura"
        user = "bob"

        # 1. 初始化记忆目录与文件
        self.memory_service.initialize_memory_files(avatar, user)

        # 2. 手动新增节点
        node_dict = self.memory_service.add_custom_node(
            avatar_name=avatar,
            user_id=user,
            content="Bob 喜欢喝大红袍乌龙茶",
            node_type="preference",
            tags=["茶", "喜好"],
            importance=0.9
        )
        self.assertIsNotNone(node_dict)
        self.assertEqual(node_dict["content"], "Bob 喜欢喝大红袍乌龙茶")

        # 3. 获取所有节点
        all_nodes = self.memory_service.get_all_nodes(avatar, user)
        self.assertEqual(len(all_nodes), 1)
        self.assertIn("effective_weight", all_nodes[0])

        # 4. 删除节点
        node_id = node_dict["node_id"]
        deleted = self.memory_service.delete_node(avatar, user, node_id)
        self.assertTrue(deleted)
        self.assertEqual(len(self.memory_service.get_all_nodes(avatar, user)), 0)

    def test_legacy_data_auto_migration(self):
        avatar = "legacy_bot"
        user = "alice"

        # 模拟旧版的 memory_nodes.json
        user_mem_dir = self.memory_service.storage._get_user_memory_dir(avatar, user)
        legacy_nodes_path = self.memory_service.storage.get_nodes_path(avatar, user)

        legacy_data = [
            {
                "node_id": "legacy_n1",
                "user_id": user,
                "node_type": "fact",
                "content": "Alice 是一名 Python 后端工程师",
                "tags": ["职业", "Python"],
                "base_importance": 0.85,
                "emotional_score": 0.0,
                "access_count": 3,
                "created_at": "2026-07-20 10:00:00",
                "last_accessed_at": "2026-07-24 10:00:00",
                "confidence": 0.95
            }
        ]

        with open(legacy_nodes_path, "w", encoding="utf-8") as f:
            json.dump(legacy_data, f, ensure_ascii=False, indent=2)

        # 触发获取节点，底层应自动进行无感迁移到 SQLite DB
        nodes = self.memory_service.get_all_nodes(avatar, user)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["content"], "Alice 是一名 Python 后端工程师")
        self.assertEqual(nodes[0]["node_id"], "legacy_n1")

        # 检查 erii_memory.db 文件已生成
        sqlite_db = os.path.join(user_mem_dir, "erii_memory.db")
        self.assertTrue(os.path.exists(sqlite_db))

    def test_memory_pack_export_import(self):
        avatar = "hero"
        user = "gamer"

        self.memory_service.add_custom_node(
            avatar_name=avatar,
            user_id=user,
            content="玩家获得了传说级法杖",
            node_type="event",
            tags=["装备"]
        )

        self.memory_service.save_core_memory(avatar, user, "角色是异世界的守护者")

        # 1. 导出 MemoryPack
        pack = self.memory_service.export_memory_pack(avatar, user)
        self.assertEqual(pack.core_memory, "角色是异世界的守护者")
        self.assertEqual(len(pack.nodes), 1)

        # 2. 恢复/导入到新用户
        new_user = "gamer_backup"
        self.memory_service.import_memory_pack(avatar, new_user, pack.to_dict(), overwrite=True)

        imported_nodes = self.memory_service.get_all_nodes(avatar, new_user)
        self.assertEqual(len(imported_nodes), 1)
        self.assertEqual(imported_nodes[0]["content"], "玩家获得了传说级法杖")
        self.assertEqual(self.memory_service.get_core_memory(avatar, new_user), "角色是异世界的守护者")

    def test_selective_memory_prompt(self):
        avatar = "sakura"
        user = "bob"

        self.memory_service.add_custom_node(
            avatar_name=avatar,
            user_id=user,
            content="Bob 喜欢暗黑主题和简约风格界面",
            node_type="preference",
            tags=["UI", "偏好"]
        )

        prompt = self.memory_service.get_selective_memory_prompt(avatar, user, query="暗黑主题")
        self.assertIn("Bob 喜欢暗黑主题", prompt)

    def test_chinese_user_id_validation(self):
        avatar = "Uesugi_Erii"
        user = "白龙"

        node_dict = self.memory_service.add_custom_node(
            avatar_name=avatar,
            user_id=user,
            content="白龙是系统的管理者",
            node_type="fact"
        )
        self.assertIsNotNone(node_dict)

        nodes = self.memory_service.get_all_nodes(avatar, user)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes[0]["content"], "白龙是系统的管理者")

    def test_legacy_timeline_migration(self):
        avatar = "legacy_avatar"
        user = "test_user"

        timeline_path = self.memory_service.storage.get_timeline_path(avatar, user)
        os.makedirs(os.path.dirname(timeline_path), exist_ok=True)
        with open(timeline_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"timestamp": "2026年07月23日 17:40:24", "entry": "白龙告诉我他完成了一个项目叫Erii"}) + "\n")

        entries = self.memory_service.get_timeline_entries(avatar, user)
        self.assertEqual(len(entries), 1)
        self.assertIn("白龙告诉我他完成了一个项目叫Erii", entries[0])


if __name__ == "__main__":
    unittest.main()



