import unittest
import datetime
from unittest.mock import patch, MagicMock
from src.services.ai.llm_service import LLMService

class TestTimePerceptionAndContext(unittest.TestCase):
    def setUp(self):
        with patch("src.services.ai.llm_service.LLMService._get_available_models", return_value=["deepseek-ai/DeepSeek-V4-Pro"]):
            self.llm = LLMService(
                api_key="mock_key",
                base_url="https://api.openai.com/v1",
                model="deepseek-ai/DeepSeek-V4-Pro",
                max_token=2000,
                temperature=0.7,
                max_groups=5,
                auto_model_switch=False
            )

    def test_time_context_building(self):
        user_id = "白龙"
        t1 = datetime.datetime(2026, 7, 24, 21, 38, 41)
        t2 = datetime.datetime(2026, 7, 24, 21, 38, 54)

        self.llm.chat_contexts[user_id] = [
            {"role": "user", "content": "到家了，绘梨衣", "timestamp": t1.isoformat()},
            {"role": "assistant", "content": "晚安啦", "timestamp": t2.isoformat()},
            {"role": "user", "content": "好啊", "timestamp": t2.isoformat()}
        ]

        time_ctx = self.llm._build_time_context(user_id)
        # Should describe time interval correctly
        self.assertTrue("距离上条消息" in time_ctx or "时间的流逝" in time_ctx)

    def test_message_history_timestamps_included(self):
        """
        Verify whether clean_history or history messages sent to LLM include timestamp/time context metadata
        so that LLM understands each past turn's actual sending time and does not hallucinate next-day transitions.
        """
        user_id = "白龙"
        t1 = datetime.datetime(2026, 7, 24, 21, 38, 41)
        t2 = datetime.datetime(2026, 7, 24, 21, 39, 46)

        self.llm.chat_contexts[user_id] = [
            {"role": "user", "content": "到家了，绘梨衣", "timestamp": t1.isoformat()},
            {"role": "assistant", "content": "晚安啦Sakura要梦到明天草坪上的样子哦", "timestamp": t1.isoformat()},
            {"role": "user", "content": "好啊", "timestamp": t2.isoformat()}
        ]

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": "好的Sakura"}}]
        }

        with patch.object(self.llm.client.chat.completions, "create", return_value=mock_response) as mock_create:
            self.llm.get_response(
                message="昨天，我梦见了你",
                user_id=user_id,
                system_prompt="人设提示词"
            )
            
            call_kwargs = mock_create.call_args[1]
            sent_messages = call_kwargs["messages"]
            
            # Verify timestamps are formatted in history messages sent to LLM
            history_messages = sent_messages[1:]  # Exclude system prompt
            has_time_prefix = any(msg["content"].startswith("[") for msg in history_messages)
            self.assertTrue(has_time_prefix, "History messages sent to API must include time prefix [HH:MM]")

if __name__ == "__main__":
    unittest.main()
