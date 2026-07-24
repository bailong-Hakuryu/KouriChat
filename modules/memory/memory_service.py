import json
import logging
import os
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any

from data.config import MAX_GROUPS
from src.services.ai.llm_service import LLMService

from erii import ERIIEngine, ERIIConfig, MemoryNode as EriiNode, MemoryType as EriiType, MemoryPack
from erii.adapters.custom_adapter import CallableLLMAdapter
from erii.storage.sqlite_storage import SQLiteStorage
from erii.core.queue.persistent_queue import PersistentTaskQueue

from modules.memory.memory_storage import MemoryStorage
from modules.memory.memory_node import MemoryNode, MemoryType

logger = logging.getLogger('main')


class MemoryService:
    """
    KouriChat 长期人格记忆层 (E.R.I.I. Engine 整合版):
    基于开源引擎 E.R.I.I. (v0.2.0) 构建：
    1. 动态实例池 (Per Avatar & User ERIIEngine Pool) + SQLite 数据库持久化存储 (WAL 模式)
    2. 体验时间线 + 动态印象节点 + 第一人称心理独白与悬念保鲜
    3. RRF (Reciprocal Rank Fusion) 倒排与语义权重合成 + 多样性熔断 (Diversity Cap)
    4. 零延迟后台持久化任务队列 (PersistentTaskQueue) + 异常退避重试
    5. 透明无感历史数据迁移 (Auto Migration from legacy memory_nodes.json)
    6. MemoryPack 标准快照导出/导入支持
    """

    def __init__(self, root_dir: str, api_key: str, base_url: str, model: str, max_token: int, temperature: float,
                 max_groups: int = 10):
        self.root_dir = root_dir
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_token = max_token
        self.temperature = temperature
        self.max_groups = MAX_GROUPS if MAX_GROUPS else max_groups

        # 辅助分析 LLM
        self.analyzer_llm = LLMService(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_token=max_token,
            temperature=temperature,
            max_groups=max_groups
        )

        self.storage = MemoryStorage(root_dir=root_dir)
        self._engine_pool: Dict[Tuple[str, str], ERIIEngine] = {}
        self._pool_lock = threading.Lock()

    def close(self):
        """关闭所有 ERIIEngine 实例的后台归档 Worker 线程"""
        with self._pool_lock:
            for engine in self._engine_pool.values():
                if hasattr(engine, 'archiver') and engine.archiver:
                    engine.archiver.shutdown()
            self._engine_pool.clear()


    def _call_llm_chat(self, prompt: str) -> str:
        """调用分析用 LLM 生成记忆分析结果"""
        try:
            res = self.analyzer_llm.chat([{"role": "user", "content": prompt}])
            if res and not res.startswith("Error:"):
                return res
            return "{}"
        except Exception as e:
            logger.error(f"分析专用 LLM 调用异常: {e}")
            return "{}"


    def _get_engine(self, avatar_name: str, user_id: str) -> ERIIEngine:
        """获取或创建指定 (avatar_name, user_id) 的 ERIIEngine 实例，带加锁与历史数据无感迁移"""
        key = (avatar_name, user_id)
        if key in self._engine_pool:
            return self._engine_pool[key]

        with self._pool_lock:
            if key in self._engine_pool:
                return self._engine_pool[key]

            user_mem_dir = self.storage._get_user_memory_dir(avatar_name, user_id)

            # 实例化 SQLite 存储驱动与持久化队列
            db_path = os.path.join(user_mem_dir, "erii_memory.db")
            sqlite_storage = SQLiteStorage(db_path=db_path)
            task_queue = PersistentTaskQueue(db_path=db_path)


            # 适配器包装 LLM 方法
            llm_adapter = CallableLLMAdapter(fn=self._call_llm_chat)


            # ERII引擎配置
            config = ERIIConfig(
                storage_dir=user_mem_dir,
                enable_security_sanitizer=True,
                enable_pii_scrubbing=False
            )

            engine = ERIIEngine(
                storage_dir=user_mem_dir,
                llm=llm_adapter,
                storage_driver=sqlite_storage,
                config=config,
                task_queue=task_queue
            )

            self._engine_pool[key] = engine

            # 自动迁移旧数据：如果数据库中无节点，且存在旧版 memory_nodes.json，进行透明迁移
            self._migrate_legacy_data_if_needed(engine, avatar_name, user_id)

            return engine

    def _migrate_legacy_data_if_needed(self, engine: ERIIEngine, avatar_name: str, user_id: str):
        """将旧版的 memory_nodes.json 和 core_memory.json 一次性透明迁移至 ERII 存储中"""
        try:
            nodes_path = self.storage.get_nodes_path(avatar_name, user_id)
            if not os.path.exists(nodes_path):
                return

            existing_erii_nodes = engine.storage.load_nodes(avatar_name, user_id)
            if existing_erii_nodes:
                return  # 已存在 ERII 节点，说明已经迁移过

            legacy_nodes = self.storage.load_nodes(avatar_name, user_id)
            if not legacy_nodes:
                return

            erii_nodes = []
            for node in legacy_nodes:
                # 转换 MemoryType
                try:
                    type_val = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)
                    e_type = EriiType(type_val.lower())
                except ValueError:
                    e_type = EriiType.FACT

                e_node = EriiNode(
                    node_id=node.node_id,
                    user_id=user_id,
                    agent_id=avatar_name,
                    node_type=e_type,
                    content=node.content,
                    tags=node.tags or [],
                    base_importance=node.base_importance,
                    emotional_score=node.emotional_score,
                    access_count=node.access_count,
                    created_at=node.created_at,
                    last_accessed_at=node.last_accessed_at,
                    confidence=getattr(node, 'confidence', 1.0)
                )
                erii_nodes.append(e_node)

            engine.storage.save_nodes(avatar_name, user_id, erii_nodes)

            # 备份旧 core_memory 到 erii core_memory
            core_memory_path = self._get_core_memory_path(avatar_name, user_id)
            if os.path.exists(core_memory_path):
                try:
                    with open(core_memory_path, "r", encoding="utf-8") as f:
                        c_data = json.load(f)
                        c_content = c_data.get("content", "") if isinstance(c_data, dict) else ""
                        if c_content:
                            engine.set_core_memory(avatar_name, user_id, c_content)
                except Exception:
                    pass

            # 迁移旧版 timeline.jsonl 到 erii timeline_entries
            timeline_path = self.storage.get_timeline_path(avatar_name, user_id)
            if os.path.exists(timeline_path):
                existing_timeline = engine.storage.get_recent_timeline(avatar_name, user_id, limit=1)
                if not existing_timeline:
                    try:
                        with open(timeline_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    record = json.loads(line)
                                    ts = record.get("timestamp", "")
                                    entry_text = record.get("entry", "")
                                    if entry_text:
                                        engine.storage.add_timeline_entry(
                                            agent_id=avatar_name,
                                            user_id=user_id,
                                            entry=entry_text,
                                            timestamp=ts
                                        )
                    except Exception as e:
                        logger.error(f"迁移旧版 timeline.jsonl 失败: {e}")

            logger.info(f"成功为 [{avatar_name} / {user_id}] 迁移了 {len(erii_nodes)} 条旧记忆节点与时间线到 ERII 数据库。")



        except Exception as e:
            logger.error(f"旧记忆数据自动迁移失败 ({avatar_name}/{user_id}): {e}", exc_info=True)

    def initialize_memory_files(self, avatar_name: str, user_id: str):
        """初始化角色的记忆存储目录与交互文件"""
        try:
            memory_dir = self.storage._get_user_memory_dir(avatar_name, user_id)
            short_memory_path = self._get_short_memory_path(avatar_name, user_id)

            if not os.path.exists(short_memory_path):
                with open(short_memory_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)

            # 触发引擎及可能的迁移
            self._get_engine(avatar_name, user_id)

        except Exception as e:
            logger.error(f"初始化记忆文件失败: {str(e)}")

    def _get_short_memory_path(self, avatar_name: str, user_id: str) -> str:
        return os.path.join(self.storage._get_user_memory_dir(avatar_name, user_id), "short_memory.json")

    def _get_core_memory_path(self, avatar_name: str, user_id: str) -> str:
        return os.path.join(self.storage._get_user_memory_dir(avatar_name, user_id), "core_memory.json")

    def get_selective_memory_prompt(self, avatar_name: str, user_id: str, query: str) -> str:
        """
        前置选择性检索：利用 ERIIEngine 混合召回与 Diversity Cap
        生成注入 Prompt 的 Markdown 内容。
        """
        try:
            engine = self._get_engine(avatar_name, user_id)
            recalled_context = engine.recall(agent_id=avatar_name, user_id=user_id, query=query, top_k=5)
            if recalled_context:
                return recalled_context

            core_mem = self.get_core_memory(avatar_name, user_id)
            return f"# 核心记忆\n{core_mem}" if core_mem else ""
        except Exception as e:
            logger.error(f"获取选择性记忆 Prompt 失败: {e}")
            core_mem = self.get_core_memory(avatar_name, user_id)
            return f"# 核心记忆\n{core_mem}" if core_mem else ""

    def add_conversation(self, avatar_name: str, user_message: str, bot_reply: str, user_id: str,
                          is_system_message: bool = False):
        """添加对话到短期记忆，并推入 ERII 持久化后台归档队列"""
        if is_system_message or bot_reply.startswith("Error:"):
            return

        try:
            # 1. 更新短期对话缓存（UI渲染用）
            short_memory_path = self._get_short_memory_path(avatar_name, user_id)
            short_memory = []
            if os.path.exists(short_memory_path):
                try:
                    with open(short_memory_path, "r", encoding="utf-8") as f:
                        short_memory = json.load(f)
                except Exception:
                    short_memory = []

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            short_memory.append({
                "timestamp": timestamp,
                "user": user_message,
                "bot": bot_reply
            })

            if len(short_memory) > self.max_groups:
                short_memory = short_memory[-self.max_groups:]

            with open(short_memory_path, "w", encoding="utf-8") as f:
                json.dump(short_memory, f, ensure_ascii=False, indent=2)

            # 2. 推入 ERII 持久化队列
            engine = self._get_engine(avatar_name, user_id)
            engine.remember(agent_id=avatar_name, user_id=user_id, user_message=user_message, bot_reply=bot_reply)

        except Exception as e:
            logger.error(f"添加对话到记忆服务失败: {str(e)}")

    def get_core_memory(self, avatar_name: str, user_id: str) -> str:
        try:
            engine = self._get_engine(avatar_name, user_id)
            core_mem = engine.get_core_memory(agent_id=avatar_name, user_id=user_id)
            if core_mem:
                return core_mem
            
            # Fallback legacy check
            core_memory_path = self._get_core_memory_path(avatar_name, user_id)
            if not os.path.exists(core_memory_path):
                return ""
            with open(core_memory_path, "r", encoding="utf-8") as f:
                core_data = json.load(f)
                if isinstance(core_data, list) and len(core_data) > 0:
                    return core_data[0].get("content", "")
                return core_data.get("content", "") if isinstance(core_data, dict) else ""
        except Exception as e:
            logger.error(f"获取核心记忆失败: {str(e)}")
            return ""

    def update_core_memory(self, avatar_name: str, user_id: str, context: List[Dict] = None) -> bool:
        """更新核心记忆内容"""
        try:
            engine = self._get_engine(avatar_name, user_id)
            nodes = engine.storage.load_nodes(avatar_name, user_id)
            core_nodes = [n for n in nodes if n.node_type == EriiType.CORE or engine.decay_evaluator.evaluate_node(n) >= 0.75]
            top_nodes = sorted(core_nodes, key=lambda n: engine.decay_evaluator.evaluate_node(n), reverse=True)[:10]
            summary_lines = [f"[{n.node_type.value}] {n.content}" for n in top_nodes]

            if not summary_lines and context:
                summary_lines = [f"{item['role']}: {item['content']}" for item in context[-6:]]

            content = "；".join(summary_lines) if summary_lines else ""
            engine.set_core_memory(agent_id=avatar_name, user_id=user_id, content=content)

            # 保持同步 core_memory.json
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated_core_data = {"timestamp": now_str, "content": content}
            with open(self._get_core_memory_path(avatar_name, user_id), "w", encoding="utf-8") as f:
                json.dump(updated_core_data, f, ensure_ascii=False, indent=2)

            return True
        except Exception as e:
            logger.error(f"更新核心记忆失败: {str(e)}")
            return False

    def save_core_memory(self, avatar_name: str, user_id: str, content: str) -> bool:
        """从外部手动设置 Core Memory"""
        try:
            engine = self._get_engine(avatar_name, user_id)
            engine.set_core_memory(agent_id=avatar_name, user_id=user_id, content=content)
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            updated_core_data = {"timestamp": now_str, "content": content}
            with open(self._get_core_memory_path(avatar_name, user_id), "w", encoding="utf-8") as f:
                json.dump(updated_core_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存核心记忆失败: {str(e)}")
            return False

    def get_recent_context(self, avatar_name: str, user_id: str, context_size: int = None) -> List[Dict]:
        try:
            short_memory_path = self._get_short_memory_path(avatar_name, user_id)
            if not os.path.exists(short_memory_path):
                return []

            with open(short_memory_path, "r", encoding="utf-8") as f:
                short_memory = json.load(f)

            context = []
            for conv in short_memory[-self.max_groups:]:
                user_entry = {"role": "user", "content": conv["user"]}
                # 保留时间戳，供 LLMService._build_time_context 计算对话时间间隔
                if "timestamp" in conv:
                    user_entry["timestamp"] = conv["timestamp"]
                bot_entry = {"role": "assistant", "content": conv["bot"]}
                if "timestamp" in conv:
                    bot_entry["timestamp"] = conv["timestamp"]
                context.append(bot_entry)

            return context
        except Exception as e:
            logger.error(f"获取最近上下文失败: {str(e)}")
            return []

    def has_user_memory(self, avatar_name: str, user_id: str) -> bool:
        try:
            engine = self._get_engine(avatar_name, user_id)
            nodes = engine.storage.load_nodes(avatar_name, user_id)
            if nodes:
                return True
            return bool(self.get_core_memory(avatar_name, user_id))
        except Exception:
            return False

    # --------------------------------------------------------------------------
    # 动态记忆节点 SDK 操作方法 (提供给 WebUI 路由或开发接口调用)
    # --------------------------------------------------------------------------

    def get_all_nodes(self, avatar_name: str, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有记忆节点，附带实时算出的有效衰减权重"""
        engine = self._get_engine(avatar_name, user_id)
        nodes = engine.storage.load_nodes(avatar_name, user_id)
        res = []
        for n in nodes:
            d = n.to_dict()
            d['effective_weight'] = engine.decay_evaluator.evaluate_node(n)
            res.append(d)
        return res

    def add_custom_node(
        self, avatar_name: str, user_id: str, content: str, node_type: str = "fact",
        tags: Optional[List[str]] = None, importance: float = 0.8, confidence: float = 0.9
    ) -> Dict[str, Any]:
        """手动添加一个新的动态记忆节点"""
        import uuid
        engine = self._get_engine(avatar_name, user_id)

        try:
            e_type = EriiType(node_type.lower())
        except ValueError:
            e_type = EriiType.FACT

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_node = EriiNode(
            node_id=f"n_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            agent_id=avatar_name,
            node_type=e_type,
            content=content,
            tags=tags or [],
            base_importance=importance,
            emotional_score=0.0,
            confidence=confidence,
            created_at=ts,
            last_accessed_at=ts
        )

        nodes = engine.storage.load_nodes(avatar_name, user_id)
        nodes.append(new_node)
        engine.storage.save_nodes(avatar_name, user_id, nodes)
        return new_node.to_dict()

    def delete_node(self, avatar_name: str, user_id: str, node_id: str) -> bool:
        """删除指定 ID 的记忆节点"""
        engine = self._get_engine(avatar_name, user_id)
        nodes = engine.storage.load_nodes(avatar_name, user_id)
        updated = [n for n in nodes if n.node_id != node_id]
        if len(updated) != len(nodes):
            engine.storage.save_nodes(avatar_name, user_id, updated)
            return True
        return False

    def export_memory_pack(self, avatar_name: str, user_id: str, export_path: Optional[str] = None) -> MemoryPack:
        """导出 MemoryPack 记忆快照"""
        engine = self._get_engine(avatar_name, user_id)
        return engine.export_memory(agent_id=avatar_name, user_id=user_id, export_path=export_path)

    def import_memory_pack(self, avatar_name: str, user_id: str, pack_or_path: Any, overwrite: bool = False) -> MemoryPack:
        """导入 MemoryPack 记忆快照"""
        engine = self._get_engine(avatar_name, user_id)
        return engine.import_memory(pack_or_path=pack_or_path, agent_id=avatar_name, user_id=user_id, overwrite=overwrite)

    def get_timeline_entries(self, avatar_name: str, user_id: str, limit: int = 50) -> List[str]:
        """获取角色的第一人称体验时间线与心路感悟综合记录"""
        engine = self._get_engine(avatar_name, user_id)

        # 1. 如果存在旧版 timeline.jsonl 且数据库尚无记录，补全无感迁移
        timeline_path = self.storage.get_timeline_path(avatar_name, user_id)
        if os.path.exists(timeline_path):
            existing_timeline = engine.storage.get_recent_timeline(avatar_name, user_id, limit=1)
            if not existing_timeline:
                try:
                    with open(timeline_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                ts = record.get("timestamp", "")
                                entry_text = record.get("entry", "")
                                if entry_text:
                                    engine.storage.add_timeline_entry(
                                        agent_id=avatar_name,
                                        user_id=user_id,
                                        entry=entry_text,
                                        timestamp=ts
                                    )
                except Exception as e:
                    logger.error(f"补全迁移旧版 timeline.jsonl 失败: {e}")

        # 2. 获取体验时间线记录
        entries = engine.storage.get_recent_timeline(avatar_name, user_id, limit=limit)

        # 3. 结合 E.R.I.I 第一人称心理独白/日记节点 (THOUGHT/DIARY)
        diaries = engine.get_diary_timeline(avatar_name, user_id, limit=limit)
        for d in diaries:
            ts = d.get("created_at", "")
            content = d.get("content", "")
            if content:
                diary_line = f"[{ts}] (心路感悟) {content}"
                if diary_line not in entries:
                    entries.append(diary_line)

        return entries

