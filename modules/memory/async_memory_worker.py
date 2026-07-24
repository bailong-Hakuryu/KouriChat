"""Asynchronous Worker Queue for Post-Dialogue Memory Archiving.

Decouples response generation from background memory processing to ensure zero
response latency. Provides per-user locking to prevent concurrent write races.

Follows Google Python Style Guide.
"""

import logging
import queue
import threading
import time
from typing import Callable, Dict
from modules.memory.memory_decay_engine import MemoryDecayEngine

logger = logging.getLogger('main')


class AsyncMemoryWorker:
    """Background worker thread processing post-dialogue memory archiving tasks."""

    def __init__(self, decay_engine: MemoryDecayEngine, update_memory_callback: Callable):
        self.decay_engine = decay_engine
        self.update_memory_callback = update_memory_callback
        self.task_queue: queue.Queue = queue.Queue()
        self.user_locks: Dict[str, threading.Lock] = {}
        self.global_lock = threading.Lock()
        self.is_running = True

        self.worker_thread = threading.Thread(
            target=self._process_queue, name="AsyncMemoryWorkerThread", daemon=True
        )
        self.worker_thread.start()
        logger.info("AsyncMemoryWorker background thread started.")

    def _get_user_lock(self, user_key: str) -> threading.Lock:
        with self.global_lock:
            if user_key not in self.user_locks:
                self.user_locks[user_key] = threading.Lock()
            return self.user_locks[user_key]

    def push_task(self, avatar_name: str, user_id: str, user_msg: str, bot_reply: str):
        if not user_msg or not bot_reply or bot_reply.startswith("Error:"):
            return

        task = {
            "avatar_name": avatar_name,
            "user_id": user_id,
            "user_msg": user_msg,
            "bot_reply": bot_reply,
            "timestamp": time.time()
        }
        self.task_queue.put(task)
        logger.debug(f"Pushed async memory task for user {user_id} to queue (Queue size: {self.task_queue.qsize()})")

    def _process_queue(self):
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=2.0)
            except queue.Empty:
                continue

            avatar_name = task["avatar_name"]
            user_id = task["user_id"]
            user_msg = task["user_msg"]
            bot_reply = task["bot_reply"]

            user_key = f"{avatar_name}_{user_id}"
            user_lock = self._get_user_lock(user_key)

            with user_lock:
                try:
                    logger.debug(f"Executing async memory analysis for user {user_id}...")
                    user_state, emotional_score, importance_score, timeline_entry, extracted_nodes = (
                        self.decay_engine.analyze_dialogue_round(
                            avatar_name, user_id, user_msg, bot_reply
                        )
                    )

                    self.update_memory_callback(
                        avatar_name=avatar_name,
                        user_id=user_id,
                        user_state=user_state,
                        emotional_score=emotional_score,
                        importance_score=importance_score,
                        timeline_entry=timeline_entry,
                        extracted_nodes=extracted_nodes
                    )
                except Exception as e:
                    logger.error(f"Error processing async memory task for {user_id}: {e}", exc_info=True)
                finally:
                    self.task_queue.task_done()

    def shutdown(self):
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3.0)
        logger.info("AsyncMemoryWorker shut down.")
