import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import sqlite3
import aiosqlite
from datetime import datetime
from together_client import AsyncBatchClient, BatchRequest

@dataclass
class QueueMessage:
    content: str
    priority: int = 0
    created_at: str = str(datetime.utcnow())
    processed_at: Optional[str] = None
    status: str = "pending"
    result: Optional[str] = None
    error: Optional[str] = None

class MessageQueue:
    def __init__(self, db_path: str = "message_queue.db", batch_size: int = 5):
        self.db_path = db_path
        self.batch_size = batch_size
        self.batch_client = AsyncBatchClient()
        self.setup_database()

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                processed_at TEXT,
                status TEXT DEFAULT 'pending',
                result TEXT,
                error TEXT
            )
        """)
        conn.close()

    def _hash_message(self, message: str) -> str:
        return hashlib.sha256(message.encode()).hexdigest()

    async def enqueue(self, messages: List[str], priority: int = 0) -> List[str]:
        hashes = []
        async with aiosqlite.connect(self.db_path) as db:
            for message in messages:
                msg_hash = self._hash_message(message)
                hashes.append(msg_hash)
                
                # Check if message already exists
                cursor = await db.execute("SELECT hash FROM messages WHERE hash = ?", (msg_hash,))
                if not await cursor.fetchone():
                    queue_message = QueueMessage(content=message, priority=priority)
                    await db.execute(
                        "INSERT INTO messages (hash, content, priority, created_at) VALUES (?, ?, ?, ?)",
                        (msg_hash, message, priority, queue_message.created_at)
                    )
            await db.commit()
        return hashes

    async def _fetch_pending_messages(self, limit: int = 32) -> AsyncGenerator[List[tuple], None]:
        async with aiosqlite.connect(self.db_path) as db:
            offset = 0
            while True:
                cursor = await db.execute(
                    """SELECT hash, content 
                       FROM messages 
                       WHERE status = 'pending' 
                       ORDER BY priority DESC, created_at ASC 
                       LIMIT ? OFFSET ?""",
                    (limit, offset)
                )
                batch = await cursor.fetchall()
                if not batch:
                    break
                yield batch
                offset += limit

    async def process_queue(self, model_name:str) -> Dict[str, Any]:
        total_processed = total_errors = 0
        
        async for pending_batch in self._fetch_pending_messages(32):
            hashes, messages = zip(*pending_batch)
            
            for i in range(0, len(messages), self.batch_size):
                batch_slice = slice(i, i + self.batch_size)
                current_hashes = hashes[batch_slice]
                current_messages = messages[batch_slice]
                
                batch_request = BatchRequest(messages=list(current_messages), model=model_name)
                results = await self.batch_client.process_requests(batch_request)
                
                async with aiosqlite.connect(self.db_path) as db:
                    for msg_hash, result in zip(current_hashes, results):
                        status = "error" if isinstance(result, str) and result.startswith("Error") else "completed"
                        error = result if status == "error" else None
                        result_text = None if status == "error" else result
                        
                        await db.execute(
                            """UPDATE messages 
                               SET status = ?, processed_at = ?, result = ?, error = ?
                               WHERE hash = ?""",
                            (status, str(datetime.utcnow()), result_text, error, msg_hash)
                        )
                        
                        if status == "error":
                            total_errors += 1
                        else:
                            total_processed += 1
                    
                    await db.commit()

        return {"processed": total_processed, "errors": total_errors}

    async def get_result(self, msg_hash: str) -> Optional[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT content, status, result, error, created_at, processed_at FROM messages WHERE hash = ?",
                (msg_hash,)
            )
            row = await cursor.fetchone()
            
            if not row:
                return None
                
            return {
                "content": row[0],
                "status": row[1],
                "result": row[2],
                "error": row[3],
                "created_at": row[4],
                "processed_at": row[5]
            }