import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
import sqlite3
import aiosqlite
from datetime import datetime
from together_client import AsyncBatchClient, BatchRequest
from tqdm import tqdm

@dataclass
class BaseQueueMessage:
    priority: int = 0
    created_at: str = str(datetime.utcnow())
    processed_at: Optional[str] = None
    status: str = "pending"
    error: Optional[str] = None

@dataclass
class QueueMessage(BaseQueueMessage):
    content: Optional[str] = None
    result: Optional[str] = None

@dataclass
class SummarizationQueueMessage(BaseQueueMessage):
    text: Optional[str] = None
    summary: Optional[str] = None


class BaseQueue:
    def __init__(self, db_path: str, batch_size: int = 5):
        self.db_path = db_path
        self.batch_size = batch_size
        self.batch_client = AsyncBatchClient()
        self.setup_database()
    
    def setup_database(self):
        raise NotImplementedError

    def _hash_message(self, message: str) -> str:
        return hashlib.sha256(message.encode()).hexdigest()
    

# =============================================================================
# SummarizationQueue
# =============================================================================

class SummarizationQueue(BaseQueue):
    def __init__(self, db_path:str,  batch_size: int = 5):
        super().__init__(db_path, batch_size)
    
    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                language TEXT,
                prompt_type TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                processed_at TEXT,
                status TEXT DEFAULT 'pending',
                summary TEXT,
                error TEXT
            )
        """)
        conn.close()
    
    async def enqueue(self, texts: List[str], language: str, prompt_type:str,  priority: int = 0) -> List[str]:
        hashes = []
        async with aiosqlite.connect(self.db_path) as db:
            for text in texts:
                msg_hash = self._hash_message(text)
                hashes.append(msg_hash)
                
                # Check if message already exists
                cursor = await db.execute("SELECT hash FROM messages WHERE hash = ?", (msg_hash,))
                if not await cursor.fetchone():
                    queue_message = SummarizationQueueMessage(text=text, priority=priority)
                    await db.execute(
                        "INSERT INTO messages (hash, text, language, prompt_type, priority, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (msg_hash, text, language, prompt_type, priority, queue_message.created_at)
                    )
            await db.commit()
        return hashes
    
    async def _fetch_pending_messages(self, limit: int) -> AsyncGenerator[List[tuple], None]:
        async with aiosqlite.connect(self.db_path) as db:
            offset = 0
            while True:
                cursor = await db.execute(
                    """SELECT hash, text 
                       FROM messages 
                       WHERE status in ('pending' , 'error')
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
        
        async for pending_batch in self._fetch_pending_messages(1000):
            hashes, messages = zip(*pending_batch)
            
            for i in tqdm(range(0, len(messages), self.batch_size)):
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
                               SET status = ?, processed_at = ?, summary = ?, error = ?
                               WHERE hash = ?""",
                            (status, str(datetime.utcnow()), result_text, error, msg_hash)
                        )
                        
                        if status == "error":
                            total_errors += 1
                        else:
                            total_processed += 1
                    
                    await db.commit()

        return {"processed": total_processed, "errors": total_errors}

    async def get_batch_results(self, msg_hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        async with aiosqlite.connect(self.db_path) as db:
            for msg_hash in msg_hashes:
                cursor = await db.execute(
                    "SELECT text, language, prompt_type, summary, created_at, processed_at FROM messages WHERE hash = ?",
                    (msg_hash,)
                )
                row = await cursor.fetchone()
                
                if not row:
                    results[msg_hash] = None
                else:
                    results[msg_hash] = {
                        "text": row[0],
                        "language": row[1],
                        "prompt_type": row[2],
                        "summary": row[3],
                        "created_at": row[4],
                        "processed_at": row[5]
                    }
        return results

    async def get_all_prompt_types(self) -> List[tuple]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT DISTINCT prompt_type FROM messages")
            prompt_types = await cursor.fetchall()
        return prompt_types
# =============================================================================
# MessageQueue
# =============================================================================  

class MessageQueue(BaseQueue):
    def __init__(self, db_path: str, batch_size: int = 5):
        super().__init__(db_path, batch_size)

    def setup_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                hash TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                translation_direction TEXT,
                prompt_type TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                processed_at TEXT,
                status TEXT DEFAULT 'pending',
                result TEXT,
                error TEXT
            )
        """)
        conn.close()

    async def enqueue(self, messages: List[str], translation_direction: str, prompt_type:str,  priority: int = 0) -> List[str]:
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
                        "INSERT INTO messages (hash, content, translation_direction, prompt_type, priority, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (msg_hash, message,translation_direction, prompt_type, priority, queue_message.created_at)
                    )
            await db.commit()
        return hashes

    async def _fetch_pending_messages(self, limit: int) -> AsyncGenerator[List[tuple], None]:
        async with aiosqlite.connect(self.db_path) as db:
            offset = 0
            while True:
                cursor = await db.execute(
                    """SELECT hash, content 
                       FROM messages 
                       WHERE status in ('pending' , 'error')
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
        
        async for pending_batch in self._fetch_pending_messages(1000):
            hashes, messages = zip(*pending_batch)
            
            for i in tqdm(range(0, len(messages), self.batch_size)):
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

    async def get_result(self, msg_hash: Optional[str] = None,
                         prompt_type: Optional[str] = None,
                         translation_direction: Optional[str] = None) -> Optional[Dict[str, Any]]:

        # build query
        condition = []
        condition_values = []

        if msg_hash:
            condition.append("hash = ?")
            condition_values.append(msg_hash)
        if prompt_type:
            condition.append("prompt_type = ?")
            condition_values.append(prompt_type)
        if translation_direction:
            condition.append("translation_direction = ?")
            condition_values.append(translation_direction)

        query = "SELECT content, status, result, error, created_at, processed_at, translation_direction, prompt_type, hash FROM messages"
        if condition:
            query += " WHERE " + " AND ".join(condition)    

        print(query)
        print(condition_values)

        results = []
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                query, tuple(condition_values)
            )
            rows = await cursor.fetchall()
            
            if not rows:
                return None
            
            for row in rows:
                results.append({
                    "content": row[0],
                    "status": row[1],
                    "result": row[2],
                    "error": row[3],
                    "created_at": row[4],
                    "processed_at": row[5],
                    "translation_direction": row[6],
                    "prompt_type": row[7],
                    "hash": row[8]
                })
            
        return results
    
    async def get_batch_results(self, msg_hashes: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        async with aiosqlite.connect(self.db_path) as db:
            for msg_hash in msg_hashes:
                cursor = await db.execute(
                    "SELECT content, status, result, error, created_at, processed_at, translation_direction, prompt_type FROM messages WHERE hash = ?",
                    (msg_hash,)
                )
                row = await cursor.fetchone()
                
                if not row:
                    results[msg_hash] = None
                else:
                    results[msg_hash] = {
                        "content": row[0],
                        "status": row[1],
                        "result": row[2],
                        "error": row[3],
                        "created_at": row[4],
                        "processed_at": row[5],
                        "translation_direction": row[6],
                        "prompt_type": row[7]
                    }
        return results
    
    async def get_all_translation_direction_and_prompt_types(self) -> List[tuple]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT DISTINCT translation_direction FROM messages")
            translation_directions = await cursor.fetchall()

            cursor = await db.execute("SELECT DISTINCT prompt_type FROM messages")
            prompt_types = await cursor.fetchall()

        return translation_directions, prompt_types
