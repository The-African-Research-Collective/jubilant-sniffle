import os
import asyncio
from dataclasses import dataclass
from typing import List, Any, Optional, Dict
from together import AsyncTogether
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

@dataclass
class BatchRequest:
    messages: List[Dict[str, str]]
    model: str
    temperature: float = 0.0
    max_batch_size: int = 32
    timeout: int = 30000
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_tokens: Optional[int] = None


class TogetherAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API Error {status_code}: {message}")

class ErrorCode(Enum):
    INVALID_REQUEST = 400
    AUTH_ERROR = 401
    PAYMENT_REQUIRED = 402
    BAD_REQUEST = 403
    NOT_FOUND = 404
    RATE_LIMIT = 429
    SERVER_ERROR = 500
    ENGINE_OVERLOADED = 503

    @classmethod
    def get_error_message(cls, code: int) -> str:
        error_messages = {
            400: "Invalid request configuration. Check JSON format, API key, and prompt format.",
            401: "Authentication failed. Check API key.",
            402: "Account reached spending limit.",
            403: "Token count exceeds model context length.",
            404: "Invalid endpoint URL or model name.",
            429: "Rate limit exceeded. Please throttle requests.",
            500: "Server error. Please retry after a brief wait.",
            503: "Servers overloaded. Please retry after a brief wait."
        }
        return error_messages.get(code, "Unknown error occurred")

class AsyncBatchClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.client = AsyncTogether(api_key=self.api_key)
    
    async def _handle_error(self, status_code: int) -> None:
        if status_code == ErrorCode.RATE_LIMIT.value:
            await asyncio.sleep(self.retry_delay)
        elif status_code in [ErrorCode.SERVER_ERROR.value, ErrorCode.ENGINE_OVERLOADED.value]:
            await asyncio.sleep(self.retry_delay * 2)
        else:
            raise TogetherAPIError(status_code, ErrorCode.get_error_message(status_code))

    async def _process_single_request(self, message: str, model: str, max_tokens: Optional[int], attempt: int = 0) -> Any:
        try:
            return await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                max_tokens=max_tokens
            )
        except Exception as e:
            status_code = getattr(e, 'status', 500)
            
            if attempt < self.retry_attempts and status_code in [429, 500, 503]:
                await self._handle_error(status_code)
                return await self._process_single_request(message, model, max_tokens, attempt + 1)
            
            raise TogetherAPIError(status_code, ErrorCode.get_error_message(status_code))

    async def _process_batch(self, batch: List[str], model: str, max_tokens: Optional[int]) -> List[Any]:
        tasks = [
            self._process_single_request(message, model, max_tokens)
            for message in batch
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def process_requests(self, batch_request: BatchRequest) -> List[str]:
        results = []
        
        for i in range(0, len(batch_request.messages), batch_request.max_batch_size):
            batch = batch_request.messages[i:i + batch_request.max_batch_size]
            
            try:
                async with asyncio.timeout(batch_request.timeout):
                    responses = await self._process_batch(batch, batch_request.model, batch_request.max_tokens)
                    
                    for response in responses:
                        if isinstance(response, Exception):
                            results.append(f"Error: {str(response)}")
                        else:
                            results.append(response.choices[0].message.content)
                            
            except asyncio.TimeoutError:
                results.extend([f"Timeout error for message: {msg}" for msg in batch])
            except Exception as e:
                results.extend([f"Error processing message: {msg}, Error: {str(e)}" for msg in batch])
                
        return results