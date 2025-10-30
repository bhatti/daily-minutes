"""Async base models with async/await support for performance."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import Field

from src.models.base import BaseModel

T = TypeVar("T")


class AsyncBaseModel(BaseModel, ABC, Generic[T]):
    """Base model with async support for high-performance operations."""

    # Async operation tracking
    is_processing: bool = Field(False, description="Currently processing flag")
    last_async_operation: Optional[datetime] = Field(None, description="Last async operation time")
    async_operations_count: int = Field(0, ge=0, description="Total async operations")

    @abstractmethod
    async def fetch_async(self) -> T:
        """Async fetch operation to be implemented by subclasses."""
        pass

    @abstractmethod
    async def process_async(self, data: T) -> Any:
        """Async process operation to be implemented by subclasses."""
        pass

    async def execute_async(self) -> Any:
        """Execute async operation with tracking."""
        self.is_processing = True
        self.update_timestamp()

        try:
            # Fetch data
            data = await self.fetch_async()

            # Process data
            result = await self.process_async(data)

            # Update tracking
            self.last_async_operation = datetime.now()
            self.async_operations_count += 1

            return result

        finally:
            self.is_processing = False
            self.update_timestamp()

    async def execute_with_timeout(self, timeout: float = 30.0) -> Any:
        """Execute async operation with timeout."""
        try:
            return await asyncio.wait_for(
                self.execute_async(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Operation timed out after {timeout} seconds")

    async def execute_with_retry(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0
    ) -> Any:
        """Execute async operation with exponential backoff retry."""
        last_exception = None
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                return await self.execute_async()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
                else:
                    raise last_exception

    @staticmethod
    async def gather_results(*coroutines, return_exceptions: bool = False) -> list:
        """Gather results from multiple coroutines."""
        return await asyncio.gather(*coroutines, return_exceptions=return_exceptions)

    @staticmethod
    async def run_in_parallel(tasks: list, max_concurrent: int = 10) -> list:
        """Run tasks in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_task(task):
            async with semaphore:
                return await task

        return await asyncio.gather(
            *[bounded_task(task) for task in tasks],
            return_exceptions=True
        )


class AsyncBatchProcessor(AsyncBaseModel[list]):
    """Async batch processor for handling multiple items efficiently."""

    batch_size: int = Field(100, gt=0, description="Batch size")
    items_processed: int = Field(0, ge=0, description="Total items processed")
    batches_processed: int = Field(0, ge=0, description="Total batches processed")

    # Performance metrics
    average_batch_time: float = Field(0.0, ge=0.0, description="Average batch processing time")
    last_batch_time: float = Field(0.0, ge=0.0, description="Last batch processing time")

    async def fetch_async(self) -> list:
        """Fetch batch of items - to be implemented by subclasses."""
        raise NotImplementedError

    async def process_async(self, data: list) -> Any:
        """Process batch of items - to be implemented by subclasses."""
        raise NotImplementedError

    async def process_item_async(self, item: Any) -> Any:
        """Process single item - to be implemented by subclasses."""
        raise NotImplementedError

    async def process_batch(self, items: list) -> list:
        """Process a batch of items in parallel."""
        start_time = datetime.now()

        # Process items in parallel
        tasks = [self.process_item_async(item) for item in items]
        results = await self.run_in_parallel(tasks, max_concurrent=10)

        # Update metrics
        batch_time = (datetime.now() - start_time).total_seconds()
        self.last_batch_time = batch_time
        self.batches_processed += 1
        self.items_processed += len(items)

        # Update average
        self.average_batch_time = (
            (self.average_batch_time * (self.batches_processed - 1) + batch_time) /
            self.batches_processed
        )

        self.update_timestamp()
        return results

    async def process_all_batches(self, all_items: list) -> list:
        """Process all items in batches."""
        results = []

        for i in range(0, len(all_items), self.batch_size):
            batch = all_items[i:i + self.batch_size]
            batch_results = await self.process_batch(batch)
            results.extend(batch_results)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get batch processing performance statistics."""
        throughput = (
            self.items_processed / (self.batches_processed * self.average_batch_time)
            if self.batches_processed > 0 and self.average_batch_time > 0
            else 0
        )

        return {
            "items_processed": self.items_processed,
            "batches_processed": self.batches_processed,
            "average_batch_time": f"{self.average_batch_time:.2f}s",
            "last_batch_time": f"{self.last_batch_time:.2f}s",
            "throughput": f"{throughput:.2f} items/sec",
            "batch_size": self.batch_size,
        }


class AsyncStreamProcessor(AsyncBaseModel[Any]):
    """Async stream processor for handling real-time data streams."""

    stream_id: str = Field(..., description="Stream identifier")
    is_connected: bool = Field(False, description="Stream connection status")

    # Stream metrics
    messages_received: int = Field(0, ge=0, description="Total messages received")
    messages_processed: int = Field(0, ge=0, description="Total messages processed")
    messages_failed: int = Field(0, ge=0, description="Total messages failed")

    last_message_time: Optional[datetime] = Field(None, description="Last message timestamp")
    connection_start: Optional[datetime] = Field(None, description="Connection start time")

    # Buffer
    buffer_size: int = Field(1000, gt=0, description="Buffer size")
    buffer: list = Field(default_factory=list, description="Message buffer")

    async def connect_async(self) -> None:
        """Connect to stream - to be implemented by subclasses."""
        self.is_connected = True
        self.connection_start = datetime.now()
        self.update_timestamp()

    async def disconnect_async(self) -> None:
        """Disconnect from stream."""
        self.is_connected = False
        self.update_timestamp()

    async def fetch_async(self) -> Any:
        """Fetch from stream - to be implemented by subclasses."""
        raise NotImplementedError

    async def process_async(self, data: Any) -> Any:
        """Process stream message - to be implemented by subclasses."""
        raise NotImplementedError

    async def consume_stream(self, max_messages: Optional[int] = None) -> None:
        """Consume messages from stream."""
        if not self.is_connected:
            await self.connect_async()

        messages_consumed = 0

        try:
            while self.is_connected:
                if max_messages and messages_consumed >= max_messages:
                    break

                try:
                    # Fetch message with timeout
                    message = await asyncio.wait_for(
                        self.fetch_async(),
                        timeout=5.0
                    )

                    if message:
                        self.messages_received += 1
                        self.last_message_time = datetime.now()

                        # Add to buffer
                        self.buffer.append(message)
                        if len(self.buffer) > self.buffer_size:
                            self.buffer.pop(0)

                        # Process message
                        try:
                            await self.process_async(message)
                            self.messages_processed += 1
                        except Exception as e:
                            self.messages_failed += 1
                            # Log error but continue consuming
                            print(f"Failed to process message: {e}")

                        messages_consumed += 1

                except asyncio.TimeoutError:
                    # No message available, continue
                    await asyncio.sleep(0.1)

        finally:
            await self.disconnect_async()

    def get_stream_stats(self) -> Dict[str, Any]:
        """Get stream processing statistics."""
        uptime = (
            (datetime.now() - self.connection_start).total_seconds()
            if self.connection_start and self.is_connected
            else 0
        )

        success_rate = (
            self.messages_processed / self.messages_received
            if self.messages_received > 0
            else 0
        )

        return {
            "stream_id": self.stream_id,
            "is_connected": self.is_connected,
            "uptime_seconds": uptime,
            "messages_received": self.messages_received,
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "success_rate": f"{success_rate:.2%}",
            "buffer_usage": f"{len(self.buffer)}/{self.buffer_size}",
            "last_message": self.last_message_time.isoformat() if self.last_message_time else None,
        }


class AsyncAggregator(AsyncBaseModel[Dict[str, Any]]):
    """Async aggregator for combining data from multiple sources."""

    sources: List[str] = Field(..., description="Data sources to aggregate")
    aggregation_strategy: str = Field("merge", description="Aggregation strategy")

    # Aggregation metrics
    sources_fetched: int = Field(0, ge=0, description="Sources successfully fetched")
    sources_failed: int = Field(0, ge=0, description="Sources that failed")
    last_aggregation: Optional[datetime] = Field(None, description="Last aggregation time")

    async def fetch_from_source(self, source: str) -> Any:
        """Fetch data from a single source - to be implemented."""
        raise NotImplementedError

    async def fetch_async(self) -> Dict[str, Any]:
        """Fetch from all sources in parallel."""
        tasks = [self.fetch_from_source(source) for source in self.sources]
        results = await self.gather_results(*tasks, return_exceptions=True)

        aggregated = {}
        for source, result in zip(self.sources, results):
            if isinstance(result, Exception):
                self.sources_failed += 1
                aggregated[source] = {"error": str(result)}
            else:
                self.sources_fetched += 1
                aggregated[source] = result

        self.last_aggregation = datetime.now()
        self.update_timestamp()
        return aggregated

    async def process_async(self, data: Dict[str, Any]) -> Any:
        """Process aggregated data based on strategy."""
        if self.aggregation_strategy == "merge":
            # Simple merge
            merged = {}
            for source_data in data.values():
                if isinstance(source_data, dict) and "error" not in source_data:
                    merged.update(source_data)
            return merged

        elif self.aggregation_strategy == "combine":
            # Combine into list
            combined = []
            for source_data in data.values():
                if "error" not in source_data:
                    combined.append(source_data)
            return combined

        else:
            # Custom strategy - to be implemented
            return data

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get aggregation statistics."""
        success_rate = (
            self.sources_fetched / (self.sources_fetched + self.sources_failed)
            if (self.sources_fetched + self.sources_failed) > 0
            else 0
        )

        return {
            "total_sources": len(self.sources),
            "sources_fetched": self.sources_fetched,
            "sources_failed": self.sources_failed,
            "success_rate": f"{success_rate:.2%}",
            "aggregation_strategy": self.aggregation_strategy,
            "last_aggregation": self.last_aggregation.isoformat() if self.last_aggregation else None,
        }