"""Metrics manager for observability and monitoring.

Provides centralized metrics emission for tracking:
- API call counts and latencies
- Success/failure rates
- Resource usage
- Business metrics (emails fetched, events processed, etc.)

TODO: Integration with monitoring systems:
- Prometheus metrics export
- CloudWatch metrics
- Datadog/New Relic integration
- OpenTelemetry support
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Metric:
    """Metric data point.

    Attributes:
        name: Metric name (e.g., "gmail_emails_fetched")
        value: Metric value (numeric)
        timestamp: When metric was recorded
        labels: Additional labels/tags for filtering
        unit: Metric unit (e.g., "count", "seconds", "bytes")
    """
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = "count"


class MetricsManager:
    """Centralized metrics collection and emission.

    Features:
    - In-memory metrics storage (with configurable retention)
    - Structured logging of all metrics
    - Aggregation support (sum, avg, min, max)
    - Export to monitoring systems (TODO)

    Example:
        ```python
        metrics = get_metrics_manager()

        # Emit a counter metric
        metrics.emit({
            "metric": "api_calls",
            "value": 1,
            "labels": {"service": "gmail", "status": "success"}
        })

        # Emit a timing metric
        metrics.emit({
            "metric": "fetch_duration",
            "value": 2.5,
            "unit": "seconds",
            "labels": {"service": "gmail"}
        })
        ```
    """

    def __init__(self, max_metrics_in_memory: int = 10000):
        """Initialize metrics manager.

        Args:
            max_metrics_in_memory: Maximum metrics to keep in memory (FIFO)
        """
        self.max_metrics = max_metrics_in_memory
        self.metrics: List[Metric] = []
        logger.info("metrics_manager_initialized", max_metrics=max_metrics_in_memory)

    def emit(self, metric_data: Dict[str, Any]) -> None:
        """Emit a metric.

        Args:
            metric_data: Metric dictionary with keys:
                - metric (str): Metric name
                - value (float): Metric value
                - timestamp (str, optional): ISO format timestamp
                - labels (dict, optional): Labels/tags
                - unit (str, optional): Metric unit
        """
        try:
            # Parse metric data
            name = metric_data.get("metric", "unknown")
            value = float(metric_data.get("value", 0))
            labels = metric_data.get("labels", {})
            unit = metric_data.get("unit", "count")

            # Parse timestamp
            timestamp_str = metric_data.get("timestamp")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = datetime.now()

            # Create metric
            metric = Metric(
                name=name,
                value=value,
                timestamp=timestamp,
                labels=labels,
                unit=unit
            )

            # Store in memory (with FIFO eviction)
            self.metrics.append(metric)
            if len(self.metrics) > self.max_metrics:
                self.metrics.pop(0)

            # Log metric for observability
            logger.debug(
                "metric_emitted",
                metric=name,
                value=value,
                unit=unit,
                labels=labels
            )

            # TODO: Push to external monitoring system
            # self._export_to_prometheus(metric)
            # self._export_to_cloudwatch(metric)

        except Exception as e:
            logger.warning("metric_emit_failed", error=str(e), metric_data=metric_data)

    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        limit: int = 100
    ) -> List[Metric]:
        """Retrieve metrics from memory.

        Args:
            metric_name: Filter by metric name
            labels: Filter by labels (partial match)
            limit: Maximum metrics to return

        Returns:
            List of matching metrics (most recent first)
        """
        filtered = self.metrics

        # Filter by metric name
        if metric_name:
            filtered = [m for m in filtered if m.name == metric_name]

        # Filter by labels
        if labels:
            filtered = [
                m for m in filtered
                if all(m.labels.get(k) == v for k, v in labels.items())
            ]

        # Return most recent first
        return list(reversed(filtered))[:limit]

    def aggregate(
        self,
        metric_name: str,
        operation: str = "sum",
        labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Aggregate metrics.

        Args:
            metric_name: Metric to aggregate
            operation: Aggregation operation ("sum", "avg", "min", "max", "count")
            labels: Filter by labels

        Returns:
            Aggregated value
        """
        metrics = self.get_metrics(metric_name=metric_name, labels=labels, limit=None)

        if not metrics:
            return 0.0

        values = [m.value for m in metrics]

        if operation == "sum":
            return sum(values)
        elif operation == "avg":
            return sum(values) / len(values)
        elif operation == "min":
            return min(values)
        elif operation == "max":
            return max(values)
        elif operation == "count":
            return float(len(values))
        else:
            raise ValueError(f"Unknown aggregation operation: {operation}")

    def clear_metrics(self) -> None:
        """Clear all metrics from memory."""
        self.metrics.clear()
        logger.info("metrics_cleared")


# Singleton instance
_metrics_manager_instance: Optional[MetricsManager] = None


def get_metrics_manager() -> MetricsManager:
    """Get singleton MetricsManager instance.

    Returns:
        Global MetricsManager instance
    """
    global _metrics_manager_instance

    if _metrics_manager_instance is None:
        _metrics_manager_instance = MetricsManager()

    return _metrics_manager_instance
