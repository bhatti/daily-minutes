"""Observability Service for tracking metrics, errors, and performance."""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import asyncio
from dataclasses import dataclass, field
import json

from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ErrorRecord:
    """Error tracking record."""
    timestamp: datetime
    error_type: str
    message: str
    component: str
    traceback: Optional[str] = None
    count: int = 1


@dataclass
class LatencyRecord:
    """Latency tracking record."""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    tags: Dict[str, str] = field(default_factory=dict)


class ObservabilityService:
    """Service for comprehensive observability."""

    def __init__(self, max_history: int = 1000):
        """Initialize observability service.

        Args:
            max_history: Maximum number of records to keep in memory
        """
        self.max_history = max_history

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))

        # Error tracking
        self.errors: deque = deque(maxlen=max_history)
        self.error_counts: Dict[str, int] = defaultdict(int)

        # Latency tracking
        self.latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.active_operations: Dict[str, float] = {}

        # Component health
        self.component_health: Dict[str, Dict[str, Any]] = {}

        # Request tracking
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0

        # Service start time
        self.start_time = datetime.now()

        logger.info("observability_service_initialized")

    def track_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Track a metric value.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        self.metrics[name].append(point)
        logger.debug("metric_tracked", name=name, value=value, tags=tags)

    def track_error(self, error: Exception, component: str, traceback_str: Optional[str] = None):
        """Track an error occurrence.

        Args:
            error: The exception that occurred
            component: Component where error occurred
            traceback_str: Optional traceback string
        """
        error_type = type(error).__name__
        message = str(error)

        record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=error_type,
            message=message,
            component=component,
            traceback=traceback_str
        )

        self.errors.append(record)
        self.error_counts[f"{component}.{error_type}"] += 1
        self.failure_count += 1

        logger.error("error_tracked",
                    component=component,
                    error_type=error_type,
                    message=message)

    def start_operation(self, operation: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Start tracking an operation's latency.

        Args:
            operation: Operation name
            tags: Optional tags

        Returns:
            Operation ID for ending the operation
        """
        op_id = f"{operation}_{time.time()}"
        self.active_operations[op_id] = time.time()
        return op_id

    def end_operation(self, op_id: str, success: bool = True):
        """End tracking an operation's latency.

        Args:
            op_id: Operation ID from start_operation
            success: Whether operation succeeded
        """
        if op_id not in self.active_operations:
            return

        start_time = self.active_operations.pop(op_id)
        duration_ms = (time.time() - start_time) * 1000
        operation = op_id.rsplit('_', 1)[0]

        record = LatencyRecord(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            success=success
        )

        self.latencies[operation].append(record)

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.request_count += 1

        logger.debug("operation_completed",
                    operation=operation,
                    duration_ms=duration_ms,
                    success=success)

    def update_component_health(self, component: str, healthy: bool, details: Optional[Dict[str, Any]] = None):
        """Update component health status.

        Args:
            component: Component name
            healthy: Whether component is healthy
            details: Optional health details
        """
        self.component_health[component] = {
            "healthy": healthy,
            "last_check": datetime.now(),
            "details": details or {}
        }

        logger.info("component_health_updated",
                   component=component,
                   healthy=healthy)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics.

        Returns:
            Dictionary with metrics summary
        """
        summary = {}

        for metric_name, points in self.metrics.items():
            if points:
                values = [p.value for p in points]
                recent_points = list(points)[-10:]  # Last 10 points

                summary[metric_name] = {
                    "current": values[-1],
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                    "recent": [
                        {
                            "timestamp": p.timestamp.isoformat(),
                            "value": p.value
                        }
                        for p in recent_points
                    ]
                }

        return summary

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors.

        Returns:
            Dictionary with error summary
        """
        recent_errors = list(self.errors)[-10:]  # Last 10 errors

        return {
            "total_errors": sum(self.error_counts.values()),
            "unique_errors": len(self.error_counts),
            "error_counts": dict(self.error_counts),
            "recent_errors": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "type": e.error_type,
                    "message": e.message,
                    "component": e.component
                }
                for e in recent_errors
            ]
        }

    def get_latency_stats(self) -> Dict[str, Any]:
        """Get latency statistics.

        Returns:
            Dictionary with latency stats
        """
        stats = {}

        for operation, records in self.latencies.items():
            if records:
                latencies = [r.duration_ms for r in records]
                success_count = sum(1 for r in records if r.success)

                stats[operation] = {
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "p50_ms": self._percentile(latencies, 50),
                    "p95_ms": self._percentile(latencies, 95),
                    "p99_ms": self._percentile(latencies, 99),
                    "count": len(records),
                    "success_rate": success_count / len(records)
                }

        return stats

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health.

        Returns:
            Dictionary with system health info
        """
        uptime = datetime.now() - self.start_time
        success_rate = self.success_count / self.request_count if self.request_count > 0 else 0

        return {
            "uptime_seconds": uptime.total_seconds(),
            "total_requests": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "components": {
                name: {
                    "status": "healthy" if info["healthy"] else "unhealthy",
                    "last_check": info["last_check"].isoformat(),
                    **info.get("details", {})
                }
                for name, info in self.component_health.items()
            }
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data for dashboard display.

        Returns:
            Complete dashboard data
        """
        return {
            "system_health": self.get_system_health(),
            "metrics": self.get_metrics_summary(),
            "errors": self.get_error_summary(),
            "latencies": self.get_latency_stats(),
            "timestamp": datetime.now().isoformat()
        }

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        # System metrics
        lines.append(f"# HELP daily_minutes_uptime_seconds Uptime in seconds")
        lines.append(f"# TYPE daily_minutes_uptime_seconds gauge")
        uptime = (datetime.now() - self.start_time).total_seconds()
        lines.append(f"daily_minutes_uptime_seconds {uptime}")

        lines.append(f"# HELP daily_minutes_requests_total Total requests")
        lines.append(f"# TYPE daily_minutes_requests_total counter")
        lines.append(f"daily_minutes_requests_total {self.request_count}")

        lines.append(f"# HELP daily_minutes_success_rate Success rate")
        lines.append(f"# TYPE daily_minutes_success_rate gauge")
        success_rate = self.success_count / self.request_count if self.request_count > 0 else 0
        lines.append(f"daily_minutes_success_rate {success_rate}")

        # Custom metrics
        for metric_name, points in self.metrics.items():
            if points:
                current_value = points[-1].value
                metric_name_clean = metric_name.replace(".", "_").replace("-", "_")
                lines.append(f"# HELP daily_minutes_{metric_name_clean} {metric_name}")
                lines.append(f"# TYPE daily_minutes_{metric_name_clean} gauge")
                lines.append(f"daily_minutes_{metric_name_clean} {current_value}")

        # Error counts
        lines.append(f"# HELP daily_minutes_errors_total Total errors by type")
        lines.append(f"# TYPE daily_minutes_errors_total counter")
        for error_key, count in self.error_counts.items():
            component, error_type = error_key.rsplit(".", 1)
            lines.append(f'daily_minutes_errors_total{{component="{component}",type="{error_type}"}} {count}')

        # Latencies
        for operation, records in self.latencies.items():
            if records:
                latencies = [r.duration_ms for r in records]
                op_clean = operation.replace(".", "_").replace("-", "_")

                lines.append(f"# HELP daily_minutes_{op_clean}_duration_ms Operation duration in ms")
                lines.append(f"# TYPE daily_minutes_{op_clean}_duration_ms summary")
                lines.append(f"daily_minutes_{op_clean}_duration_ms{{quantile=\"0.5\"}} {self._percentile(latencies, 50)}")
                lines.append(f"daily_minutes_{op_clean}_duration_ms{{quantile=\"0.95\"}} {self._percentile(latencies, 95)}")
                lines.append(f"daily_minutes_{op_clean}_duration_ms{{quantile=\"0.99\"}} {self._percentile(latencies, 99)}")
                lines.append(f"daily_minutes_{op_clean}_duration_ms_sum {sum(latencies)}")
                lines.append(f"daily_minutes_{op_clean}_duration_ms_count {len(latencies)}")

        return "\n".join(lines)

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values.

        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]

    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        self.metrics.clear()
        self.errors.clear()
        self.error_counts.clear()
        self.latencies.clear()
        self.active_operations.clear()
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.start_time = datetime.now()
        logger.info("metrics_reset")


# Global instance
_observability_service: Optional[ObservabilityService] = None


def get_observability_service() -> ObservabilityService:
    """Get or create observability service instance.

    Returns:
        ObservabilityService instance
    """
    global _observability_service
    if _observability_service is None:
        _observability_service = ObservabilityService()
    return _observability_service