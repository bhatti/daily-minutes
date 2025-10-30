"""Models for observability, metrics, and tracking."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from src.models.base import BaseModel, DataSource


class MetricType(str, Enum):
    """Metric type enum."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class SpanStatus(str, Enum):
    """Span status for distributed tracing."""

    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


class Metric(BaseModel):
    """Model for a single metric."""

    name: str = Field(..., description="Metric name")
    metric_type: MetricType = Field(..., description="Metric type")

    value: float = Field(0.0, description="Current value")
    unit: Optional[str] = Field(None, description="Unit of measurement")

    tags: Dict[str, str] = Field(default_factory=dict, description="Metric tags")
    labels: Dict[str, str] = Field(default_factory=dict, description="Metric labels")

    # Statistics for complex metrics
    count: int = Field(0, ge=0, description="Number of observations")
    sum: float = Field(0.0, description="Sum of values")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")

    # Time window
    window_start: datetime = Field(default_factory=datetime.now, description="Window start")
    window_duration_seconds: int = Field(60, gt=0, description="Window duration")

    # Histogram buckets
    buckets: List[float] = Field(default_factory=list, description="Histogram buckets")
    bucket_counts: List[int] = Field(default_factory=list, description="Counts per bucket")

    def increment(self, value: float = 1.0) -> None:
        """Increment metric value."""
        if self.metric_type == MetricType.COUNTER:
            self.value += value
            self.count += 1
            self.sum += value
        elif self.metric_type == MetricType.GAUGE:
            self.value = value
            self.count += 1
        self.update_timestamp()

    def record(self, value: float) -> None:
        """Record a value for the metric."""
        self.count += 1
        self.sum += value

        if self.min_value is None or value < self.min_value:
            self.min_value = value

        if self.max_value is None or value > self.max_value:
            self.max_value = value

        if self.metric_type == MetricType.HISTOGRAM and self.buckets:
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    if i < len(self.bucket_counts):
                        self.bucket_counts[i] += 1
                    break

        self.value = value
        self.update_timestamp()

    def get_average(self) -> float:
        """Get average value."""
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def get_rate(self) -> float:
        """Get rate per second."""
        if self.metric_type != MetricType.RATE:
            return 0.0

        window_elapsed = (datetime.now() - self.window_start).total_seconds()
        if window_elapsed == 0:
            return 0.0

        return self.value / window_elapsed

    def reset(self) -> None:
        """Reset metric values."""
        self.value = 0.0
        self.count = 0
        self.sum = 0.0
        self.min_value = None
        self.max_value = None
        self.bucket_counts = [0] * len(self.buckets)
        self.window_start = datetime.now()
        self.update_timestamp()

    def to_prometheus_format(self) -> str:
        """Export metric in Prometheus format."""
        labels_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
        base_name = f"{self.name}{{{labels_str}}}" if labels_str else self.name

        if self.metric_type == MetricType.COUNTER:
            return f"{base_name} {self.value}"
        elif self.metric_type == MetricType.GAUGE:
            return f"{base_name} {self.value}"
        elif self.metric_type == MetricType.HISTOGRAM:
            lines = []
            for i, bucket in enumerate(self.buckets):
                count = self.bucket_counts[i] if i < len(self.bucket_counts) else 0
                lines.append(f'{self.name}_bucket{{le="{bucket}"}} {count}')
            lines.append(f"{self.name}_sum {self.sum}")
            lines.append(f"{self.name}_count {self.count}")
            return "\n".join(lines)
        else:
            return f"{base_name} {self.value}"


class Span(BaseModel):
    """Model for distributed tracing span."""

    span_id: str = Field(..., description="Unique span ID")
    trace_id: str = Field(..., description="Trace ID")
    parent_span_id: Optional[str] = Field(None, description="Parent span ID")

    operation_name: str = Field(..., description="Operation name")
    service_name: str = Field(..., description="Service name")

    start_time: datetime = Field(default_factory=datetime.now, description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")

    status: SpanStatus = Field(SpanStatus.RUNNING, description="Span status")

    tags: Dict[str, Any] = Field(default_factory=dict, description="Span tags")
    logs: List[Dict[str, Any]] = Field(default_factory=list, description="Span logs")
    baggage: Dict[str, str] = Field(default_factory=dict, description="Span baggage")

    # Error information
    error: bool = Field(False, description="Error flag")
    error_message: Optional[str] = Field(None, description="Error message")
    error_type: Optional[str] = Field(None, description="Error type")

    # Performance
    cpu_usage: Optional[float] = Field(None, description="CPU usage")
    memory_usage: Optional[float] = Field(None, description="Memory usage (MB)")

    def finish(self, status: SpanStatus = SpanStatus.SUCCESS) -> None:
        """Finish the span."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = status
        self.update_timestamp()

    def add_tag(self, key: str, value: Any) -> None:
        """Add a tag to the span."""
        self.tags[key] = value
        self.update_timestamp()

    def add_log(self, message: str, level: str = "info", **fields) -> None:
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **fields
        }
        self.logs.append(log_entry)
        self.update_timestamp()

    def set_error(self, error_message: str, error_type: Optional[str] = None) -> None:
        """Set span error."""
        self.error = True
        self.error_message = error_message
        self.error_type = error_type or "Exception"
        self.status = SpanStatus.ERROR
        self.add_log(error_message, level="error", error_type=error_type)

    def is_root(self) -> bool:
        """Check if span is root span."""
        return self.parent_span_id is None

    def get_depth(self) -> int:
        """Get span depth in trace tree."""
        # Simplified - would need full trace context
        return 0 if self.is_root() else 1


class Trace(BaseModel):
    """Model for distributed trace."""

    trace_id: str = Field(..., description="Unique trace ID")
    spans: List[Span] = Field(default_factory=list, description="Trace spans")

    start_time: Optional[datetime] = Field(None, description="Trace start time")
    end_time: Optional[datetime] = Field(None, description="Trace end time")
    duration_ms: Optional[float] = Field(None, description="Total duration")

    service_names: List[str] = Field(default_factory=list, description="Services involved")
    operation_names: List[str] = Field(default_factory=list, description="Operations performed")

    has_error: bool = Field(False, description="Trace has errors")
    error_count: int = Field(0, ge=0, description="Number of errors")

    def add_span(self, span: Span) -> None:
        """Add span to trace."""
        self.spans.append(span)

        # Update trace metadata
        if span.service_name not in self.service_names:
            self.service_names.append(span.service_name)

        if span.operation_name not in self.operation_names:
            self.operation_names.append(span.operation_name)

        if span.error:
            self.has_error = True
            self.error_count += 1

        # Update times
        if not self.start_time or span.start_time < self.start_time:
            self.start_time = span.start_time

        if span.end_time and (not self.end_time or span.end_time > self.end_time):
            self.end_time = span.end_time

        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        self.update_timestamp()

    def get_root_span(self) -> Optional[Span]:
        """Get root span of trace."""
        for span in self.spans:
            if span.is_root():
                return span
        return None

    def get_critical_path(self) -> List[Span]:
        """Get critical path (longest duration path)."""
        # Simplified - returns longest span
        return sorted(self.spans, key=lambda s: s.duration_ms or 0, reverse=True)[:1]


class PerformanceMetrics(BaseModel):
    """Model for performance metrics."""

    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")

    # Request metrics
    request_count: int = Field(0, ge=0, description="Total requests")
    request_rate: float = Field(0.0, ge=0.0, description="Requests per second")
    error_count: int = Field(0, ge=0, description="Error count")
    error_rate: float = Field(0.0, ge=0.0, le=1.0, description="Error rate")

    # Latency metrics (milliseconds)
    latency_p50: float = Field(0.0, ge=0.0, description="50th percentile latency")
    latency_p95: float = Field(0.0, ge=0.0, description="95th percentile latency")
    latency_p99: float = Field(0.0, ge=0.0, description="99th percentile latency")
    latency_avg: float = Field(0.0, ge=0.0, description="Average latency")

    # Resource metrics
    cpu_usage_percent: float = Field(0.0, ge=0.0, le=100.0, description="CPU usage %")
    memory_usage_mb: float = Field(0.0, ge=0.0, description="Memory usage MB")
    disk_io_mb: float = Field(0.0, ge=0.0, description="Disk I/O MB/s")
    network_io_mb: float = Field(0.0, ge=0.0, description="Network I/O MB/s")

    # Cache metrics
    cache_hit_rate: float = Field(0.0, ge=0.0, le=1.0, description="Cache hit rate")
    cache_size_mb: float = Field(0.0, ge=0.0, description="Cache size MB")

    # Data source metrics
    source_latencies: Dict[str, float] = Field(
        default_factory=dict,
        description="Latency by data source"
    )
    source_errors: Dict[str, int] = Field(
        default_factory=dict,
        description="Errors by data source"
    )

    def calculate_health_score(self) -> float:
        """Calculate overall health score (0-1)."""
        score = 1.0

        # Penalize for high error rate
        score -= self.error_rate * 0.3

        # Penalize for high latency
        if self.latency_p95 > 1000:  # > 1 second
            score -= 0.2
        elif self.latency_p95 > 500:  # > 500ms
            score -= 0.1

        # Penalize for high resource usage
        if self.cpu_usage_percent > 80:
            score -= 0.1
        if self.memory_usage_mb > 1000:  # > 1GB
            score -= 0.1

        # Bonus for good cache hit rate
        score += self.cache_hit_rate * 0.1

        return max(0.0, min(1.0, score))

    def get_sla_status(self) -> Dict[str, bool]:
        """Check SLA compliance."""
        return {
            "availability": self.error_rate < 0.01,  # 99% availability
            "latency": self.latency_p99 < 1000,  # p99 < 1s
            "error_rate": self.error_rate < 0.05,  # < 5% errors
        }


class ObservabilityManager(BaseModel):
    """Model for managing observability data."""

    # Metrics
    metrics: Dict[str, Metric] = Field(default_factory=dict, description="Metrics")

    # Traces
    traces: Dict[str, Trace] = Field(default_factory=dict, description="Active traces")
    completed_traces: List[Trace] = Field(default_factory=list, description="Completed traces")

    # Performance history
    performance_history: List[PerformanceMetrics] = Field(
        default_factory=list,
        description="Performance metrics history"
    )

    # Settings
    retention_days: int = Field(7, gt=0, description="Data retention in days")
    max_traces: int = Field(1000, gt=0, description="Maximum traces to keep")
    sampling_rate: float = Field(1.0, ge=0.0, le=1.0, description="Trace sampling rate")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = Metric(
                name=name,
                metric_type=metric_type,
                tags=tags or {}
            )

        self.metrics[name].record(value)

    def start_span(
        self,
        trace_id: str,
        span_id: str,
        operation_name: str,
        parent_span_id: Optional[str] = None
    ) -> Span:
        """Start a new span."""
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name="daily-minutes"
        )

        # Add to trace
        if trace_id not in self.traces:
            self.traces[trace_id] = Trace(trace_id=trace_id)

        self.traces[trace_id].add_span(span)
        return span

    def finish_span(self, span_id: str, status: SpanStatus = SpanStatus.SUCCESS) -> None:
        """Finish a span."""
        for trace in self.traces.values():
            for span in trace.spans:
                if span.span_id == span_id:
                    span.finish(status)
                    return

    def complete_trace(self, trace_id: str) -> None:
        """Complete a trace."""
        if trace_id in self.traces:
            trace = self.traces[trace_id]
            self.completed_traces.append(trace)

            # Maintain max traces
            if len(self.completed_traces) > self.max_traces:
                self.completed_traces.pop(0)

            del self.traces[trace_id]

    def get_current_performance(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if self.performance_history:
            return self.performance_history[-1]

        return PerformanceMetrics()

    def record_performance(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        self.performance_history.append(metrics)

        # Maintain retention
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        self.performance_history = [
            m for m in self.performance_history
            if m.timestamp > cutoff
        ]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_metrics": len(self.metrics),
            "active_traces": len(self.traces),
            "completed_traces": len(self.completed_traces),
            "performance_samples": len(self.performance_history),
            "current_health": self.get_current_performance().calculate_health_score(),
            "metrics": {
                name: {
                    "type": metric.metric_type.value,
                    "value": metric.value,
                    "count": metric.count,
                    "average": metric.get_average()
                }
                for name, metric in self.metrics.items()
            }
        }

    def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format == "prometheus":
            lines = []
            for metric in self.metrics.values():
                lines.append(metric.to_prometheus_format())
            return "\n".join(lines)
        else:
            # JSON format
            import json
            return json.dumps(self.get_metrics_summary(), indent=2, default=str)