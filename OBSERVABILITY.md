# AI Application Observability and Monitoring

## Overview

Comprehensive observability is critical for managing AI application costs, performance, and quality. This document outlines metrics, logging strategies, monitoring systems, and alerting mechanisms for AI development.

---

## Core Principle

**"You can't optimize what you don't measure."**

Every LLM call must be tracked, logged, and analyzed to identify optimization opportunities.

---

## 1. Key Metrics to Track

### 1.1 Cost Metrics (Critical)

```python
class CostMetrics:
    """Track every dollar spent"""

    def __init__(self):
        self.metrics = {
            # Per-request metrics
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'cost_usd': 0.0,

            # Aggregated metrics
            'total_requests': 0,
            'total_cost_today': 0.0,
            'total_cost_month': 0.0,

            # Cost by dimension
            'cost_by_endpoint': {},
            'cost_by_model': {},
            'cost_by_user': {},
            'cost_by_feature': {},
        }

    def track_request(self, request_data):
        """Log every LLM API call"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request_data['id'],
            'endpoint': request_data['endpoint'],
            'user_id': request_data['user_id'],
            'model': request_data['model'],
            'input_tokens': request_data['input_tokens'],
            'output_tokens': request_data['output_tokens'],
            'total_tokens': request_data['input_tokens'] + request_data['output_tokens'],
            'cost_usd': self.calculate_cost(request_data),
            'latency_ms': request_data['latency_ms'],
            'cache_hit': request_data.get('cache_hit', False),
        }

        # Write to metrics database
        self.write_metric(log_entry)
        return log_entry

    def calculate_cost(self, request_data):
        """Calculate cost based on model pricing"""
        model = request_data['model']
        input_tokens = request_data['input_tokens']
        output_tokens = request_data['output_tokens']

        # Pricing per 1M tokens
        PRICING = {
            'claude-3-haiku': {'input': 0.25, 'output': 1.25},
            'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},
            'claude-3-opus': {'input': 15.00, 'output': 75.00},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-4': {'input': 30.00, 'output': 60.00},
        }

        if model not in PRICING:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * PRICING[model]['input']
        output_cost = (output_tokens / 1_000_000) * PRICING[model]['output']

        return input_cost + output_cost
```

### 1.2 Performance Metrics

```python
class PerformanceMetrics:
    """Track response times and throughput"""

    METRICS = [
        'request_latency_ms',      # Total request time
        'llm_latency_ms',          # LLM API call time
        'cache_latency_ms',        # Cache lookup time
        'preprocessing_ms',        # Before LLM
        'postprocessing_ms',       # After LLM
        'requests_per_second',     # Throughput
        'concurrent_requests',     # Load
    ]

    def track_latency(self, operation: str, start_time: float):
        """Track operation duration"""
        duration_ms = (time.time() - start_time) * 1000

        self.histogram(
            f'{operation}_latency_ms',
            duration_ms,
            tags={
                'operation': operation,
                'endpoint': current_endpoint(),
            }
        )

        return duration_ms
```

### 1.3 Quality Metrics

```python
class QualityMetrics:
    """Track output quality and accuracy"""

    def track_response_quality(self, request, response):
        """Log quality indicators"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'request_id': request['id'],

            # Confidence
            'confidence_score': response.get('confidence', 0.0),

            # Fallback tracking
            'used_fallback': response.get('fallback_used', False),
            'fallback_reason': response.get('fallback_reason'),

            # Validation
            'passed_validation': self.validate_response(response),
            'validation_errors': response.get('validation_errors', []),

            # User feedback
            'user_rating': None,  # Updated later
            'user_feedback': None,
        }

        self.write_metric(metrics)
        return metrics

    def track_user_feedback(self, request_id: str, rating: int, feedback: str = None):
        """Track user satisfaction"""
        self.update_metric(request_id, {
            'user_rating': rating,  # 1-5 stars
            'user_feedback': feedback,
            'feedback_timestamp': datetime.utcnow().isoformat(),
        })
```

### 1.4 Usage Metrics

```python
class UsageMetrics:
    """Track feature usage and patterns"""

    def track_usage(self, event):
        """Log feature usage"""
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': event['user_id'],
            'session_id': event['session_id'],
            'feature': event['feature'],
            'action': event['action'],

            # Context
            'user_tier': event.get('user_tier', 'free'),
            'request_count_today': self.get_user_request_count(event['user_id']),

            # Source tracking
            'handled_by': event.get('handled_by'),  # 'deterministic', 'cache', 'llm'
            'model_used': event.get('model'),
        }

        self.write_metric(metrics)
```

---

## 2. Logging Strategy

### 2.1 Structured Logging

```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    """JSON-formatted logging for AI operations"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self.JSONFormatter())
        self.logger.addHandler(handler)

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
            }

            # Add extra fields
            if hasattr(record, 'extra'):
                log_obj.update(record.extra)

            return json.dumps(log_obj)

    def log_llm_request(self, request_data):
        """Log LLM API request"""
        self.logger.info('LLM Request', extra={
            'event_type': 'llm_request',
            'request_id': request_data['id'],
            'model': request_data['model'],
            'input_tokens': request_data['input_tokens'],
            'prompt_length': len(request_data['prompt']),
            'user_id': request_data['user_id'],
            'endpoint': request_data['endpoint'],
        })

    def log_llm_response(self, request_id, response_data):
        """Log LLM API response"""
        self.logger.info('LLM Response', extra={
            'event_type': 'llm_response',
            'request_id': request_id,
            'output_tokens': response_data['output_tokens'],
            'latency_ms': response_data['latency_ms'],
            'cost_usd': response_data['cost_usd'],
            'cache_hit': response_data.get('cache_hit', False),
        })

    def log_error(self, error, context):
        """Log errors with context"""
        self.logger.error('Error occurred', extra={
            'event_type': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
        })

    def log_cost_alert(self, alert_data):
        """Log cost threshold alerts"""
        self.logger.warning('Cost Alert', extra={
            'event_type': 'cost_alert',
            'alert_type': alert_data['type'],
            'current_cost': alert_data['current_cost'],
            'threshold': alert_data['threshold'],
            'period': alert_data['period'],
        })
```

### 2.2 Log Levels and Events

```python
# INFO: Normal operations
logger.info('Cache hit', extra={'key': cache_key})
logger.info('Deterministic handler', extra={'handler': 'regex'})

# WARNING: Cost alerts, fallbacks
logger.warning('Using fallback model', extra={'reason': 'low_confidence'})
logger.warning('High token usage', extra={'tokens': 5000})

# ERROR: API failures, validation errors
logger.error('LLM API timeout', extra={'model': 'gpt-4', 'timeout': 30})
logger.error('Invalid response format', extra={'response': response})

# CRITICAL: Budget exceeded, system failures
logger.critical('Daily budget exceeded', extra={'cost': 1000, 'limit': 500})
```

---

## 3. Monitoring Dashboards

### 3.1 Real-Time Cost Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│                   Cost Monitoring Dashboard                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Today's Spend: $47.32 / $100.00 (47%)  ████████░░░░░      │
│  This Month:   $856.21 / $2,000 (43%)   ████████░░░░░░░    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Cost by Model (Last 24h)                           │   │
│  │ Haiku:   $12.45  ████████████░░░░░░░░               │   │
│  │ Sonnet:  $28.90  ████████████████████████████       │   │
│  │ Opus:     $5.97  ███████░░░░░░░░░░░░░░░░░░░░░       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Top 5 Expensive Endpoints                          │   │
│  │ /api/summarize     $18.23  (1,234 calls)           │   │
│  │ /api/chat          $15.67  (2,890 calls)           │   │
│  │ /api/analyze       $10.45  (456 calls)             │   │
│  │ /api/translate     $2.97   (134 calls)             │   │
│  │ /api/classify      $0.00   (5,678 calls) [cached]  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Cache Hit Rate: 68% ████████████████████░░░░░░░░░░        │
│  Avg Cost/Request: $0.0082                                 │
│  Requests Today: 5,789                                     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Performance Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│                Performance Monitoring Dashboard             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Latency (p50/p95/p99):  245ms / 1.2s / 3.4s              │
│  Requests/sec:           12.4                               │
│  Error Rate:             0.3%                               │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Latency Distribution (Last Hour)                   │   │
│  │  0-100ms:    █████████████ 45%                     │   │
│  │  100-500ms:  ██████████████████ 35%                │   │
│  │  500ms-2s:   ████████ 15%                          │   │
│  │  2s-5s:      ██ 4%                                 │   │
│  │  >5s:        █ 1%                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Request Routing                                     │   │
│  │ Deterministic: 3,245 (56%) ████████████████         │   │
│  │ Cache Hit:     1,789 (31%) ████████████             │   │
│  │ Haiku:          456 (8%)   ████                     │   │
│  │ Sonnet:         234 (4%)   ██                       │   │
│  │ Opus:            65 (1%)   █                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Quality Dashboard

```
┌─────────────────────────────────────────────────────────────┐
│                   Quality Monitoring Dashboard              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Satisfaction:  4.6 / 5.0 ⭐⭐⭐⭐⭐                    │
│  Validation Pass Rate: 97.2%                               │
│  Fallback Rate: 8.3%                                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Confidence Score Distribution                       │   │
│  │  0.9-1.0:  ████████████████████ 78%                │   │
│  │  0.8-0.9:  ████████ 15%                            │   │
│  │  0.7-0.8:  ██ 5%                                   │   │
│  │  <0.7:     █ 2%                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Recent Issues:                                            │
│  • 3 validation failures (last hour)                       │
│  • 12 fallback triggers (last hour)                        │
│  • 1 user complaint (last day)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Alerting System

### 4.1 Cost Alerts

```python
class CostAlerter:
    """Monitor and alert on cost thresholds"""

    def __init__(self, config):
        self.thresholds = {
            'hourly_limit': config.get('hourly_limit', 10.0),
            'daily_limit': config.get('daily_limit', 100.0),
            'monthly_limit': config.get('monthly_limit', 2000.0),
        }

        self.alert_levels = {
            'warning': 0.8,   # 80% of limit
            'critical': 0.95,  # 95% of limit
            'emergency': 1.0,  # 100% of limit
        }

    def check_thresholds(self):
        """Check if any threshold is breached"""
        current_costs = self.get_current_costs()

        for period, limit in self.thresholds.items():
            current = current_costs[period]
            percentage = current / limit

            # Check alert levels
            if percentage >= self.alert_levels['emergency']:
                self.trigger_emergency_alert(period, current, limit)
            elif percentage >= self.alert_levels['critical']:
                self.trigger_critical_alert(period, current, limit)
            elif percentage >= self.alert_levels['warning']:
                self.trigger_warning_alert(period, current, limit)

    def trigger_emergency_alert(self, period, current, limit):
        """Emergency: Budget exceeded, disable features"""
        alert = {
            'level': 'EMERGENCY',
            'type': 'cost_exceeded',
            'period': period,
            'current_cost': current,
            'limit': limit,
            'percentage': (current / limit) * 100,
            'action': 'FEATURES_DISABLED',
        }

        # Immediate actions
        self.disable_expensive_features()
        self.send_emergency_notification(alert)
        self.log_alert(alert)

    def trigger_critical_alert(self, period, current, limit):
        """Critical: Approaching limit"""
        alert = {
            'level': 'CRITICAL',
            'type': 'cost_approaching_limit',
            'period': period,
            'current_cost': current,
            'limit': limit,
            'percentage': (current / limit) * 100,
            'action': 'RATE_LIMITING_INCREASED',
        }

        # Protective actions
        self.increase_rate_limits()
        self.send_pager_alert(alert)
        self.log_alert(alert)

    def trigger_warning_alert(self, period, current, limit):
        """Warning: 80% threshold reached"""
        alert = {
            'level': 'WARNING',
            'type': 'cost_warning',
            'period': period,
            'current_cost': current,
            'limit': limit,
            'percentage': (current / limit) * 100,
            'action': 'MONITORING_INTENSIFIED',
        }

        self.send_slack_notification(alert)
        self.log_alert(alert)
```

### 4.2 Performance Alerts

```python
class PerformanceAlerter:
    """Alert on performance degradation"""

    def __init__(self):
        self.thresholds = {
            'p95_latency_ms': 2000,    # 2 seconds
            'error_rate': 0.05,        # 5%
            'timeout_rate': 0.02,      # 2%
        }

    def check_performance(self):
        """Monitor performance metrics"""
        metrics = self.get_recent_metrics(window='5m')

        # Check latency
        if metrics['p95_latency_ms'] > self.thresholds['p95_latency_ms']:
            self.alert_high_latency(metrics)

        # Check error rate
        if metrics['error_rate'] > self.thresholds['error_rate']:
            self.alert_high_errors(metrics)

        # Check timeout rate
        if metrics['timeout_rate'] > self.thresholds['timeout_rate']:
            self.alert_high_timeouts(metrics)

    def alert_high_latency(self, metrics):
        """Alert on high latency"""
        self.send_alert({
            'level': 'WARNING',
            'type': 'high_latency',
            'p95_latency': metrics['p95_latency_ms'],
            'threshold': self.thresholds['p95_latency_ms'],
            'recommendation': 'Check LLM provider status, review recent code changes',
        })
```

### 4.3 Quality Alerts

```python
class QualityAlerter:
    """Alert on quality degradation"""

    def __init__(self):
        self.thresholds = {
            'min_confidence': 0.7,
            'max_fallback_rate': 0.15,  # 15%
            'min_satisfaction': 4.0,     # out of 5
        }

    def check_quality(self):
        """Monitor quality metrics"""
        metrics = self.get_recent_quality_metrics()

        # Low confidence
        if metrics['avg_confidence'] < self.thresholds['min_confidence']:
            self.alert_low_confidence(metrics)

        # High fallback rate
        if metrics['fallback_rate'] > self.thresholds['max_fallback_rate']:
            self.alert_high_fallback(metrics)

        # Low user satisfaction
        if metrics['avg_satisfaction'] < self.thresholds['min_satisfaction']:
            self.alert_low_satisfaction(metrics)
```

---

## 5. Implementation Examples

### 5.1 Complete Observability Wrapper

```python
import time
from functools import wraps
from typing import Callable, Any

class ObservabilityWrapper:
    """Wrap LLM calls with complete observability"""

    def __init__(self, metrics_client, logger, alerter):
        self.metrics = metrics_client
        self.logger = logger
        self.alerter = alerter

    def observe(self, operation: str):
        """Decorator to add observability to any function"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                request_id = self.generate_request_id()
                start_time = time.time()

                # Pre-execution logging
                self.logger.log_llm_request({
                    'id': request_id,
                    'operation': operation,
                    'function': func.__name__,
                    'args': str(args)[:100],  # Truncate
                })

                try:
                    # Execute function
                    result = func(*args, **kwargs)

                    # Calculate metrics
                    latency_ms = (time.time() - start_time) * 1000

                    # Track metrics
                    self.metrics.track_request({
                        'request_id': request_id,
                        'operation': operation,
                        'latency_ms': latency_ms,
                        'input_tokens': result.get('input_tokens', 0),
                        'output_tokens': result.get('output_tokens', 0),
                        'cost_usd': result.get('cost_usd', 0),
                        'model': result.get('model'),
                        'cache_hit': result.get('cache_hit', False),
                    })

                    # Post-execution logging
                    self.logger.log_llm_response(request_id, {
                        'latency_ms': latency_ms,
                        'success': True,
                        'output_tokens': result.get('output_tokens', 0),
                        'cost_usd': result.get('cost_usd', 0),
                    })

                    # Check alerts
                    self.alerter.check_thresholds()

                    return result

                except Exception as e:
                    # Error logging
                    self.logger.log_error(e, {
                        'request_id': request_id,
                        'operation': operation,
                        'function': func.__name__,
                    })

                    # Track error metrics
                    self.metrics.track_error({
                        'request_id': request_id,
                        'error_type': type(e).__name__,
                        'operation': operation,
                    })

                    raise

            return wrapper
        return decorator

# Usage
obs = ObservabilityWrapper(metrics_client, logger, alerter)

@obs.observe('text_generation')
def generate_text(prompt: str) -> dict:
    response = llm.call(prompt)
    return {
        'text': response.text,
        'input_tokens': response.input_tokens,
        'output_tokens': response.output_tokens,
        'model': response.model,
        'cost_usd': calculate_cost(response),
    }
```

### 5.2 Metrics Export to Prometheus

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

class PrometheusMetrics:
    """Export metrics to Prometheus"""

    def __init__(self):
        # Counters
        self.request_count = Counter(
            'llm_requests_total',
            'Total LLM requests',
            ['model', 'endpoint', 'status']
        )

        self.token_count = Counter(
            'llm_tokens_total',
            'Total tokens processed',
            ['model', 'token_type']  # input/output
        )

        self.cost_usd = Counter(
            'llm_cost_usd_total',
            'Total cost in USD',
            ['model', 'endpoint']
        )

        # Histograms
        self.latency = Histogram(
            'llm_latency_seconds',
            'Request latency',
            ['model', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )

        # Gauges
        self.cache_hit_rate = Gauge(
            'llm_cache_hit_rate',
            'Cache hit rate percentage'
        )

        self.active_requests = Gauge(
            'llm_active_requests',
            'Currently active requests'
        )

    def track_request(self, data):
        """Track request metrics"""
        # Increment counters
        self.request_count.labels(
            model=data['model'],
            endpoint=data['endpoint'],
            status='success'
        ).inc()

        self.token_count.labels(
            model=data['model'],
            token_type='input'
        ).inc(data['input_tokens'])

        self.token_count.labels(
            model=data['model'],
            token_type='output'
        ).inc(data['output_tokens'])

        self.cost_usd.labels(
            model=data['model'],
            endpoint=data['endpoint']
        ).inc(data['cost_usd'])

        # Record latency
        self.latency.labels(
            model=data['model'],
            endpoint=data['endpoint']
        ).observe(data['latency_ms'] / 1000.0)

# Start Prometheus metrics server
start_http_server(8000)
```

---

## 6. Tools and Platforms

### 6.1 Recommended Stack

```yaml
# Metrics & Monitoring
- Prometheus: Time-series metrics database
- Grafana: Visualization dashboards
- Datadog: All-in-one observability (paid)
- New Relic: APM and monitoring (paid)

# Logging
- ELK Stack: Elasticsearch, Logstash, Kibana
- Loki: Log aggregation (lightweight)
- CloudWatch Logs: AWS native
- Splunk: Enterprise logging (paid)

# Alerting
- PagerDuty: Incident management
- Opsgenie: Alert management
- Slack: Team notifications
- Email: Basic alerts

# Tracing
- Jaeger: Distributed tracing
- Zipkin: Request tracing
- AWS X-Ray: AWS native tracing

# Cost Tracking
- CloudWatch Billing: AWS costs
- Custom dashboards: LLM-specific costs
```

### 6.2 Open Source Tools

```python
# Langfuse - Open source LLM observability
from langfuse import Langfuse

langfuse = Langfuse(
    public_key="your_public_key",
    secret_key="your_secret_key"
)

@langfuse.observe()
def generate_text(prompt):
    return llm.call(prompt)

# Helicone - LLM observability proxy
import openai

openai.api_base = "https://oai.hconeai.com/v1"
openai.api_key = "your_openai_key"
openai.default_headers = {
    "Helicone-Auth": "Bearer your_helicone_key"
}

# LangSmith - LangChain observability
from langsmith import Client

client = Client()
client.create_run(
    name="text_generation",
    inputs={"prompt": prompt},
    run_type="llm"
)
```

---

## 7. Best Practices

### 7.1 Logging Best Practices

✅ **DO:**
- Use structured JSON logging
- Include request IDs for tracing
- Log all LLM calls with tokens and cost
- Sanitize PII before logging
- Set appropriate log levels
- Use sampling for high-volume logs

❌ **DON'T:**
- Log full prompts/responses (use summaries)
- Log sensitive user data
- Use string concatenation for logs
- Ignore errors silently
- Log everything at INFO level

### 7.2 Metrics Best Practices

✅ **DO:**
- Track cost per endpoint
- Monitor cache hit rates
- Measure p95/p99 latencies
- Track model distribution
- Monitor error rates
- Calculate cost per user

❌ **DON'T:**
- Only track averages (use percentiles)
- Ignore outliers
- Track vanity metrics
- Forget to aggregate by dimensions
- Store metrics indefinitely without rollups

### 7.3 Alerting Best Practices

✅ **DO:**
- Set tiered alert levels (warning/critical/emergency)
- Define clear action items for each alert
- Use rate limiting to prevent alert storms
- Test alert configurations regularly
- Document escalation procedures
- Set up alert fatigue prevention

❌ **DON'T:**
- Alert on everything
- Use single threshold for all scenarios
- Ignore time of day patterns
- Alert without context
- Forget to update on-call schedules

---

## 8. Sample Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "LLM Cost & Performance",
    "panels": [
      {
        "title": "Total Cost Today",
        "type": "stat",
        "targets": [{
          "expr": "sum(llm_cost_usd_total{period='today'})"
        }]
      },
      {
        "title": "Requests by Model",
        "type": "graph",
        "targets": [{
          "expr": "rate(llm_requests_total[5m])",
          "legendFormat": "{{model}}"
        }]
      },
      {
        "title": "P95 Latency",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, llm_latency_seconds)",
          "legendFormat": "{{endpoint}}"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "type": "gauge",
        "targets": [{
          "expr": "llm_cache_hit_rate"
        }]
      },
      {
        "title": "Cost by Endpoint",
        "type": "piechart",
        "targets": [{
          "expr": "sum by (endpoint) (llm_cost_usd_total)"
        }]
      }
    ]
  }
}
```

---

## 9. Cost Analysis Queries

### 9.1 Daily Cost Breakdown

```sql
-- Total cost by model (last 30 days)
SELECT
    DATE(timestamp) as date,
    model,
    SUM(cost_usd) as total_cost,
    COUNT(*) as request_count,
    AVG(cost_usd) as avg_cost_per_request
FROM llm_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY date, model
ORDER BY date DESC, total_cost DESC;
```

### 9.2 Most Expensive Endpoints

```sql
-- Top 10 most expensive endpoints
SELECT
    endpoint,
    COUNT(*) as request_count,
    SUM(cost_usd) as total_cost,
    AVG(cost_usd) as avg_cost,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens
FROM llm_metrics
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY endpoint
ORDER BY total_cost DESC
LIMIT 10;
```

### 9.3 Cost Per User

```sql
-- Top spending users
SELECT
    user_id,
    COUNT(*) as request_count,
    SUM(cost_usd) as total_cost,
    AVG(cost_usd) as avg_cost,
    user_tier
FROM llm_metrics
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY user_id, user_tier
ORDER BY total_cost DESC
LIMIT 20;
```

### 9.4 Cache Effectiveness

```sql
-- Cache hit rate by endpoint
SELECT
    endpoint,
    COUNT(*) as total_requests,
    SUM(CASE WHEN cache_hit = true THEN 1 ELSE 0 END) as cache_hits,
    ROUND(100.0 * SUM(CASE WHEN cache_hit = true THEN 1 ELSE 0 END) / COUNT(*), 2) as hit_rate_pct,
    SUM(CASE WHEN cache_hit = true THEN cost_usd ELSE 0 END) as cost_saved
FROM llm_metrics
WHERE timestamp >= NOW() - INTERVAL '7 days'
GROUP BY endpoint
ORDER BY hit_rate_pct DESC;
```

---

## 10. Observability Checklist

### Initial Setup
- [ ] Set up structured logging (JSON format)
- [ ] Configure metrics collection (Prometheus/Datadog)
- [ ] Create cost tracking tables/streams
- [ ] Set up dashboards (Grafana/custom)
- [ ] Configure alerting rules
- [ ] Document alert response procedures

### Metrics to Track
- [ ] Input/output tokens per request
- [ ] Cost per request (USD)
- [ ] Cost by endpoint
- [ ] Cost by model
- [ ] Cost by user
- [ ] Request latency (p50/p95/p99)
- [ ] Cache hit rate
- [ ] Error rate
- [ ] Fallback rate
- [ ] User satisfaction scores

### Alerts to Configure
- [ ] Daily cost exceeds 80% of budget
- [ ] Hourly cost spike (3x average)
- [ ] Error rate > 5%
- [ ] P95 latency > 2 seconds
- [ ] Cache hit rate drops below 50%
- [ ] User satisfaction drops below 4.0

### Regular Reviews
- [ ] Daily: Check cost vs budget
- [ ] Weekly: Review expensive endpoints
- [ ] Monthly: Analyze cost trends
- [ ] Quarterly: Optimize based on insights

---

**Version:** 1.0
**Last Updated:** February 8, 2026
**Status:** Active
