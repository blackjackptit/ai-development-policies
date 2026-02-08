# AI Application Metrics Guide

## Overview

This guide provides a comprehensive catalog of metrics for AI/LLM applications, including definitions, formulas, collection methods, thresholds, and best practices. Use this as a reference for implementing complete observability for your AI systems.

**Contents:**
- Cost metrics
- Performance metrics
- Quality metrics
- Usage metrics
- Infrastructure metrics
- Business metrics
- Security metrics
- Cache metrics

**Related Guides:**
- [Observability Architecture](OBSERVABILITY_ARCHITECTURE.md) - Monitoring infrastructure
- [Observability Guide](OBSERVABILITY.md) - Monitoring strategies
- [Cost Reduction](COST_REDUCTION_RULES.md) - Cost optimization

---

## 1. Cost Metrics

### 1.1 Core Cost Metrics

**Total Cost (USD)**
- **Definition**: Total spend on LLM API calls
- **Formula**: `sum(input_tokens × input_price + output_tokens × output_price)`
- **Unit**: USD
- **Collection**: Counter, increment on each request
- **Thresholds**:
  - Daily budget: Set based on business requirements
  - Alert if > 80% of daily budget
  - Critical alert if > 95% of daily budget

```prometheus
# Prometheus query
sum(llm_cost_usd_total)
```

**Cost Per Request**
- **Definition**: Average cost per API request
- **Formula**: `total_cost / total_requests`
- **Unit**: USD
- **Collection**: Calculated from counters
- **Thresholds**:
  - Target: < $0.01 per request
  - Alert if > $0.05 per request

```prometheus
sum(rate(llm_cost_usd_total[5m])) / sum(rate(llm_requests_total[5m]))
```

**Cost Rate (USD/hour)**
- **Definition**: Current spending rate
- **Formula**: `cost_in_last_hour`
- **Unit**: USD/hour
- **Collection**: Gauge, updated every minute
- **Thresholds**:
  - Warning if > $10/hour
  - Critical if > $50/hour (unless expected)

```prometheus
sum(rate(llm_cost_usd_total[1h])) * 3600
```

**Cost by Model**
- **Definition**: Cost breakdown by model
- **Formula**: `sum(cost) group by model`
- **Unit**: USD
- **Collection**: Counter with model label
- **Use**: Identify most expensive models

```prometheus
sum by (model) (llm_cost_usd_total)
```

**Cost by User**
- **Definition**: Cost per user/API key
- **Formula**: `sum(cost) group by user_id`
- **Unit**: USD
- **Collection**: Counter with user_id label
- **Use**: Track user spending, enforce quotas

```prometheus
sum by (user_id) (increase(llm_cost_usd_total[24h]))
```

**Cost by Provider**
- **Definition**: Cost breakdown by LLM provider
- **Formula**: `sum(cost) group by provider`
- **Unit**: USD
- **Collection**: Counter with provider label
- **Use**: Compare provider costs

```prometheus
sum by (provider) (llm_cost_usd_total)
```

### 1.2 Token Metrics

**Total Tokens**
- **Definition**: Total tokens processed (input + output)
- **Formula**: `sum(input_tokens + output_tokens)`
- **Unit**: Tokens
- **Collection**: Counter
- **Use**: Track usage volume

```prometheus
sum(llm_tokens_total)
```

**Input Tokens**
- **Definition**: Total input tokens sent to LLM
- **Formula**: `sum(input_tokens)`
- **Unit**: Tokens
- **Collection**: Counter with token_type=input label

```prometheus
sum(llm_tokens_total{token_type="input"})
```

**Output Tokens**
- **Definition**: Total output tokens from LLM
- **Formula**: `sum(output_tokens)`
- **Unit**: Tokens
- **Collection**: Counter with token_type=output label

```prometheus
sum(llm_tokens_total{token_type="output"})
```

**Tokens Per Request**
- **Definition**: Average tokens per request
- **Formula**: `total_tokens / total_requests`
- **Unit**: Tokens
- **Collection**: Calculated
- **Thresholds**:
  - Target: < 2000 tokens/request
  - Alert if > 4000 tokens/request

```prometheus
sum(rate(llm_tokens_total[5m])) / sum(rate(llm_requests_total[5m]))
```

**Token Efficiency**
- **Definition**: Cost per million tokens
- **Formula**: `(total_cost / total_tokens) × 1,000,000`
- **Unit**: USD per million tokens
- **Use**: Compare efficiency across models

```python
# Python calculation
cost_per_million_tokens = (total_cost / total_tokens) * 1_000_000
```

### 1.3 Cost Optimization Metrics

**Cache Cost Savings**
- **Definition**: Cost saved by cache hits
- **Formula**: `cache_hits × avg_cost_per_request`
- **Unit**: USD
- **Collection**: Calculated from cache metrics

```prometheus
sum(cache_operations_total{result="hit"}) * avg(llm_cost_usd_total / llm_requests_total)
```

**Deterministic Logic Savings**
- **Definition**: Cost saved by using deterministic logic instead of LLM
- **Formula**: `deterministic_requests × avg_llm_cost`
- **Unit**: USD
- **Collection**: Track deterministic path usage
- **Target**: > 30% of simple queries use deterministic logic

**Cost Per User Tier**
- **Definition**: Average cost per user by subscription tier
- **Formula**: `sum(cost) / count(users) group by tier`
- **Unit**: USD per user
- **Use**: Ensure profitability per tier

```prometheus
sum by (user_tier) (llm_cost_usd_total) / count by (user_tier) (users)
```

---

## 2. Performance Metrics

### 2.1 Latency Metrics

**Request Latency (p50, p95, p99)**
- **Definition**: End-to-end request duration percentiles
- **Formula**: Histogram percentiles
- **Unit**: Seconds
- **Collection**: Histogram with buckets [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
- **Thresholds**:
  - p50: < 500ms (target), alert > 2s
  - p95: < 2s (target), alert > 5s
  - p99: < 5s (target), alert > 10s

```prometheus
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))
```

**Average Latency**
- **Definition**: Mean request duration
- **Formula**: `sum(duration) / count(requests)`
- **Unit**: Seconds
- **Collection**: Summary or calculated from histogram
- **Note**: Less useful than percentiles, but easier to understand

```prometheus
rate(llm_request_duration_seconds_sum[5m]) / rate(llm_request_duration_seconds_count[5m])
```

**Latency by Model**
- **Definition**: Latency breakdown by model
- **Formula**: Percentiles grouped by model
- **Unit**: Seconds
- **Use**: Compare model performance

```prometheus
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket{model="haiku"}[5m]))
```

**LLM API Latency**
- **Definition**: Time spent in LLM API call (excluding app logic)
- **Formula**: Tracked in dedicated span/timer
- **Unit**: Seconds
- **Use**: Separate LLM latency from application latency

**First Token Latency (TTFT)**
- **Definition**: Time to first token in streaming responses
- **Formula**: `time_first_token - request_start_time`
- **Unit**: Seconds
- **Thresholds**: < 1s (good UX for streaming)

**Tokens Per Second**
- **Definition**: Generation speed
- **Formula**: `output_tokens / generation_time`
- **Unit**: Tokens/second
- **Use**: Measure LLM throughput

```python
tokens_per_second = output_tokens / (request_end_time - request_start_time)
```

### 2.2 Throughput Metrics

**Requests Per Second (RPS)**
- **Definition**: Request rate
- **Formula**: `count(requests) / time_window`
- **Unit**: Requests/second
- **Collection**: Counter with rate calculation
- **Use**: Capacity planning

```prometheus
rate(llm_requests_total[5m])
```

**Requests Per Minute (RPM)**
- **Definition**: Requests per minute
- **Formula**: `rate(requests[1m]) × 60`
- **Unit**: Requests/minute
- **Use**: Easier to understand than RPS for lower volumes

```prometheus
rate(llm_requests_total[1m]) * 60
```

**Concurrent Requests**
- **Definition**: Number of in-flight requests
- **Formula**: Active request gauge
- **Unit**: Count
- **Collection**: Gauge, increment on start, decrement on end
- **Thresholds**:
  - Warning if > 80% of max capacity
  - Critical if > 95% of max capacity

```prometheus
sum(llm_active_requests)
```

**Request Queue Length**
- **Definition**: Requests waiting in queue
- **Formula**: Count of queued requests
- **Unit**: Count
- **Collection**: Gauge
- **Thresholds**:
  - Target: < 10
  - Warning if > 50
  - Critical if > 100

```prometheus
request_queue_length
```

### 2.3 Reliability Metrics

**Error Rate**
- **Definition**: Percentage of failed requests
- **Formula**: `errors / total_requests × 100`
- **Unit**: Percentage
- **Collection**: Counters for success and errors
- **Thresholds**:
  - Target: < 0.1%
  - Warning if > 1%
  - Critical if > 5%

```prometheus
rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) * 100
```

**Error by Type**
- **Definition**: Errors grouped by error type
- **Formula**: `sum(errors) group by error_type`
- **Unit**: Count
- **Use**: Identify common failure modes

```prometheus
sum by (error_type) (rate(llm_errors_total[5m]))
```

**Success Rate**
- **Definition**: Percentage of successful requests
- **Formula**: `(total - errors) / total × 100`
- **Unit**: Percentage
- **Thresholds**: > 99% (target), > 95% (minimum)

```prometheus
(rate(llm_requests_total[5m]) - rate(llm_errors_total[5m])) / rate(llm_requests_total[5m]) * 100
```

**Retry Rate**
- **Definition**: Percentage of requests that required retries
- **Formula**: `retries / total_requests × 100`
- **Unit**: Percentage
- **Collection**: Counter for retries
- **Target**: < 5%

**Rate Limit Hits**
- **Definition**: Number of rate limit violations
- **Formula**: `count(rate_limit_errors)`
- **Unit**: Count
- **Collection**: Counter
- **Use**: Identify if rate limits are too restrictive

```prometheus
sum(rate(rate_limit_hits_total[5m]))
```

**Availability**
- **Definition**: Service uptime percentage
- **Formula**: `(total_time - downtime) / total_time × 100`
- **Unit**: Percentage
- **Target**: > 99.9% (three nines)

```prometheus
avg(up{job="ai-app"}) * 100
```

---

## 3. Quality Metrics

### 3.1 Response Quality

**Response Length (tokens)**
- **Definition**: Length of generated responses
- **Formula**: Histogram of output_tokens
- **Unit**: Tokens
- **Collection**: Histogram
- **Use**: Ensure responses are appropriate length

```prometheus
histogram_quantile(0.50, rate(response_length_tokens_bucket[5m]))
```

**Response Length (characters)**
- **Definition**: Character count of responses
- **Formula**: `len(response_text)`
- **Unit**: Characters
- **Use**: User-facing metric

**Truncation Rate**
- **Definition**: Percentage of responses that hit max_tokens limit
- **Formula**: `truncated_responses / total_responses × 100`
- **Unit**: Percentage
- **Collection**: Track finish_reason=length
- **Target**: < 5% (indicates good max_tokens setting)

```python
truncation_rate = (responses_with_finish_reason_length / total_responses) * 100
```

**Empty Response Rate**
- **Definition**: Percentage of empty/very short responses
- **Formula**: `responses_with_length_<_10 / total × 100`
- **Unit**: Percentage
- **Target**: < 1%

**Response Quality Score**
- **Definition**: Quality score from evaluation (0-1)
- **Formula**: Varies by evaluation method
- **Unit**: Score (0-1)
- **Collection**: If using automated evaluation
- **Target**: > 0.8

### 3.2 User Feedback Metrics

**User Satisfaction Score**
- **Definition**: User rating of responses (thumbs up/down, 1-5 stars)
- **Formula**: `sum(ratings) / count(ratings)`
- **Unit**: Score or percentage
- **Collection**: User feedback events
- **Target**: > 80% positive feedback

**Feedback Rate**
- **Definition**: Percentage of requests with user feedback
- **Formula**: `requests_with_feedback / total_requests × 100`
- **Unit**: Percentage
- **Use**: Gauge user engagement with feedback system

**Negative Feedback Rate**
- **Definition**: Percentage of negative feedback
- **Formula**: `negative_feedback / total_feedback × 100`
- **Unit**: Percentage
- **Target**: < 20%

### 3.3 Content Safety Metrics

**PII Detection Rate**
- **Definition**: Percentage of inputs with detected PII
- **Formula**: `inputs_with_pii / total_inputs × 100`
- **Unit**: Percentage
- **Collection**: Track PII detector results
- **Use**: Monitor PII exposure risk

```prometheus
rate(pii_detections_total[5m]) / rate(llm_requests_total[5m]) * 100
```

**Toxic Content Rate**
- **Definition**: Percentage of responses flagged as toxic
- **Formula**: `toxic_responses / total_responses × 100`
- **Unit**: Percentage
- **Target**: < 0.1%

**Prompt Injection Attempts**
- **Definition**: Number of detected prompt injection attempts
- **Formula**: `count(injection_attempts)`
- **Unit**: Count
- **Collection**: Security event counter
- **Use**: Security monitoring

```prometheus
sum(rate(security_events_total{event_type="prompt_injection"}[5m]))
```

---

## 4. Usage Metrics

### 4.1 User Metrics

**Active Users (DAU/MAU)**
- **Definition**: Unique users making requests
- **Formula**: `count(distinct user_id)` in time period
- **Unit**: Count
- **Collection**: Track unique user IDs
- **Use**: Measure product adoption

**Requests Per User**
- **Definition**: Average requests per user
- **Formula**: `total_requests / active_users`
- **Unit**: Requests/user
- **Use**: Understand usage patterns

```prometheus
sum(rate(llm_requests_total[24h])) / count(distinct user_id)
```

**Cost Per User**
- **Definition**: Average cost per user
- **Formula**: `total_cost / active_users`
- **Unit**: USD/user
- **Use**: Unit economics

```prometheus
sum(increase(llm_cost_usd_total[24h])) / count(distinct user_id)
```

**User Retention**
- **Definition**: Percentage of users who return
- **Formula**: `returning_users / total_users × 100`
- **Unit**: Percentage
- **Collection**: Track user activity over time

**Power Users**
- **Definition**: Users exceeding certain usage threshold
- **Formula**: `count(users where requests > threshold)`
- **Unit**: Count
- **Use**: Identify high-value users

### 4.2 Feature Usage

**Model Usage Distribution**
- **Definition**: Request distribution across models
- **Formula**: `count(requests) group by model`
- **Unit**: Percentage
- **Use**: Understand model preferences

```prometheus
sum by (model) (rate(llm_requests_total[24h]))
```

**Feature Adoption Rate**
- **Definition**: Percentage of users using a feature
- **Formula**: `users_with_feature / total_users × 100`
- **Unit**: Percentage
- **Collection**: Track feature usage events

**API Endpoint Usage**
- **Definition**: Request distribution across endpoints
- **Formula**: `count(requests) group by endpoint`
- **Unit**: Count
- **Use**: Identify popular endpoints

```prometheus
sum by (endpoint) (rate(http_requests_total[5m]))
```

### 4.3 Temporal Metrics

**Requests by Hour of Day**
- **Definition**: Request distribution by hour
- **Formula**: `count(requests) group by hour`
- **Unit**: Count
- **Use**: Capacity planning, identify peak hours

**Requests by Day of Week**
- **Definition**: Request distribution by weekday
- **Formula**: `count(requests) group by day_of_week`
- **Unit**: Count
- **Use**: Understand weekly patterns

**Growth Rate**
- **Definition**: Request growth over time
- **Formula**: `(current_period - previous_period) / previous_period × 100`
- **Unit**: Percentage
- **Use**: Track product growth

---

## 5. Cache Metrics

### 5.1 Cache Performance

**Cache Hit Rate**
- **Definition**: Percentage of requests served from cache
- **Formula**: `cache_hits / (cache_hits + cache_misses) × 100`
- **Unit**: Percentage
- **Collection**: Counters for hits and misses
- **Thresholds**:
  - Target: > 60%
  - Warning if < 40%
  - Good: > 80%

```prometheus
sum(rate(cache_operations_total{result="hit"}[5m])) /
(sum(rate(cache_operations_total{result="hit"}[5m])) +
 sum(rate(cache_operations_total{result="miss"}[5m]))) * 100
```

**Cache Miss Rate**
- **Definition**: Percentage of cache misses
- **Formula**: `100 - cache_hit_rate`
- **Unit**: Percentage

**Cache Latency**
- **Definition**: Time to retrieve from cache
- **Formula**: Histogram of cache operation duration
- **Unit**: Milliseconds
- **Target**: < 10ms (Redis/Memcached)

```prometheus
histogram_quantile(0.95, rate(cache_operation_duration_seconds_bucket[5m]))
```

### 5.2 Cache Efficiency

**Cache Memory Usage**
- **Definition**: Memory used by cache
- **Formula**: Gauge from cache metrics
- **Unit**: Bytes or GB
- **Thresholds**: Alert if > 80% of max memory

**Cache Eviction Rate**
- **Definition**: Number of items evicted from cache
- **Formula**: `count(evictions) / time`
- **Unit**: Evictions/second
- **Use**: Indicates if cache size is sufficient

**Cache Size**
- **Definition**: Number of items in cache
- **Formula**: Gauge of cached items
- **Unit**: Count
- **Use**: Monitor cache growth

**Cache Cost Savings**
- **Definition**: Cost saved by cache (avoid LLM calls)
- **Formula**: `cache_hits × avg_cost_per_llm_request`
- **Unit**: USD
- **Use**: ROI of caching

```python
cache_savings_usd = cache_hits * average_llm_cost_per_request
```

---

## 6. Infrastructure Metrics

### 6.1 Resource Utilization

**CPU Usage**
- **Definition**: CPU utilization percentage
- **Formula**: System metric
- **Unit**: Percentage
- **Thresholds**:
  - Target: 50-70% (room for spikes)
  - Warning if > 80%
  - Critical if > 90%

```prometheus
avg(rate(process_cpu_seconds_total[5m])) * 100
```

**Memory Usage**
- **Definition**: Memory utilization percentage
- **Formula**: `used_memory / total_memory × 100`
- **Unit**: Percentage
- **Thresholds**:
  - Warning if > 80%
  - Critical if > 90%

```prometheus
process_resident_memory_bytes / node_memory_total_bytes * 100
```

**Disk Usage**
- **Definition**: Disk space utilization
- **Formula**: `used_disk / total_disk × 100`
- **Unit**: Percentage
- **Thresholds**:
  - Warning if > 80%
  - Critical if > 90%

**Network I/O**
- **Definition**: Network bandwidth usage
- **Formula**: Bytes sent/received per second
- **Unit**: Bytes/second
- **Use**: Identify network bottlenecks

### 6.2 Database Metrics

**Database Connections**
- **Definition**: Number of active DB connections
- **Formula**: Gauge of connection pool
- **Unit**: Count
- **Thresholds**: Alert if > 80% of max connections

```prometheus
pg_stat_activity_count
```

**Query Duration**
- **Definition**: Database query latency percentiles
- **Formula**: Histogram of query times
- **Unit**: Seconds
- **Target**: p95 < 100ms

**Slow Queries**
- **Definition**: Queries exceeding threshold
- **Formula**: `count(queries where duration > 1s)`
- **Unit**: Count/minute
- **Target**: < 1 slow query per minute

**Connection Pool Exhaustion**
- **Definition**: Requests waiting for DB connection
- **Formula**: Gauge of waiting requests
- **Unit**: Count
- **Target**: 0

### 6.3 External Dependencies

**Provider Availability**
- **Definition**: Uptime of LLM providers
- **Formula**: `successful_requests / total_requests × 100`
- **Unit**: Percentage per provider
- **Use**: Track provider reliability

```prometheus
(rate(llm_requests_total{status="success",provider="anthropic"}[5m]) /
 rate(llm_requests_total{provider="anthropic"}[5m])) * 100
```

**Provider Latency**
- **Definition**: Latency by provider
- **Formula**: Percentiles grouped by provider
- **Unit**: Seconds
- **Use**: Compare provider performance

```prometheus
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket{provider="anthropic"}[5m]))
```

**Fallback Usage Rate**
- **Definition**: Percentage of requests using fallback provider
- **Formula**: `fallback_requests / total_requests × 100`
- **Unit**: Percentage
- **Use**: Indicates primary provider issues

---

## 7. Business Metrics

### 7.1 Revenue Metrics

**Revenue Per Request**
- **Definition**: Average revenue per API request
- **Formula**: `total_revenue / total_requests`
- **Unit**: USD
- **Use**: Unit economics

**Profit Per Request**
- **Definition**: Revenue minus cost per request
- **Formula**: `revenue_per_request - cost_per_request`
- **Unit**: USD
- **Target**: Positive (profitable)

**Gross Margin**
- **Definition**: Profit as percentage of revenue
- **Formula**: `(revenue - costs) / revenue × 100`
- **Unit**: Percentage
- **Target**: > 70% for SaaS

**Customer Acquisition Cost (CAC)**
- **Definition**: Cost to acquire a customer
- **Formula**: `total_acquisition_costs / new_customers`
- **Unit**: USD
- **Use**: Marketing efficiency

**Customer Lifetime Value (LTV)**
- **Definition**: Total revenue from a customer
- **Formula**: `avg_revenue_per_user × avg_customer_lifetime`
- **Unit**: USD
- **Target**: LTV:CAC ratio > 3:1

### 7.2 Conversion Metrics

**Free to Paid Conversion Rate**
- **Definition**: Percentage of free users converting to paid
- **Formula**: `paid_conversions / free_users × 100`
- **Unit**: Percentage
- **Target**: Varies by product (typically 2-5%)

**Trial to Paid Conversion Rate**
- **Definition**: Percentage of trial users converting
- **Formula**: `paid_after_trial / trial_users × 100`
- **Unit**: Percentage
- **Target**: > 25%

**Upgrade Rate**
- **Definition**: Users upgrading to higher tiers
- **Formula**: `upgrades / total_users × 100`
- **Unit**: Percentage
- **Use**: Measure pricing strategy effectiveness

### 7.3 Churn Metrics

**Churn Rate**
- **Definition**: Percentage of customers leaving
- **Formula**: `churned_customers / total_customers × 100`
- **Unit**: Percentage (monthly or annually)
- **Target**: < 5% monthly for SaaS

**Revenue Churn**
- **Definition**: MRR lost from churned customers
- **Formula**: `mrr_lost / total_mrr × 100`
- **Unit**: Percentage
- **Target**: < churn rate (indicates expansion revenue)

---

## 8. Security Metrics

### 8.1 Threat Metrics

**Authentication Failures**
- **Definition**: Failed login/API key attempts
- **Formula**: `count(auth_failures)`
- **Unit**: Count
- **Collection**: Counter
- **Thresholds**: Alert if > 10/minute from single IP

```prometheus
rate(auth_failures_total[5m])
```

**Suspicious Activity Rate**
- **Definition**: Percentage of suspicious requests
- **Formula**: `suspicious_requests / total_requests × 100`
- **Unit**: Percentage
- **Collection**: Security event tracking
- **Target**: < 0.1%

**Blocked Requests**
- **Definition**: Requests blocked by security rules
- **Formula**: `count(blocked_requests)`
- **Unit**: Count
- **Use**: Measure security rule effectiveness

**Attack Patterns**
- **Definition**: Known attack signatures detected
- **Formula**: `count(attacks) group by pattern`
- **Unit**: Count
- **Use**: Identify common attack vectors

### 8.2 Compliance Metrics

**PII Exposure Events**
- **Definition**: Instances of PII in logs/responses
- **Formula**: `count(pii_exposure_events)`
- **Unit**: Count
- **Target**: 0

**Data Breach Response Time**
- **Definition**: Time to respond to security incident
- **Formula**: `response_start_time - incident_detection_time`
- **Unit**: Minutes
- **Target**: < 15 minutes (GDPR requirement: 72 hours to notify)

**Access Control Violations**
- **Definition**: Unauthorized access attempts
- **Formula**: `count(access_denied)`
- **Unit**: Count
- **Target**: Low, investigate all instances

---

## 9. Metric Collection Best Practices

### 9.1 Collection Guidelines

**1. Use Appropriate Metric Types:**
- **Counter**: Monotonically increasing (requests, errors, cost)
- **Gauge**: Current value (queue length, active requests, memory)
- **Histogram**: Distribution (latency, response length)
- **Summary**: Statistical summaries (percentiles)

**2. Label Cardinality:**
- Keep label combinations < 10,000
- Avoid high-cardinality labels (user_id, request_id)
- Use aggregation for high-cardinality data

**3. Sampling:**
- Sample traces (1-10% of requests)
- Don't sample critical business metrics
- Use reservoir sampling for fair representation

**4. Aggregation:**
- Pre-aggregate high-frequency metrics
- Use time windows appropriate for use case
- Balance granularity vs storage cost

**5. Retention:**
- Raw metrics: 7-30 days
- Aggregated metrics: 1-2 years
- Long-term trends: Downsample to hourly/daily

### 9.2 Dashboard Organization

**Executive Dashboard:**
- Daily cost
- Request volume
- Error rate
- Revenue (if applicable)

**Operations Dashboard:**
- Latency percentiles (p50, p95, p99)
- Error rate by type
- Cache hit rate
- Active requests

**Cost Management Dashboard:**
- Cost by model
- Cost by user/tier
- Cost trend over time
- Budget vs actual

**Security Dashboard:**
- PII detections
- Prompt injection attempts
- Authentication failures
- Rate limit hits

### 9.3 Alerting Strategy

**Alert Priorities:**

**Critical (Page immediately):**
- Service down (availability < 99%)
- Error rate > 10%
- Cost spike > 2x normal
- Security breach detected

**Warning (Notify during business hours):**
- Error rate > 5%
- Latency p95 > 5s
- Cache hit rate < 40%
- Daily cost > 80% of budget

**Info (Log only):**
- Cost > 50% of daily budget
- Latency p95 > 2s
- Cache hit rate < 60%

**Alert Fatigue Prevention:**
- Set appropriate thresholds
- Use alert grouping
- Implement alert suppression during maintenance
- Regular alert review and tuning

---

## 10. Metric Formulas Reference

### Quick Reference Table

| Metric | Formula | Target |
|--------|---------|--------|
| Cost per request | `total_cost / total_requests` | < $0.01 |
| Error rate | `errors / total × 100` | < 1% |
| Cache hit rate | `hits / (hits + misses) × 100` | > 60% |
| p95 latency | `95th percentile(durations)` | < 2s |
| Tokens per request | `total_tokens / requests` | < 2000 |
| Cost per million tokens | `(cost / tokens) × 1M` | Model-dependent |
| Success rate | `(total - errors) / total × 100` | > 99% |
| Throughput | `requests / time` | Capacity-dependent |
| Availability | `uptime / total_time × 100` | > 99.9% |
| Profit margin | `(revenue - cost) / revenue × 100` | > 70% |

### PromQL Query Examples

```prometheus
# Cost rate (USD/hour)
sum(rate(llm_cost_usd_total[1h])) * 3600

# Error rate (%)
rate(llm_errors_total[5m]) / rate(llm_requests_total[5m]) * 100

# p95 latency
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))

# Cache hit rate (%)
sum(rate(cache_operations_total{result="hit"}[5m])) /
sum(rate(cache_operations_total[5m])) * 100

# Requests per minute
rate(llm_requests_total[1m]) * 60

# Cost by model
sum by (model) (increase(llm_cost_usd_total[24h]))

# Average tokens per request
sum(rate(llm_tokens_total[5m])) / sum(rate(llm_requests_total[5m]))

# Top 10 spenders
topk(10, sum by (user_id) (increase(llm_cost_usd_total[24h])))

# Daily cost trend
sum(increase(llm_cost_usd_total[24h]))

# Provider availability
(rate(llm_requests_total{status="success"}[5m]) /
 rate(llm_requests_total[5m])) * 100
```

---

## 11. Metrics Implementation Checklist

**Essential Metrics (Must Have):**
- [ ] Total cost (USD)
- [ ] Cost per request
- [ ] Request count
- [ ] Error rate
- [ ] Latency (p50, p95, p99)
- [ ] Input/output tokens
- [ ] Cache hit rate

**Important Metrics (Should Have):**
- [ ] Cost by model
- [ ] Cost by user
- [ ] Cost rate (USD/hour)
- [ ] Throughput (RPS)
- [ ] Error by type
- [ ] Cache latency
- [ ] Active requests

**Nice to Have Metrics:**
- [ ] Response quality score
- [ ] User satisfaction
- [ ] PII detection rate
- [ ] Tokens per second
- [ ] Model efficiency (cost per token)
- [ ] Feature usage
- [ ] Business metrics (revenue, profit)

**Infrastructure Metrics:**
- [ ] CPU usage
- [ ] Memory usage
- [ ] Database connections
- [ ] Disk usage
- [ ] Network I/O

**Security Metrics:**
- [ ] Authentication failures
- [ ] Prompt injection attempts
- [ ] Rate limit hits
- [ ] Access control violations

---

## 12. Summary

This guide provides a complete catalog of metrics for AI/LLM applications:

**Metric Categories:**
1. **Cost** - Track spending, optimize budget
2. **Performance** - Ensure responsiveness
3. **Quality** - Maintain high-quality outputs
4. **Usage** - Understand user behavior
5. **Cache** - Maximize efficiency
6. **Infrastructure** - Monitor system health
7. **Business** - Track profitability
8. **Security** - Detect threats

**Key Principles:**
- 📊 **Measure what matters** - Focus on actionable metrics
- 🎯 **Set clear thresholds** - Know when to alert
- 💰 **Track costs religiously** - Cost is critical for AI apps
- ⚡ **Monitor performance** - Latency impacts UX
- 🔒 **Security always** - Track threats and anomalies

**Related Documentation:**
- [Observability Architecture](OBSERVABILITY_ARCHITECTURE.md) - Infrastructure
- [Observability Guide](OBSERVABILITY.md) - Strategies
- [Cost Reduction](COST_REDUCTION_RULES.md) - Optimization

---

**Version:** 1.0
**Last Updated:** February 6, 2026
**Status:** Active
