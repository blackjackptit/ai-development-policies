# AI Development Quick Reference Guide

**A consolidated, actionable reference for cost-efficient, secure, and compliant AI application development.**

This guide consolidates the most critical information from 19,812 lines of documentation across 15 comprehensive guides into a single quick reference.

📖 **[Complete Index](INDEX.md)** | 🏗️ **[Architecture](ARCHITECTURE.md)** | 💰 **[Cost](COST_REDUCTION_RULES.md)** | 🔒 **[Security](SECURITY.md)** | ⚖️ **[Compliance](COMPLIANCE.md)**

---

## Table of Contents

1. [The Golden Rule](#the-golden-rule)
2. [Cost-Aware Pipeline](#cost-aware-pipeline)
3. [Critical Checklists](#critical-checklists)
4. [Code Snippets](#code-snippets)
5. [Configuration Examples](#configuration-examples)
6. [Metrics & Thresholds](#metrics--thresholds)
7. [Quick Decisions](#quick-decisions)
8. [Emergency Procedures](#emergency-procedures)

---

## The Golden Rule

**"LLMs are expensive last-resort tools, not first-choice solutions."**

Before using an LLM, ask: "Can I solve this with code, libraries, or rules?"

### Decision Flow
```
┌─────────────┐
│   Request   │
└──────┬──────┘
       ↓
┌──────────────────────┐
│ Can code solve this? │──YES──→ Use deterministic logic (FREE)
└──────┬───────────────┘
       ↓ NO
┌──────────────────────┐
│ Can rules solve it?  │──YES──→ Use rule engine (FREE)
└──────┬───────────────┘
       ↓ NO
┌──────────────────────┐
│ Is it cached?        │──YES──→ Return cached result (CHEAP)
└──────┬───────────────┘
       ↓ NO
┌──────────────────────┐
│ Use smallest model   │──→ Haiku/GPT-3.5 ($)
│ that works           │──→ Sonnet/GPT-4 Turbo ($$)
└──────────────────────┘    └→ Opus/GPT-4 ($$$)
```

**Source:** [Architecture - Decision Matrix](ARCHITECTURE.md#4-decision-matrix-when-to-use-llm)

---

## Cost-Aware Pipeline

```
Request → Validation → Rules → Cache → Cheap LLM → Expensive LLM
  FREE      FREE        FREE    CHEAP      $$           $$$$$
   0ms       <1ms       <5ms     <10ms     500ms+       2000ms+
```

### When to Use Each Layer

| Layer | Use When | Cost | Latency | Coverage |
|-------|----------|------|---------|----------|
| **Validation** | Always | $0 | <1ms | 100% |
| **Rules** | Extractable patterns | $0 | <5ms | 30-40% |
| **Cache** | Repeated requests | $0.0001 | <10ms | 40-60% |
| **Haiku/GPT-3.5** | Simple tasks | $0.001 | 500ms | 80% |
| **Sonnet/GPT-4 Turbo** | Medium complexity | $0.01 | 1000ms | 15% |
| **Opus/GPT-4** | Complex reasoning | $0.05 | 2000ms | 5% |

**Source:** [Cost-Efficient Architecture](COST_EFFICIENT_ARCHITECTURE.md)

---

## Critical Checklists

### ⚡ Pre-Launch Checklist (30 items)

**Cost Optimization (6 items):**
- [ ] All API calls have `max_tokens` limits set
- [ ] Response caching implemented (Redis, 24h TTL)
- [ ] Using Haiku/GPT-3.5 for 80%+ of requests
- [ ] Deterministic logic used for extractable patterns
- [ ] Conversation history limited to 10-20 messages
- [ ] Rate limiting per user/tier implemented

**Security (8 items):**
- [ ] Input validation on all user inputs (length, format)
- [ ] Prompt injection detection active (9+ patterns)
- [ ] PII detection and redaction configured
- [ ] HTTPS/TLS enforced on all connections
- [ ] API keys in secret manager (not in code)
- [ ] Rate limiting: 10 req/min (free), 100 (pro)
- [ ] Authentication implemented (JWT/API keys)
- [ ] Output sanitization for sensitive data

**Observability (6 items):**
- [ ] Structured logging (JSON format) with request IDs
- [ ] Token usage tracked per request
- [ ] Cost tracked per endpoint/model/user
- [ ] Budget alerts at 80%, 95%, 100%
- [ ] Error rate monitoring (<1% target)
- [ ] Latency tracking (p50/p95/p99)

**Compliance (6 items):**
- [ ] Privacy policy published and accessible
- [ ] Cookie consent banner implemented
- [ ] Data retention policy documented (90 days)
- [ ] User data deletion mechanism (GDPR Article 17)
- [ ] Consent management system active
- [ ] Audit logging enabled (7-year retention)

**Testing (4 items):**
- [ ] Unit tests with mocked LLM (>80% coverage)
- [ ] Integration tests with real API
- [ ] CI/CD pipeline runs tests on every PR
- [ ] Load testing completed (100 concurrent users)

**Sources:**
- [Cost Reduction - Quick Wins](COST_REDUCTION_RULES.md#12-quick-wins-checklist)
- [Security Checklist](SECURITY.md#11-security-checklist)
- [Observability Checklist](OBSERVABILITY.md#10-observability-checklist)
- [Compliance Checklist](COMPLIANCE.md#10-compliance-checklist)

### 🚨 Daily Monitoring Checklist (10 items)

- [ ] Cost today vs. budget (< 80%)
- [ ] Error rate (target: <1%, alert: >5%)
- [ ] p95 latency (target: <2s, alert: >5s)
- [ ] Cache hit rate (target: >60%, alert: <40%)
- [ ] Top 5 most expensive endpoints reviewed
- [ ] Failed requests investigated
- [ ] Budget forecast for month
- [ ] Security alerts reviewed (prompt injection, DDoS)
- [ ] Data subject rights requests processed (<30 days)
- [ ] Audit log integrity verified

**Sources:**
- [Observability - Daily Operations](OBSERVABILITY.md#4-alerting-system)
- [Security - Monitoring](SECURITY.md#10-monitoring-and-logging)
- [Compliance - Operations](COMPLIANCE.md#7-operational-procedures)

---

## Code Snippets

### 1. Complete LLM Client with Cost Optimization

```python
import anthropic
import redis
import hashlib
import json
from typing import Optional, Dict
from dataclasses import dataclass
import time

@dataclass
class LLMConfig:
    model: str = "claude-3-haiku-20240307"
    max_tokens: int = 1024
    temperature: float = 0.7
    cache_ttl: int = 86400  # 24 hours

class CostEfficientLLMClient:
    def __init__(self, api_key: str, redis_url: str = "redis://localhost"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.cache = redis.from_url(redis_url)
        self.config = LLMConfig()

    def _cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        content = f"{model}:{prompt}"
        return f"llm_cache:{hashlib.sha256(content.encode()).hexdigest()}"

    def _get_cached(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response."""
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        return None

    def _set_cache(self, cache_key: str, response: str, ttl: int):
        """Cache response with TTL."""
        self.cache.setex(cache_key, ttl, json.dumps(response))

    def _log_usage(self, model: str, input_tokens: int, output_tokens: int,
                   latency_ms: float, cost_usd: float, cache_hit: bool):
        """Log token usage and cost."""
        log_entry = {
            "timestamp": time.time(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "cache_hit": cache_hit
        }
        print(json.dumps(log_entry))  # Replace with proper logging

    def generate(self, prompt: str, use_cache: bool = True) -> Dict:
        """Generate response with caching and cost tracking."""
        start_time = time.time()

        # Check cache first
        cache_key = self._cache_key(prompt, self.config.model)
        if use_cache:
            cached_response = self._get_cached(cache_key)
            if cached_response:
                latency_ms = (time.time() - start_time) * 1000
                self._log_usage(
                    self.config.model, 0, 0, latency_ms, 0.0, cache_hit=True
                )
                return {
                    "content": cached_response,
                    "cache_hit": True,
                    "latency_ms": latency_ms
                }

        # Call LLM
        try:
            message = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens

            # Calculate cost (prices per 1M tokens)
            pricing = {
                "claude-3-haiku-20240307": (0.25, 1.25),
                "claude-3-sonnet-20240229": (3.0, 15.0),
                "claude-3-opus-20240229": (15.0, 75.0),
            }
            input_price, output_price = pricing.get(
                self.config.model, (3.0, 15.0)
            )
            cost_usd = (
                (input_tokens * input_price / 1_000_000) +
                (output_tokens * output_price / 1_000_000)
            )

            latency_ms = (time.time() - start_time) * 1000

            # Cache response
            if use_cache:
                self._set_cache(cache_key, response_text, self.config.cache_ttl)

            # Log usage
            self._log_usage(
                self.config.model, input_tokens, output_tokens,
                latency_ms, cost_usd, cache_hit=False
            )

            return {
                "content": response_text,
                "cache_hit": False,
                "latency_ms": latency_ms,
                "cost_usd": cost_usd,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens
                }
            }

        except Exception as e:
            print(f"LLM error: {e}")
            raise

# Usage
client = CostEfficientLLMClient(api_key="your-api-key")
result = client.generate("What is the capital of France?")
print(f"Response: {result['content']}")
print(f"Cost: ${result.get('cost_usd', 0):.4f}")
print(f"Cache hit: {result['cache_hit']}")
```

**Source:** [Integration - Direct API](INTEGRATION.md#11-direct-api-integration)

### 2. Input Validation Pipeline

```python
import re
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    sanitized_input: Optional[str] = None

class InputValidator:
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"system\s*[:：]\s*you\s+are",
        r"new\s+instructions\s*[:：]",
        r"<\s*/?system\s*>",
        r"\[INST\]|\[/INST\]",
        r"###\s*Instruction\s*[:：]",
        r"developer\s+mode",
        r"god\s+mode",
        r"sudo\s+mode"
    ]

    # PII patterns
    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
    }

    def __init__(self, max_length: int = 10000):
        self.max_length = max_length

    def validate(self, user_input: str, redact_pii: bool = True) -> ValidationResult:
        """Validate and sanitize user input."""
        errors = []
        sanitized = user_input

        # 1. Length check
        if len(user_input) > self.max_length:
            errors.append(f"Input exceeds maximum length of {self.max_length}")
            return ValidationResult(is_valid=False, errors=errors)

        # 2. Empty check
        if not user_input.strip():
            errors.append("Input cannot be empty")
            return ValidationResult(is_valid=False, errors=errors)

        # 3. Prompt injection detection
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, user_input, re.IGNORECASE):
                errors.append(f"Potential prompt injection detected: {pattern}")

        # 4. PII detection and redaction
        if redact_pii:
            pii_found = []
            for pii_type, pattern in self.PII_PATTERNS.items():
                matches = re.findall(pattern, sanitized)
                if matches:
                    pii_found.append(pii_type)
                    sanitized = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", sanitized)

            if pii_found:
                errors.append(f"PII detected and redacted: {', '.join(pii_found)}")

        # 5. Dangerous characters (code injection)
        dangerous_chars = ['<script>', '<?php', '${', '`']
        for char in dangerous_chars:
            if char in user_input.lower():
                errors.append(f"Dangerous character sequence detected: {char}")

        is_valid = len(errors) == 0 or all("detected and redacted" in e for e in errors)

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            sanitized_input=sanitized if is_valid else None
        )

# Usage
validator = InputValidator(max_length=5000)
result = validator.validate("My email is john@example.com and my SSN is 123-45-6789")

if result.is_valid:
    print(f"Sanitized input: {result.sanitized_input}")
    # Proceed with LLM call
else:
    print(f"Validation failed: {result.errors}")
    # Return error to user
```

**Source:** [Security - Input Validation](SECURITY.md#21-input-validation-layer)

### 3. Rate Limiter with Tiers

```python
import redis
import time
from typing import Optional
from enum import Enum

class UserTier(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class RateLimiter:
    # Requests per minute by tier
    TIER_LIMITS = {
        UserTier.FREE: 10,
        UserTier.PRO: 100,
        UserTier.ENTERPRISE: 1000
    }

    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis = redis.from_url(redis_url)

    def check_rate_limit(self, user_id: str, tier: UserTier) -> tuple[bool, Optional[int]]:
        """
        Check if user is within rate limit.

        Returns:
            (allowed: bool, retry_after_seconds: Optional[int])
        """
        key = f"rate_limit:{tier.value}:{user_id}"
        limit = self.TIER_LIMITS[tier]
        window = 60  # 1 minute window

        current_time = int(time.time())
        window_start = current_time - window

        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in current window
        request_count = self.redis.zcard(key)

        if request_count >= limit:
            # Get oldest request timestamp
            oldest = self.redis.zrange(key, 0, 0, withscores=True)
            if oldest:
                oldest_timestamp = int(oldest[0][1])
                retry_after = window - (current_time - oldest_timestamp)
                return False, retry_after
            return False, window

        # Add current request
        self.redis.zadd(key, {str(current_time): current_time})
        self.redis.expire(key, window)

        return True, None

# Usage with Flask
from flask import Flask, request, jsonify

app = Flask(__name__)
rate_limiter = RateLimiter()

@app.before_request
def check_rate_limit():
    user_id = request.headers.get("X-User-ID", "anonymous")
    tier_str = request.headers.get("X-User-Tier", "free")
    tier = UserTier(tier_str)

    allowed, retry_after = rate_limiter.check_rate_limit(user_id, tier)

    if not allowed:
        return jsonify({
            "error": "Rate limit exceeded",
            "retry_after_seconds": retry_after
        }), 429
```

**Source:** [Security - Rate Limiting](SECURITY.md#51-multi-tier-rate-limiting)

### 4. Complete Observability Wrapper

```python
import time
import json
import uuid
from typing import Callable, Any
from functools import wraps
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class LLMMetrics:
    request_id: str
    timestamp: str
    endpoint: str
    user_id: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    error: Optional[str] = None

class ObservabilityWrapper:
    def __init__(self):
        self.metrics_buffer = []

    def track_llm_call(self, func: Callable) -> Callable:
        """Decorator to track LLM calls."""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            request_id = str(uuid.uuid4())
            start_time = time.time()

            # Extract metadata
            endpoint = func.__name__
            user_id = kwargs.get("user_id", "anonymous")
            model = kwargs.get("model", "unknown")

            try:
                # Call function
                result = func(*args, **kwargs)

                # Extract metrics from result
                metrics = LLMMetrics(
                    request_id=request_id,
                    timestamp=datetime.utcnow().isoformat(),
                    endpoint=endpoint,
                    user_id=user_id,
                    model=model,
                    provider=result.get("provider", "unknown"),
                    input_tokens=result.get("input_tokens", 0),
                    output_tokens=result.get("output_tokens", 0),
                    cost_usd=result.get("cost_usd", 0.0),
                    latency_ms=(time.time() - start_time) * 1000,
                    cache_hit=result.get("cache_hit", False),
                    error=None
                )

                # Log metrics
                self._log_metrics(metrics)

                return result

            except Exception as e:
                # Log error
                metrics = LLMMetrics(
                    request_id=request_id,
                    timestamp=datetime.utcnow().isoformat(),
                    endpoint=endpoint,
                    user_id=user_id,
                    model=model,
                    provider="unknown",
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    cache_hit=False,
                    error=str(e)
                )
                self._log_metrics(metrics)
                raise

        return wrapper

    def _log_metrics(self, metrics: LLMMetrics):
        """Log metrics in JSON format."""
        print(json.dumps(asdict(metrics)))

        # Also buffer for batch processing
        self.metrics_buffer.append(metrics)

        # Flush buffer if too large
        if len(self.metrics_buffer) >= 100:
            self._flush_metrics()

    def _flush_metrics(self):
        """Flush metrics to storage (DB, Prometheus, etc.)."""
        # TODO: Implement batch write to time-series DB
        self.metrics_buffer = []

# Usage
obs = ObservabilityWrapper()

@obs.track_llm_call
def generate_summary(text: str, user_id: str, model: str = "claude-3-haiku-20240307"):
    # Your LLM call here
    return {
        "content": "Summary...",
        "provider": "anthropic",
        "input_tokens": 100,
        "output_tokens": 50,
        "cost_usd": 0.0001,
        "cache_hit": False
    }

# Automatically tracked
result = generate_summary("Long text...", user_id="user123")
```

**Source:** [Observability - Wrapper Pattern](OBSERVABILITY.md#51-complete-observability-wrapper)

### 5. GDPR Consent Manager

```python
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

class ConsentPurpose(Enum):
    ESSENTIAL = "essential"  # Required for service
    ANALYTICS = "analytics"  # Usage analytics
    MARKETING = "marketing"  # Marketing communications
    PERSONALIZATION = "personalization"  # Content personalization
    THIRD_PARTY = "third_party"  # Third-party integrations

class ConsentStatus(Enum):
    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"

@dataclass
class ConsentRecord:
    user_id: str
    purpose: ConsentPurpose
    status: ConsentStatus
    timestamp: datetime
    ip_address: str
    user_agent: str
    version: str  # Privacy policy version

class ConsentManager:
    def __init__(self, db_connection):
        self.db = db_connection
        self.current_policy_version = "1.0"

    def record_consent(
        self,
        user_id: str,
        purposes: List[ConsentPurpose],
        status: ConsentStatus,
        ip_address: str,
        user_agent: str
    ) -> List[ConsentRecord]:
        """Record user consent for specified purposes."""
        records = []

        for purpose in purposes:
            record = ConsentRecord(
                user_id=user_id,
                purpose=purpose,
                status=status,
                timestamp=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                version=self.current_policy_version
            )

            # Store in database
            self._store_consent(record)
            records.append(record)

        return records

    def check_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        """Check if user has granted consent for purpose."""
        query = """
            SELECT status FROM consent_records
            WHERE user_id = ? AND purpose = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """

        result = self.db.execute(query, (user_id, purpose.value))
        row = result.fetchone()

        if not row:
            # No consent recorded - assume denied except for essential
            return purpose == ConsentPurpose.ESSENTIAL

        return row[0] == ConsentStatus.GRANTED.value

    def withdraw_consent(
        self,
        user_id: str,
        purposes: List[ConsentPurpose],
        ip_address: str,
        user_agent: str
    ) -> List[ConsentRecord]:
        """Withdraw user consent (GDPR right)."""
        return self.record_consent(
            user_id=user_id,
            purposes=purposes,
            status=ConsentStatus.WITHDRAWN,
            ip_address=ip_address,
            user_agent=user_agent
        )

    def get_consent_history(self, user_id: str) -> List[ConsentRecord]:
        """Get full consent history for user (GDPR transparency)."""
        query = """
            SELECT * FROM consent_records
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """

        results = self.db.execute(query, (user_id,))
        return [self._row_to_record(row) for row in results]

    def _store_consent(self, record: ConsentRecord):
        """Store consent record in database."""
        query = """
            INSERT INTO consent_records
            (user_id, purpose, status, timestamp, ip_address, user_agent, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        self.db.execute(query, (
            record.user_id,
            record.purpose.value,
            record.status.value,
            record.timestamp.isoformat(),
            record.ip_address,
            record.user_agent,
            record.version
        ))
        self.db.commit()

    def _row_to_record(self, row) -> ConsentRecord:
        """Convert database row to ConsentRecord."""
        return ConsentRecord(
            user_id=row[0],
            purpose=ConsentPurpose(row[1]),
            status=ConsentStatus(row[2]),
            timestamp=datetime.fromisoformat(row[3]),
            ip_address=row[4],
            user_agent=row[5],
            version=row[6]
        )

# Usage with Flask
from flask import Flask, request, jsonify

app = Flask(__name__)
consent_mgr = ConsentManager(db_connection)

@app.route("/api/consent", methods=["POST"])
def record_consent():
    data = request.json
    user_id = data["user_id"]
    purposes = [ConsentPurpose(p) for p in data["purposes"]]

    records = consent_mgr.record_consent(
        user_id=user_id,
        purposes=purposes,
        status=ConsentStatus.GRANTED,
        ip_address=request.remote_addr,
        user_agent=request.headers.get("User-Agent", "")
    )

    return jsonify({"message": "Consent recorded", "count": len(records)})

@app.before_request
def check_consent():
    """Middleware to check consent before processing."""
    user_id = request.headers.get("X-User-ID")

    if not user_id:
        return

    # Check if analytics consent granted
    if not consent_mgr.check_consent(user_id, ConsentPurpose.ANALYTICS):
        # Disable analytics tracking for this request
        g.analytics_enabled = False
```

**Source:** [Compliance - Consent Management](COMPLIANCE.md#6-consent-management-platform)

---

## Configuration Examples

### 1. GitHub Actions CI/CD Workflow

```yaml
# .github/workflows/ai-app-ci.yml
name: AI Application CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"
  MOCK_LLM: "true"  # Use mocks in CI

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov black flake8

      - name: Lint with flake8
        run: |
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
          flake8 . --count --max-complexity=10 --max-line-length=88 --statistics

      - name: Format check with black
        run: black --check .

      - name: Run unit tests
        run: |
          pytest tests/unit --cov=src --cov-report=xml --cov-report=term
        env:
          MOCK_LLM: "true"

      - name: Run integration tests
        run: |
          pytest tests/integration -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          TEST_MODE: "true"

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: false

  security-scan:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run security scan
        run: |
          pip install bandit safety
          bandit -r src/ -f json -o bandit-report.json
          safety check --json

      - name: Check for secrets
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: ${{ github.event.repository.default_branch }}
          head: HEAD

  cost-check:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Estimate cost impact
        run: |
          python scripts/estimate_cost.py --branch ${{ github.ref }}

      - name: Comment on PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const cost = fs.readFileSync('cost_estimate.txt', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## Cost Estimate\n\n${cost}`
            });

  deploy-staging:
    needs: [test, security-scan, cost-check]
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to staging
        run: |
          # Your deployment script
          ./scripts/deploy.sh staging
        env:
          DEPLOY_KEY: ${{ secrets.STAGING_DEPLOY_KEY }}

      - name: Run smoke tests
        run: |
          ./scripts/smoke_test.sh https://staging.example.com

  deploy-production:
    needs: [test, security-scan, cost-check]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Deploy to production
        run: |
          ./scripts/deploy.sh production
        env:
          DEPLOY_KEY: ${{ secrets.PROD_DEPLOY_KEY }}

      - name: Run smoke tests
        run: |
          ./scripts/smoke_test.sh https://api.example.com

      - name: Notify team
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Production deployment ${{ job.status }}'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

**Source:** [Testing - CI/CD](TESTING.md#101-github-actions-workflow)

### 2. Prometheus Metrics Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Alerting configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Rule files
rule_files:
  - "rules/*.yml"

# Scrape configurations
scrape_configs:
  # AI application metrics
  - job_name: 'ai-app'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  # Redis cache metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

  # PostgreSQL metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']
```

```yaml
# rules/ai_alerts.yml
groups:
  - name: ai_cost_alerts
    interval: 60s
    rules:
      # Daily cost threshold
      - alert: DailyCostHigh
        expr: sum(increase(llm_cost_usd_total[24h])) > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Daily AI cost exceeds $100"
          description: "Total cost in last 24h: ${{ $value }}"

      # Cost spike detection
      - alert: CostSpike
        expr: |
          rate(llm_cost_usd_total[5m]) >
          4 * avg_over_time(rate(llm_cost_usd_total[5m])[1h:5m])
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "AI cost spike detected"
          description: "Cost rate 4x higher than average"

  - name: ai_performance_alerts
    interval: 30s
    rules:
      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.95, llm_request_duration_seconds_bucket) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "95th percentile latency > 5s"
          description: "p95 latency: {{ $value }}s"

      # High error rate
      - alert: HighErrorRate
        expr: |
          rate(llm_requests_total{status="error"}[5m]) /
          rate(llm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5%"
          description: "Error rate: {{ $value | humanizePercentage }}"

      # Low cache hit rate
      - alert: LowCacheHitRate
        expr: |
          rate(llm_cache_hits_total[10m]) /
          (rate(llm_cache_hits_total[10m]) + rate(llm_cache_misses_total[10m])) < 0.4
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Cache hit rate < 40%"
          description: "Hit rate: {{ $value | humanizePercentage }}"
```

**Sources:**
- [Observability - Prometheus Setup](OBSERVABILITY.md#7-prometheus-configuration)
- [Metrics - Collection](METRICS.md#9-collection-methods)

### 3. Environment Configuration Template

```bash
# .env.example
# Copy to .env and fill in values

# =============================================================================
# API KEYS (NEVER COMMIT ACTUAL KEYS)
# =============================================================================
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
PRIMARY_MODEL=claude-3-haiku-20240307
FALLBACK_MODEL=gpt-3.5-turbo
MAX_TOKENS=1024
TEMPERATURE=0.7

# =============================================================================
# CACHING
# =============================================================================
REDIS_URL=redis://localhost:6379
CACHE_TTL=86400  # 24 hours
CACHE_ENABLED=true

# =============================================================================
# RATE LIMITING
# =============================================================================
RATE_LIMIT_FREE=10  # requests per minute
RATE_LIMIT_PRO=100
RATE_LIMIT_ENTERPRISE=1000

# =============================================================================
# SECURITY
# =============================================================================
JWT_SECRET=your-secret-key-here
JWT_EXPIRY=3600  # 1 hour
API_KEY_ROTATION_DAYS=90
MAX_INPUT_LENGTH=10000
ENABLE_PII_DETECTION=true
ENABLE_PROMPT_INJECTION_DETECTION=true

# =============================================================================
# OBSERVABILITY
# =============================================================================
LOG_LEVEL=INFO
LOG_FORMAT=json
METRICS_PORT=8000
ENABLE_TRACING=true
JAEGER_ENDPOINT=http://localhost:14268/api/traces

# =============================================================================
# COST MANAGEMENT
# =============================================================================
DAILY_BUDGET_USD=100.0
MONTHLY_BUDGET_USD=3000.0
ALERT_THRESHOLD_PERCENT=80
ENABLE_BUDGET_ENFORCEMENT=true

# =============================================================================
# COMPLIANCE
# =============================================================================
DATA_RETENTION_DAYS=90
AUDIT_LOG_RETENTION_DAYS=2555  # 7 years
ENABLE_GDPR_MODE=true
ENABLE_HIPAA_MODE=false
PRIVACY_POLICY_VERSION=1.0

# =============================================================================
# DATABASE
# =============================================================================
DATABASE_URL=postgresql://user:pass@localhost:5432/aiapp
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# =============================================================================
# TESTING
# =============================================================================
TEST_MODE=false
MOCK_LLM_ENABLED=false
```

**Source:** [AI Development - Configuration](AI_DEVELOPMENT.md#22-configuration-management)

---

## Metrics & Thresholds

### Critical Metrics Dashboard

| Metric | Target | Warning | Critical | Alert |
|--------|--------|---------|----------|-------|
| **Cost per request** | <$0.01 | >$0.05 | >$0.10 | Slack + PagerDuty |
| **Daily cost** | Budget | >80% | >95% | Email + Slack |
| **Monthly forecast** | Budget | >90% | >100% | Email immediately |
| **Error rate** | <1% | >5% | >10% | PagerDuty |
| **p50 latency** | <500ms | >1s | >2s | Slack |
| **p95 latency** | <2s | >5s | >10s | PagerDuty |
| **p99 latency** | <5s | >10s | >20s | PagerDuty |
| **Cache hit rate** | >60% | <40% | <20% | Email |
| **Availability** | >99.9% | <99% | <95% | PagerDuty |
| **Input validation failures** | <0.1% | >1% | >5% | Slack |
| **Prompt injection attempts** | 0 | >10/hour | >50/hour | Security team |
| **Rate limit violations** | <5% | >10% | >20% | Email |
| **Failed DSR requests** | 0 | >0 | >5 | Legal team |

### Cost Metrics (Hourly)

```promql
# Total cost (last hour)
sum(increase(llm_cost_usd_total[1h]))

# Cost by model (last hour)
sum by (model) (increase(llm_cost_usd_total[1h]))

# Cost by endpoint (last hour)
sum by (endpoint) (increase(llm_cost_usd_total[1h]))

# Cost per request (average, last hour)
sum(increase(llm_cost_usd_total[1h])) / sum(increase(llm_requests_total[1h]))

# Most expensive users (top 10)
topk(10, sum by (user_id) (increase(llm_cost_usd_total[1h])))

# Cost burn rate (will hit daily budget in X hours)
(daily_budget_usd) / (sum(rate(llm_cost_usd_total[1h])))
```

### Performance Metrics

```promql
# p50 latency
histogram_quantile(0.50, rate(llm_request_duration_seconds_bucket[5m]))

# p95 latency
histogram_quantile(0.95, rate(llm_request_duration_seconds_bucket[5m]))

# p99 latency
histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m]))

# Error rate (%)
100 * (
  rate(llm_requests_total{status="error"}[5m]) /
  rate(llm_requests_total[5m])
)

# Cache hit rate (%)
100 * (
  rate(llm_cache_hits_total[5m]) /
  (rate(llm_cache_hits_total[5m]) + rate(llm_cache_misses_total[5m]))
)

# Requests per second
rate(llm_requests_total[1m])
```

### Security Metrics

```promql
# Prompt injection attempts (per hour)
sum(increase(security_prompt_injection_blocked_total[1h]))

# PII detection events (per hour)
sum(increase(security_pii_detected_total[1h]))

# Rate limit violations (per hour)
sum(increase(rate_limit_exceeded_total[1h]))

# Failed authentication attempts (per hour)
sum(increase(auth_failed_total[1h]))

# Suspicious activity score (0-100)
(
  sum(rate(security_prompt_injection_blocked_total[5m])) * 50 +
  sum(rate(auth_failed_total[5m])) * 30 +
  sum(rate(rate_limit_exceeded_total[5m])) * 20
)
```

**Sources:**
- [Metrics Guide - Complete Catalog](METRICS.md)
- [Observability - Queries](OBSERVABILITY.md#9-cost-analysis-queries)

---

## Quick Decisions

### When to Use LLM vs. Code

| Task | Use LLM? | Alternative | Reason |
|------|----------|-------------|--------|
| Extract email from text | ❌ No | Regex: `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}\b` | 100% accurate, free |
| Parse date string | ❌ No | `dateutil.parser.parse()` | Handles 100+ formats |
| Detect language | ❌ No | `langdetect` library | 99%+ accurate |
| Sentiment (obvious) | ❌ No | `TextBlob` or `VADER` | 70-80% cases |
| Check profanity | ❌ No | `better-profanity` library | Deterministic |
| Validate JSON | ❌ No | `json.loads()` + try/except | Perfect accuracy |
| Format phone number | ❌ No | `phonenumbers` library | International support |
| Calculate math | ❌ No | `eval()` (safely) or `sympy` | 100% accurate |
| Classify intent (simple) | ⚠️ Maybe | Rule-based classifier first | Try rules, then LLM |
| Sentiment (nuanced) | ✅ Yes | LLM (Haiku) | Handles sarcasm, context |
| Summarize text | ✅ Yes | LLM (Haiku/Sonnet) | Complex task |
| Generate creative content | ✅ Yes | LLM (Sonnet/Opus) | Requires creativity |
| Answer questions | ✅ Yes | LLM (Haiku+) | Knowledge-based |
| Code generation | ✅ Yes | LLM (Sonnet/Opus) | Complex reasoning |

**Source:** [Architecture - Decision Matrix](ARCHITECTURE.md#4-decision-matrix-when-to-use-llm)

### Model Selection Guide

| Complexity | Model | Cost/1M tok (out) | When to Use | Examples |
|------------|-------|-------------------|-------------|----------|
| **Simple** | Haiku | $1.25 | Clear instructions, structured output | Extract info, classify, simple Q&A, formatting |
| **Medium** | Sonnet | $15.00 | Moderate reasoning, multi-step | Summarization, analysis, code review |
| **Complex** | Opus | $75.00 | Deep reasoning, creativity | Code generation, complex analysis, research |

**Selection Algorithm:**
1. Start with Haiku
2. If accuracy <90% on eval set → try Sonnet
3. If accuracy still <95% AND task requires deep reasoning → use Opus
4. Never use Opus without data proving Sonnet is insufficient

**Source:** [Cost Reduction - Model Selection](COST_REDUCTION_RULES.md#rule-11-use-the-smallest-capable-model)

### Security Threat Response

| Threat | Severity | Immediate Action | Follow-up |
|--------|----------|------------------|-----------|
| **Prompt injection detected** | 🟡 Medium | Block request, log, return error | Review pattern, update detection rules |
| **Multiple injection attempts** | 🔴 High | Block user, alert security team | Investigate user, check for breach |
| **PII detected in prompt** | 🟡 Medium | Redact, log, continue | Review why PII was submitted |
| **PII leaked in response** | 🔴 Critical | Block response, log, alert DPO | Incident response, notify affected users |
| **Rate limit exceeded** | 🟢 Low | Return 429 error with retry-after | Monitor for abuse patterns |
| **Repeated rate limit violations** | 🟡 Medium | Temporarily ban IP/user | Investigate for DDoS or scraping |
| **Failed authentication (1-3)** | 🟢 Low | Return 401 error | Normal operation |
| **Failed authentication (10+)** | 🟡 Medium | Block IP, alert security | Check for credential stuffing |
| **Cost spike (2x normal)** | 🟡 Medium | Alert on-call, review logs | Identify cause, optimize or block |
| **Cost spike (10x normal)** | 🔴 Critical | Enable circuit breaker, page team | Emergency investigation |
| **API key exposed** | 🔴 Critical | Rotate key immediately, revoke old | Audit usage, notify security |
| **Data exfiltration suspected** | 🔴 Critical | Block requests, isolate system | Full incident response, forensics |

**Sources:**
- [Security - Incident Response](SECURITY.md#102-incident-response)
- [Security Architecture - Automated Response](SECURITY_ARCHITECTURE.md#61-automated-incident-response)

---

## Emergency Procedures

### 🚨 Emergency: Cost Spike Detected

**Symptoms:** Cost 5-10x higher than normal, budget alerts firing

**Immediate Actions (< 5 minutes):**

1. **Enable circuit breaker** (stops all LLM calls):
   ```bash
   curl -X POST http://localhost:8000/admin/circuit-breaker \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -d '{"enabled": true, "reason": "cost spike emergency"}'
   ```

2. **Check cost dashboard** - Identify source:
   ```bash
   # Top endpoints by cost (last hour)
   curl http://localhost:8000/metrics | grep llm_cost_usd_total

   # Or Grafana dashboard: Cost > By Endpoint > Last 1h
   ```

3. **Block expensive operations**:
   ```bash
   # Disable specific endpoint
   curl -X POST http://localhost:8000/admin/disable-endpoint \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -d '{"endpoint": "/api/expensive-operation"}'
   ```

**Investigation (< 15 minutes):**

1. **Query logs** for anomalies:
   ```bash
   # Find requests with high token usage
   cat logs/app.log | jq 'select(.output_tokens > 10000)' | head -20

   # Find specific user causing spike
   cat logs/app.log | jq -r '.user_id' | sort | uniq -c | sort -rn | head -10
   ```

2. **Common causes:**
   - User submitting very long prompts → Add max_tokens enforcement
   - Infinite loop in code → Fix code, deploy hotfix
   - DDoS/abuse → Block user/IP
   - Caching broken → Restart Redis, verify cache layer
   - Wrong model selected → Fix model router logic

3. **Temporary mitigation:**
   ```python
   # Add emergency rate limit
   EMERGENCY_RATE_LIMIT = 1  # 1 req/min

   # Or reduce max_tokens globally
   MAX_TOKENS_EMERGENCY = 100
   ```

**Recovery (< 30 minutes):**

1. Fix root cause
2. Deploy hotfix if needed
3. Gradually re-enable services
4. Monitor closely for 1 hour
5. Write incident report

**Prevention:**
- [ ] Set up budget alerts at 80%, 95%, 100%
- [ ] Implement circuit breaker pattern
- [ ] Add max_tokens limits on all calls
- [ ] Review and test emergency procedures monthly

**Source:** [Observability - Cost Spike Response](OBSERVABILITY.md#43-cost-spike-alerts)

### 🔐 Emergency: Security Breach Suspected

**Symptoms:** Unusual activity, data exfiltration alerts, multiple failed auth attempts

**Immediate Actions (< 2 minutes):**

1. **Isolate the system**:
   ```bash
   # Enable maintenance mode (blocks all requests)
   curl -X POST http://localhost:8000/admin/maintenance-mode \
     -H "Authorization: Bearer $ADMIN_TOKEN" \
     -d '{"enabled": true}'
   ```

2. **Rotate all credentials**:
   ```bash
   # Rotate API keys immediately
   ./scripts/rotate_api_keys.sh --emergency

   # Revoke all active JWT tokens
   curl -X POST http://localhost:8000/admin/revoke-all-tokens \
     -H "Authorization: Bearer $ADMIN_TOKEN"
   ```

3. **Preserve evidence**:
   ```bash
   # Copy logs to secure location
   cp -r logs/ /secure/incident-$(date +%Y%m%d-%H%M%S)/

   # Snapshot database
   pg_dump dbname > /secure/db-snapshot-$(date +%Y%m%d-%H%M%S).sql
   ```

**Investigation (< 30 minutes):**

1. **Analyze logs** for breach indicators:
   ```bash
   # Find suspicious patterns
   cat logs/app.log | jq 'select(.event == "prompt_injection" or .event == "auth_failed")'

   # Check for data exfiltration
   cat logs/app.log | jq 'select(.output_tokens > 50000)'
   ```

2. **Identify compromised accounts:**
   ```bash
   # Users with suspicious activity
   cat logs/security.log | jq 'select(.severity == "critical") | .user_id' | sort | uniq
   ```

3. **Check for exposed data:**
   ```bash
   # Review audit logs
   SELECT * FROM audit_logs
   WHERE event_type = 'data_access'
   AND timestamp > NOW() - INTERVAL '24 hours'
   ORDER BY timestamp DESC;
   ```

**Notification (< 1 hour):**

1. **Notify stakeholders:**
   - Security team
   - Legal/compliance team
   - Executive leadership
   - Affected users (if GDPR/CCPA applies)

2. **Regulatory notification:**
   - GDPR: 72 hours to notify supervisory authority
   - CCPA: "Without unreasonable delay"
   - HIPAA: 60 days for breach affecting >500 individuals

3. **User notification template:**
   ```
   Subject: Security Incident Notification

   We detected unauthorized access to our systems on [DATE].

   What happened: [Brief description]
   What data was affected: [Specific data types]
   What we're doing: [Response actions]
   What you should do: [User actions, e.g., change password]

   Contact: security@example.com
   ```

**Recovery:**

1. Fix vulnerability
2. Harden security controls
3. Re-enable services gradually
4. Monitor closely for 7 days
5. Complete incident report
6. Update security policies

**Sources:**
- [Security - Incident Response](SECURITY.md#102-incident-response)
- [Compliance - Breach Notification](COMPLIANCE.md#9-data-breach-response)

### ⚖️ Emergency: GDPR Data Subject Rights Request

**Scenario:** User exercises GDPR rights (access, erasure, portability)

**Timeline: 30 days maximum** (GDPR Article 12)

**Right to Access (Article 15):**

1. **Collect all user data:**
   ```sql
   -- Export all user data
   SELECT * FROM users WHERE user_id = 'USER_ID';
   SELECT * FROM prompts WHERE user_id = 'USER_ID';
   SELECT * FROM responses WHERE user_id = 'USER_ID';
   SELECT * FROM audit_logs WHERE user_id = 'USER_ID';
   SELECT * FROM consent_records WHERE user_id = 'USER_ID';
   ```

2. **Package as downloadable file:**
   ```python
   import json
   from datetime import datetime

   data_package = {
       "request_date": datetime.utcnow().isoformat(),
       "user_id": user_id,
       "personal_data": {
           "profile": user_profile,
           "prompts": user_prompts,
           "responses": user_responses,
           "consent_history": consent_records
       },
       "processing_purposes": ["Service delivery", "Analytics"],
       "data_recipients": ["Anthropic (subprocessor)", "AWS (hosting)"],
       "retention_period": "90 days"
   }

   with open(f"user_data_{user_id}.json", "w") as f:
       json.dump(data_package, f, indent=2)
   ```

3. **Deliver securely** (encrypted email or download link)

**Right to Erasure (Article 17):**

1. **Delete user data:**
   ```sql
   -- Mark for deletion (soft delete first)
   UPDATE users SET deleted_at = NOW() WHERE user_id = 'USER_ID';
   UPDATE prompts SET deleted_at = NOW() WHERE user_id = 'USER_ID';
   UPDATE responses SET deleted_at = NOW() WHERE user_id = 'USER_ID';

   -- Hard delete after grace period (7 days)
   DELETE FROM users WHERE user_id = 'USER_ID' AND deleted_at < NOW() - INTERVAL '7 days';
   ```

2. **Document exceptions** (legal obligations):
   ```python
   # Data that CANNOT be deleted
   exceptions = {
       "audit_logs": "Legal obligation (7-year retention)",
       "financial_records": "Tax law requirements",
       "consent_records": "Proof of compliance"
   }
   ```

3. **Notify subprocessors:**
   ```python
   # Notify Anthropic to delete data
   requests.post(
       "https://api.anthropic.com/data-deletion",
       headers={"X-API-Key": ANTHROPIC_API_KEY},
       json={"user_id": user_id, "reason": "GDPR Article 17"}
   )
   ```

**Right to Portability (Article 20):**

1. **Export in machine-readable format:**
   ```python
   # Export as JSON
   export_data = {
       "format": "JSON",
       "version": "1.0",
       "export_date": datetime.utcnow().isoformat(),
       "data": {
           "prompts": [...],
           "preferences": {...},
           "consent": [...]
       }
   }

   # Or CSV for tabular data
   import pandas as pd
   df = pd.DataFrame(user_data)
   df.to_csv(f"user_data_{user_id}.csv", index=False)
   ```

**DSR Request Tracking:**

```python
# Log all DSR requests
dsr_log = {
    "request_id": str(uuid.uuid4()),
    "user_id": user_id,
    "request_type": "access",  # access, erasure, portability, rectification
    "received_date": datetime.utcnow(),
    "due_date": datetime.utcnow() + timedelta(days=30),
    "status": "pending",  # pending, in_progress, completed
    "completed_date": None
}
```

**Sources:**
- [Compliance - Data Subject Rights](COMPLIANCE.md#21-core-requirements)
- [Compliance Architecture - DSR Automation](COMPLIANCE_ARCHITECTURE.md#3-data-subject-rights-dsr-architecture)

---

## Reference Tables

### Model Pricing (February 2026)

| Provider | Model | Input ($/1M tok) | Output ($/1M tok) | Best For |
|----------|-------|------------------|-------------------|----------|
| Anthropic | Claude 3 Haiku | $0.25 | $1.25 | 80% of tasks: simple classification, extraction, formatting |
| Anthropic | Claude 3 Sonnet | $3.00 | $15.00 | 15% of tasks: analysis, summarization, moderate reasoning |
| Anthropic | Claude 3 Opus | $15.00 | $75.00 | 5% of tasks: complex reasoning, creative writing, research |
| OpenAI | GPT-3.5 Turbo | $0.50 | $1.50 | Fallback for simple tasks |
| OpenAI | GPT-4 Turbo | $10.00 | $30.00 | Fallback for medium tasks |
| OpenAI | GPT-4 | $30.00 | $60.00 | Most expensive - avoid if possible |

**Cost Example (100 requests/day):**
- 100% Haiku: 100 req × 1000 tokens × $1.25/1M = **$0.13/day** = **$3.75/month**
- 100% Sonnet: 100 req × 1000 tokens × $15/1M = **$1.50/day** = **$45/month**
- 100% Opus: 100 req × 1000 tokens × $75/1M = **$7.50/day** = **$225/month**

**Savings: Haiku vs Opus = 98% cost reduction**

**Source:** [Cost Reduction - Pricing](COST_REDUCTION_RULES.md#cost-calculation-reference)

### Security Attack Patterns

| Attack Type | Pattern | Detection Method | Response |
|-------------|---------|------------------|----------|
| **Prompt Injection** | "Ignore previous instructions" | Regex + ML classifier | Block request, log, alert |
| **Jailbreaking** | "DAN mode", "Developer mode" | Keyword detection | Block request, log user |
| **PII Leakage** | Email, phone, SSN in prompt | Regex patterns | Redact + continue |
| **API Key Theft** | Repeated auth failures | Rate tracking | Block IP after 5 failures |
| **Token Stuffing** | Extremely long prompts (>50K tokens) | Length check | Reject with 400 error |
| **Cache Poisoning** | Repeated similar requests with variations | Similarity hashing | Rate limit + review |
| **Model Manipulation** | Invalid model names, SQL injection | Input validation | Reject with 400 error |
| **Session Hijacking** | JWT token reuse, expired tokens | Token validation | Reject + force re-auth |

**Source:** [Security - Threat Model](SECURITY.md#1-threat-model-for-ai-applications)

### Compliance Requirements by Region

| Regulation | Region | Key Requirements | Penalties |
|------------|--------|------------------|-----------|
| **GDPR** | EU/EEA | Consent, access, erasure, portability, data protection officer (if >250 employees) | Up to €20M or 4% of global revenue |
| **CCPA/CPRA** | California, US | Right to know, delete, opt-out of sale, "Do Not Sell" link | Up to $7,500 per intentional violation |
| **HIPAA** | US (Healthcare) | BAA required, PHI encryption, audit logs (6 years), breach notification | Up to $1.5M per violation category per year |
| **PIPEDA** | Canada | Consent, access, accuracy, safeguards, accountability | Up to CAD $100,000 |
| **LGPD** | Brazil | Similar to GDPR: consent, access, erasure, portability | Up to 2% of revenue (max R$50M per violation) |
| **PDPA** | Singapore | Consent, purpose limitation, accuracy, protection | Up to SGD $1M |

**Source:** [Compliance - Regulatory Frameworks](COMPLIANCE.md#1-regulatory-frameworks)

---

## Additional Resources

### Documentation Quick Links

**By Role:**
- **Developers:** [AI Development](AI_DEVELOPMENT.md), [Integration](INTEGRATION.md), [Testing](TESTING.md)
- **DevOps:** [System Architecture](SYSTEM_ARCHITECTURE.md), [Observability](OBSERVABILITY.md), [CI/CD](AI_TESTING_ARCHITECTURE.md#4-cicd-pipeline-architecture)
- **Security Team:** [Security](SECURITY.md), [Security Architecture](SECURITY_ARCHITECTURE.md)
- **Compliance Team:** [Compliance](COMPLIANCE.md), [Compliance Architecture](COMPLIANCE_ARCHITECTURE.md)
- **Cost Owners:** [Cost Reduction](COST_REDUCTION_RULES.md), [Metrics](METRICS.md)
- **Architects:** [Architecture](ARCHITECTURE.md), [Cost-Efficient Architecture](COST_EFFICIENT_ARCHITECTURE.md)

**By Task:**
- **Starting new project:** [Development Checklist](AI_DEVELOPMENT.md#10-development-checklist)
- **Optimizing costs:** [Cost Reduction Quick Wins](COST_REDUCTION_RULES.md#12-quick-wins-checklist)
- **Improving security:** [Security Checklist](SECURITY.md#11-security-checklist)
- **Setting up monitoring:** [Observability Setup](OBSERVABILITY.md#10-observability-checklist)
- **Ensuring compliance:** [Compliance Checklist](COMPLIANCE.md#10-compliance-checklist)
- **Writing tests:** [Testing Guide](TESTING.md)

**By Problem:**
- **High costs:** [Cost Reduction Rules](COST_REDUCTION_RULES.md) → [Architecture Patterns](ARCHITECTURE.md#7-architecture-patterns)
- **Security incidents:** [Incident Response](SECURITY.md#102-incident-response) → [Emergency Procedures](#emergency-procedures)
- **Slow responses:** [Performance Optimization](SYSTEM_ARCHITECTURE.md#4-performance-optimization) → [Caching](INTEGRATION.md#7-caching-integration)
- **GDPR requests:** [Data Subject Rights](COMPLIANCE.md#21-core-requirements) → [Emergency: GDPR DSR](#-emergency-gdpr-data-subject-rights-request)

### External Tools & Resources

**Cost Management:**
- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [OpenAI Pricing](https://openai.com/pricing)
- [Token Counter Tool](https://platform.openai.com/tokenizer)

**Security:**
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [Prompt Injection Playground](https://www.promptingguide.ai/risks/adversarial)

**Compliance:**
- [GDPR Official Text](https://gdpr-info.eu/)
- [CCPA Official Text](https://oag.ca.gov/privacy/ccpa)
- [HIPAA Guidelines](https://www.hhs.gov/hipaa)

**Monitoring:**
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [OpenTelemetry](https://opentelemetry.io/)

---

## Version & Updates

**Version:** 1.0
**Last Updated:** February 6, 2026
**Consolidated From:** 19,812 lines across 15 documents

**Document Status:** ✅ Active

**Feedback:** [GitHub Issues](https://github.com/blackjackptit/ai-development-policies/issues)

---

**⚡ Pro Tip:** Bookmark this page and keep it open while developing. It contains 90% of what you'll need daily.

**🎯 Remember:** LLMs are expensive last-resort tools. Always try code, rules, and cache first.
