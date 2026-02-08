# AI Integration Architecture

## Overview

This document provides detailed architectural patterns, designs, and blueprints for integrating AI/LLM capabilities into applications. It covers system design, component architecture, data flow, and scalable integration patterns.

---

## 1. High-Level Integration Architecture

### 1.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Applications                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   Web    │  │  Mobile  │  │   CLI    │  │  Desktop │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
└───────┼─────────────┼─────────────┼─────────────┼─────────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                      │
        ┌─────────────▼─────────────┐
        │      API Gateway          │
        │  • Authentication         │
        │  • Rate Limiting          │
        │  • Request Routing        │
        └─────────────┬─────────────┘
                      │
        ┌─────────────▼─────────────────────────────────────┐
        │           Application Service Layer               │
        │                                                   │
        │  ┌──────────────┐  ┌──────────────┐            │
        │  │ AI Processor │  │ Data Service │            │
        │  │   Layer      │  │              │            │
        │  │              │  │              │            │
        │  │ ┌──────────┐ │  │ ┌──────────┐ │            │
        │  │ │Rule      │ │  │ │Cache     │ │            │
        │  │ │Engine    │ │  │ │Layer     │ │            │
        │  │ └──────────┘ │  │ └──────────┘ │            │
        │  │              │  │              │            │
        │  │ ┌──────────┐ │  │ ┌──────────┐ │            │
        │  │ │LLM       │ │  │ │Database  │ │            │
        │  │ │Router    │ │  │ │          │ │            │
        │  │ └──────────┘ │  │ └──────────┘ │            │
        │  └──────────────┘  └──────────────┘            │
        └───────────────────────┬───────────────────────────┘
                                │
        ┌───────────────────────▼───────────────────────┐
        │        LLM Abstraction Layer                  │
        │  • Provider Management                        │
        │  • Fallback Logic                            │
        │  • Token Counting                            │
        │  • Cost Tracking                             │
        └───────────────────────┬───────────────────────┘
                                │
        ┌───────────────────────▼───────────────────────┐
        │         External LLM Providers                │
        │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
        │  │Anthropic│  │ OpenAI  │  │ Others  │      │
        │  └─────────┘  └─────────┘  └─────────┘      │
        └───────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              Observability Layer (Cross-Cutting)                │
│  • Metrics (Prometheus)  • Logs (ELK)  • Tracing (Jaeger)     │
│  • Alerts (PagerDuty)    • Dashboards (Grafana)               │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

```python
class ComponentArchitecture:
    """Define clear component boundaries"""

    COMPONENTS = {
        'api_gateway': {
            'responsibilities': [
                'Authentication/Authorization',
                'Rate limiting (coarse-grained)',
                'Request routing',
                'SSL termination',
                'CORS handling',
            ],
            'does_not': [
                'Business logic',
                'LLM calls',
                'Data persistence',
            ],
        },

        'ai_processor': {
            'responsibilities': [
                'Request preprocessing',
                'Rule-based routing',
                'LLM orchestration',
                'Response post-processing',
                'Cost calculation',
            ],
            'does_not': [
                'Authentication',
                'Data storage',
                'External API calls (except LLM)',
            ],
        },

        'llm_abstraction': {
            'responsibilities': [
                'Provider abstraction',
                'Fallback handling',
                'Token counting',
                'Retry logic',
                'Circuit breaking',
            ],
            'does_not': [
                'Business logic',
                'Caching decisions',
                'User data storage',
            ],
        },

        'data_service': {
            'responsibilities': [
                'Cache management',
                'Database operations',
                'Data validation',
                'Query optimization',
            ],
            'does_not': [
                'LLM calls',
                'Business logic',
                'Authentication',
            ],
        },
    }
```

---

## 2. Layered Architecture Pattern

### 2.1 Four-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LAYER 1: API Layer                       │
│  Responsibility: HTTP handling, validation, serialization   │
│  Technology: FastAPI, Flask, Express                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  REST API   │  │  GraphQL    │  │  WebSocket  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 LAYER 2: Service Layer                      │
│  Responsibility: Business logic, orchestration              │
│  Technology: Python/Node.js services                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  AI Service │  │ User Service│  │ Auth Service│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 3: Integration Layer                     │
│  Responsibility: External integrations, adapters            │
│  Technology: Provider SDKs, HTTP clients                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ LLM Adapter │  │DB Adapter   │  │Cache Adapter│        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              LAYER 4: Infrastructure Layer                  │
│  Responsibility: External systems and storage               │
│  Technology: PostgreSQL, Redis, LLM APIs                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Database   │  │    Cache    │  │  LLM APIs   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Layer Implementation

```python
# Layer 1: API Layer
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    model: str = "haiku"
    max_tokens: int = 1024

@app.post("/api/v1/generate")
async def generate_endpoint(
    request: GenerateRequest,
    user: User = Depends(get_current_user)
):
    """API Layer: Handle HTTP, validate, delegate to service"""
    try:
        # Delegate to service layer
        result = await ai_service.generate(
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            user_id=user.id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Layer 2: Service Layer
class AIService:
    """Service Layer: Business logic and orchestration"""

    def __init__(self, llm_integration, cache_service, metrics):
        self.llm = llm_integration
        self.cache = cache_service
        self.metrics = metrics

    async def generate(self, prompt: str, model: str, max_tokens: int, user_id: str):
        """Orchestrate the generation workflow"""
        # Business logic
        validated_prompt = self._validate_prompt(prompt)

        # Check cache
        cache_key = self._get_cache_key(validated_prompt, model)
        cached = await self.cache.get(cache_key)
        if cached:
            self.metrics.record_cache_hit()
            return cached

        # Delegate to integration layer
        result = await self.llm.generate(
            prompt=validated_prompt,
            model=model,
            max_tokens=max_tokens
        )

        # Cache result
        await self.cache.set(cache_key, result, ttl=3600)

        # Record metrics
        self.metrics.record_generation(result, user_id)

        return result


# Layer 3: Integration Layer
class LLMIntegration:
    """Integration Layer: Abstract external LLM providers"""

    def __init__(self):
        self.anthropic = AnthropicAdapter()
        self.openai = OpenAIAdapter()

    async def generate(self, prompt: str, model: str, max_tokens: int):
        """Delegate to appropriate provider"""
        try:
            if model in ['haiku', 'sonnet', 'opus']:
                return await self.anthropic.generate(prompt, model, max_tokens)
            else:
                return await self.openai.generate(prompt, model, max_tokens)
        except Exception as e:
            # Fallback logic
            return await self._fallback_generate(prompt, model, max_tokens)


# Layer 4: Infrastructure Layer (External)
# PostgreSQL, Redis, Anthropic API, OpenAI API
```

---

## 3. Microservices Architecture

### 3.1 Service Decomposition

```
┌─────────────────────────────────────────────────────────────┐
│                     API Gateway                             │
│                 (Kong / AWS API Gateway)                    │
└────┬──────────┬──────────┬──────────┬──────────┬──────────┘
     │          │          │          │          │
     ▼          ▼          ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│  Auth   │ │  AI    │ │ User   │ │ Billing│ │Analytics│
│ Service │ │Service │ │Service │ │Service │ │Service │
│         │ │        │ │        │ │        │ │        │
│ JWT     │ │ LLM    │ │ CRUD   │ │ Stripe │ │ Events │
│ OAuth   │ │ Router │ │ Profile│ │ Usage  │ │ Metrics│
└─────────┘ └────┬───┘ └────────┘ └────────┘ └────────┘
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Prompt   │ │   LLM    │ │ Response │
│Processor │ │ Manager  │ │ Validator│
│          │ │          │ │          │
│ Rules    │ │ Fallback │ │ Filter   │
│ Cache    │ │ Retry    │ │ Format   │
└──────────┘ └──────────┘ └──────────┘

┌─────────────────────────────────────────────────────────────┐
│                  Shared Infrastructure                      │
│  • Message Queue (RabbitMQ/Kafka)                          │
│  • Service Discovery (Consul/Eureka)                       │
│  • Config Service (Vault)                                  │
│  • Logging (ELK)                                           │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Service Communication Patterns

```python
# Pattern 1: Synchronous REST
class AIServiceClient:
    """Synchronous HTTP communication"""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)

    def generate(self, prompt: str) -> dict:
        """Synchronous REST call"""
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={"prompt": prompt}
        )
        response.raise_for_status()
        return response.json()


# Pattern 2: Asynchronous Message Queue
class AIServiceAsync:
    """Asynchronous message-based communication"""

    def __init__(self, queue_connection):
        self.queue = queue_connection

    async def generate_async(self, prompt: str, callback_url: str):
        """Submit to queue, return immediately"""
        job = {
            'id': str(uuid.uuid4()),
            'prompt': prompt,
            'callback_url': callback_url,
            'submitted_at': datetime.utcnow().isoformat(),
        }

        # Publish to queue
        await self.queue.publish('ai.generation.requests', job)

        return {'job_id': job['id'], 'status': 'queued'}


# Pattern 3: Event-Driven
class EventBus:
    """Event-driven architecture"""

    def __init__(self):
        self.subscribers = {}

    def subscribe(self, event_type: str, handler):
        """Subscribe to events"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event_type: str, data: dict):
        """Publish events to subscribers"""
        if event_type in self.subscribers:
            for handler in self.subscribers[event_type]:
                await handler(data)


# Usage
event_bus = EventBus()

# Subscribe to generation complete events
event_bus.subscribe('generation.completed', on_generation_complete)
event_bus.subscribe('generation.completed', update_billing)
event_bus.subscribe('generation.completed', log_metrics)

# Publish event
await event_bus.publish('generation.completed', {
    'user_id': 'user123',
    'cost': 0.001,
    'tokens': 300,
})
```

---

## 4. Data Flow Architecture

### 4.1 Request Flow Diagram

```
User Request
    │
    ├─→ [API Gateway]
    │      │
    │      ├─→ Authentication ✓
    │      ├─→ Rate Limiting ✓
    │      └─→ Input Validation ✓
    │
    ├─→ [AI Service]
    │      │
    │      ├─→ PII Detection & Redaction
    │      │      (Email, phone, SSN removed)
    │      │
    │      ├─→ Prompt Injection Check
    │      │      (Malicious patterns blocked)
    │      │
    │      ├─→ Rule Engine
    │      │      │
    │      │      ├─→ Can deterministic logic handle? ──YES──┐
    │      │      │                                           │
    │      │      NO                                          │
    │      │      │                                           │
    │      ├─→ Cache Check                                    │
    │      │      │                                           │
    │      │      ├─→ Cache Hit? ──YES──┐                   │
    │      │      │                      │                   │
    │      │      NO                     │                   │
    │      │      │                      │                   │
    │      ├─→ [LLM Router]             │                   │
    │      │      │                      │                   │
    │      │      ├─→ Select Model       │                   │
    │      │      │   (Haiku/Sonnet)     │                   │
    │      │      │                      │                   │
    │      │      ├─→ Primary Provider   │                   │
    │      │      │      │               │                   │
    │      │      │      ├─→ Success ────┤                   │
    │      │      │      │               │                   │
    │      │      │      └─→ Failure     │                   │
    │      │      │          │           │                   │
    │      │      └─→ Fallback Provider  │                   │
    │      │              │              │                   │
    │      │              └─→ Success ───┤                   │
    │      │                             │                   │
    │      ├─→ Response Validation       │                   │
    │      │      │                      │                   │
    │      │      ├─→ Data Leakage Check│                   │
    │      │      ├─→ Content Filter     │                   │
    │      │      └─→ Format Validation  │                   │
    │      │                             │                   │
    │      └─→ Cache Response            │                   │
    │             │                      │                   │
    │             └──────────────────────┴───────────────────┘
    │                                    │
    ├─→ [Observability]                  │
    │      │                             │
    │      ├─→ Log Request/Response      │
    │      ├─→ Record Metrics            │
    │      ├─→ Track Costs               │
    │      └─→ Check Alerts              │
    │                                    │
    └─→ Response to User ←───────────────┘
```

### 4.2 Data Flow Implementation

```python
class RequestPipeline:
    """Implement the complete request flow"""

    def __init__(self):
        self.validator = InputValidator()
        self.pii_detector = PIIDetector()
        self.rule_engine = RuleEngine()
        self.cache = CacheService()
        self.llm_router = LLMRouter()
        self.output_validator = OutputValidator()
        self.metrics = MetricsService()

    async def process_request(self, request: dict, user: User) -> dict:
        """Process request through the pipeline"""
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Stage 1: Input validation
            is_valid, error = self.validator.validate(request['prompt'])
            if not is_valid:
                return {'error': error, 'stage': 'validation'}

            # Stage 2: PII detection
            if self.pii_detector.should_reject(request['prompt']):
                return {'error': 'PII detected', 'stage': 'pii_check'}

            redacted_prompt = self.pii_detector.redact_pii(request['prompt'])

            # Stage 3: Rule engine (deterministic)
            rule_result = self.rule_engine.try_handle(redacted_prompt)
            if rule_result:
                self.metrics.record_deterministic_handling(request_id, user.id)
                return {
                    'text': rule_result,
                    'handled_by': 'rule_engine',
                    'cost': 0.0,
                }

            # Stage 4: Cache check
            cache_key = self._get_cache_key(redacted_prompt, request.get('model'))
            cached = await self.cache.get(cache_key)
            if cached:
                self.metrics.record_cache_hit(request_id, user.id)
                cached['cache_hit'] = True
                return cached

            # Stage 5: LLM processing
            llm_result = await self.llm_router.generate(
                prompt=redacted_prompt,
                model=request.get('model', 'haiku'),
                max_tokens=request.get('max_tokens', 1024),
            )

            # Stage 6: Output validation
            if self.output_validator.check_for_leakage(llm_result['text'], request):
                return {'error': 'Data leakage detected', 'stage': 'output_validation'}

            is_safe, error = self.output_validator.validate_output(llm_result['text'])
            if not is_safe:
                return {'error': error, 'stage': 'output_validation'}

            # Stage 7: Cache response
            await self.cache.set(cache_key, llm_result, ttl=3600)

            # Stage 8: Record metrics
            elapsed = time.time() - start_time
            self.metrics.record_request(
                request_id=request_id,
                user_id=user.id,
                latency_ms=elapsed * 1000,
                cost=llm_result.get('cost_usd', 0),
                input_tokens=llm_result.get('input_tokens', 0),
                output_tokens=llm_result.get('output_tokens', 0),
                model=llm_result.get('model'),
                cache_hit=False,
            )

            llm_result['cache_hit'] = False
            return llm_result

        except Exception as e:
            self.metrics.record_error(request_id, user.id, str(e))
            raise
```

---

## 5. Scalability Patterns

### 5.1 Horizontal Scaling Architecture

```
                     [Load Balancer]
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   [API Server 1]     [API Server 2]     [API Server 3]
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    [Redis Cluster]
                    (Shared Cache)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
  [Worker Pool 1]     [Worker Pool 2]     [Worker Pool 3]
  (LLM Processing)    (LLM Processing)    (LLM Processing)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    [Message Queue]
                    (RabbitMQ/Kafka)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
  [PostgreSQL Primary]  [Read Replica 1]  [Read Replica 2]
```

### 5.2 Async Processing Architecture

```python
# Architecture for async/background processing

class AsyncProcessingArchitecture:
    """
    Architecture for handling long-running LLM requests
    """

    def __init__(self):
        self.queue = MessageQueue()
        self.job_store = JobStore()

    # API Server: Accept request, return immediately
    async def submit_job(self, request: dict, user: User) -> dict:
        """Submit job and return immediately"""
        job_id = str(uuid.uuid4())

        # Store job metadata
        job = {
            'id': job_id,
            'user_id': user.id,
            'prompt': request['prompt'],
            'status': 'queued',
            'created_at': datetime.utcnow(),
        }
        await self.job_store.create(job)

        # Publish to queue
        await self.queue.publish('ai.jobs', job)

        return {
            'job_id': job_id,
            'status': 'queued',
            'status_url': f'/api/jobs/{job_id}',
        }

    # Worker: Process jobs from queue
    async def process_job(self, job: dict):
        """Background worker processes job"""
        try:
            # Update status
            await self.job_store.update(job['id'], {'status': 'processing'})

            # Process with LLM
            result = await llm_service.generate(job['prompt'])

            # Store result
            await self.job_store.update(job['id'], {
                'status': 'completed',
                'result': result,
                'completed_at': datetime.utcnow(),
            })

            # Trigger webhook if configured
            if job.get('webhook_url'):
                await self.trigger_webhook(job['webhook_url'], result)

        except Exception as e:
            await self.job_store.update(job['id'], {
                'status': 'failed',
                'error': str(e),
            })

    # API Server: Check job status
    async def get_job_status(self, job_id: str, user: User) -> dict:
        """Get job status"""
        job = await self.job_store.get(job_id)

        if job['user_id'] != user.id:
            raise PermissionError()

        return {
            'id': job['id'],
            'status': job['status'],
            'result': job.get('result'),
            'error': job.get('error'),
        }


# Usage
@app.post("/api/generate/async")
async def generate_async(request: GenerateRequest, user: User = Depends(get_current_user)):
    """Submit async generation job"""
    return await async_arch.submit_job(request.dict(), user)

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str, user: User = Depends(get_current_user)):
    """Get job status and result"""
    return await async_arch.get_job_status(job_id, user)
```

---

## 6. High Availability Architecture

### 6.1 HA Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       Region 1 (Primary)                    │
│  ┌────────────┐         ┌────────────┐                     │
│  │   Zone A   │         │   Zone B   │                     │
│  │            │         │            │                     │
│  │ API Server ├────────►│ API Server │                     │
│  │ (Primary)  │         │ (Standby)  │                     │
│  └─────┬──────┘         └─────┬──────┘                     │
│        │                      │                            │
│  ┌─────▼──────────────────────▼──────┐                     │
│  │     Redis Sentinel Cluster        │                     │
│  │   (Master + 2 Replicas)           │                     │
│  └─────┬──────────────────────┬──────┘                     │
│        │                      │                            │
│  ┌─────▼──────┐         ┌─────▼──────┐                     │
│  │ PostgreSQL │◄───────►│ PostgreSQL │                     │
│  │  Primary   │         │  Replica   │                     │
│  └────────────┘         └────────────┘                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ Cross-region replication
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Region 2 (DR/Failover)                    │
│  ┌────────────┐         ┌────────────┐                     │
│  │   Zone C   │         │   Zone D   │                     │
│  │            │         │            │                     │
│  │ API Server │         │ API Server │                     │
│  │ (Standby)  │         │ (Standby)  │                     │
│  └────────────┘         └────────────┘                     │
│                                                             │
│  ┌──────────────────────────────────┐                      │
│  │    PostgreSQL Read Replica       │                      │
│  └──────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Failover Implementation

```python
class HighAvailabilityManager:
    """Manage HA and failover"""

    def __init__(self):
        self.primary_provider = AnthropicProvider()
        self.fallback_provider = OpenAIProvider()
        self.circuit_breaker = CircuitBreaker()

    async def generate_with_ha(self, prompt: str) -> dict:
        """Generate with automatic failover"""
        # Try primary provider
        if self.circuit_breaker.is_available('primary'):
            try:
                result = await self.primary_provider.generate(prompt)
                self.circuit_breaker.record_success('primary')
                return result
            except Exception as e:
                logger.error(f"Primary provider failed: {e}")
                self.circuit_breaker.record_failure('primary')

        # Fallback to secondary
        if self.circuit_breaker.is_available('fallback'):
            try:
                result = await self.fallback_provider.generate(prompt)
                self.circuit_breaker.record_success('fallback')
                return result
            except Exception as e:
                logger.error(f"Fallback provider failed: {e}")
                self.circuit_breaker.record_failure('fallback')

        raise ServiceUnavailableError("All providers unavailable")


class CircuitBreaker:
    """Circuit breaker pattern for resilience"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = {}
        self.last_failure_time = {}
        self.state = {}  # 'closed', 'open', 'half_open'

    def is_available(self, provider: str) -> bool:
        """Check if provider is available"""
        if provider not in self.state:
            self.state[provider] = 'closed'
            self.failure_count[provider] = 0
            return True

        # If circuit is open, check if timeout has passed
        if self.state[provider] == 'open':
            if time.time() - self.last_failure_time[provider] > self.timeout:
                self.state[provider] = 'half_open'
                return True
            return False

        return True

    def record_success(self, provider: str):
        """Record successful call"""
        self.failure_count[provider] = 0
        if self.state[provider] == 'half_open':
            self.state[provider] = 'closed'

    def record_failure(self, provider: str):
        """Record failed call"""
        self.failure_count[provider] = self.failure_count.get(provider, 0) + 1
        self.last_failure_time[provider] = time.time()

        if self.failure_count[provider] >= self.failure_threshold:
            self.state[provider] = 'open'
            logger.warning(f"Circuit breaker opened for {provider}")
```

---

## 7. Multi-Tenant Architecture

### 7.1 Tenant Isolation Patterns

```python
class MultiTenantArchitecture:
    """Architecture for multi-tenant SaaS"""

    # Pattern 1: Database per Tenant
    class DatabasePerTenant:
        """Each tenant gets own database"""

        def get_connection(self, tenant_id: str):
            """Get tenant-specific database connection"""
            db_name = f"tenant_{tenant_id}"
            return psycopg2.connect(
                host="postgres.example.com",
                database=db_name,
                user="app_user",
                password=os.getenv("DB_PASSWORD")
            )

    # Pattern 2: Schema per Tenant
    class SchemaPerTenant:
        """Shared database, separate schemas"""

        def get_connection(self, tenant_id: str):
            """Get connection with tenant schema"""
            conn = psycopg2.connect(
                host="postgres.example.com",
                database="saas_app",
                user="app_user",
                password=os.getenv("DB_PASSWORD")
            )

            # Set search path to tenant schema
            with conn.cursor() as cursor:
                cursor.execute(f"SET search_path TO tenant_{tenant_id}")

            return conn

    # Pattern 3: Shared Tables with Tenant ID
    class SharedTablesWithTenantID:
        """Most common: Shared tables, tenant_id column"""

        def query_for_tenant(self, tenant_id: str, query: str, params: tuple):
            """Always filter by tenant_id"""
            # Add tenant_id to all queries
            modified_query = f"{query} WHERE tenant_id = %s"
            return self.db.execute(modified_query, (*params, tenant_id))

    # Tenant Context Manager
    @contextmanager
    def tenant_context(self, tenant_id: str):
        """Context manager for tenant-specific operations"""
        # Set tenant in thread-local storage
        _tenant_local.tenant_id = tenant_id
        try:
            yield tenant_id
        finally:
            _tenant_local.tenant_id = None

# Usage
@app.post("/api/generate")
async def generate(request: GenerateRequest, tenant: Tenant = Depends(get_tenant)):
    """All operations scoped to tenant"""
    with multi_tenant.tenant_context(tenant.id):
        result = await ai_service.generate(request.prompt)
        return result
```

---

## 8. Integration Patterns Summary

### 8.1 Pattern Comparison

| Pattern | Use Case | Pros | Cons |
|---------|----------|------|------|
| **Monolithic** | Small apps, MVPs | Simple, easy deployment | Hard to scale, tight coupling |
| **Layered** | Standard web apps | Clear separation, testable | Can become rigid |
| **Microservices** | Large, complex systems | Independent scaling, tech flexibility | Operational complexity |
| **Event-Driven** | Real-time, async workflows | Loose coupling, scalable | Harder to debug, eventual consistency |
| **Serverless** | Variable load, cost-sensitive | Auto-scaling, pay-per-use | Cold starts, vendor lock-in |

### 8.2 Architecture Decision Matrix

```python
class ArchitectureDecisionMatrix:
    """Help choose the right architecture"""

    def recommend_architecture(self, requirements: dict) -> str:
        """Recommend based on requirements"""

        # Small team, simple requirements
        if requirements['team_size'] < 5 and requirements['complexity'] == 'low':
            return 'Monolithic with Layered Architecture'

        # Medium scale, moderate complexity
        if requirements['requests_per_second'] < 1000:
            return 'Layered Architecture with Caching'

        # High scale, need independent scaling
        if requirements['requests_per_second'] > 10000:
            return 'Microservices with Message Queue'

        # Real-time, event processing
        if requirements['real_time'] and requirements['async_processing']:
            return 'Event-Driven with Message Queue'

        # Variable load, cost optimization
        if requirements['variable_load'] and requirements['optimize_cost']:
            return 'Serverless with API Gateway'

        return 'Layered Architecture'  # Default safe choice
```

---

## 9. Reference Implementation

### 9.1 Complete Architecture Example

```python
# Complete implementation of layered + microservices hybrid

# app.py - Main application
from fastapi import FastAPI, Depends
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle"""
    # Startup
    await initialize_services()
    yield
    # Shutdown
    await cleanup_services()

app = FastAPI(lifespan=lifespan)

# Register blueprints/routers
app.include_router(api_v1_router, prefix="/api/v1")
app.include_router(health_router)


# services/ai_service.py - AI Service Layer
class AIService:
    """Main AI service orchestrator"""

    def __init__(
        self,
        llm_integration: LLMIntegration,
        cache_service: CacheService,
        metrics_service: MetricsService,
        rule_engine: RuleEngine,
    ):
        self.llm = llm_integration
        self.cache = cache_service
        self.metrics = metrics_service
        self.rules = rule_engine

    async def generate(
        self,
        prompt: str,
        model: str = "haiku",
        user_id: str = None,
    ) -> dict:
        """Main generation method"""
        # Try rule engine first
        rule_result = self.rules.try_handle(prompt)
        if rule_result:
            return {'text': rule_result, 'source': 'rules', 'cost': 0}

        # Check cache
        cache_key = self._cache_key(prompt, model)
        cached = await self.cache.get(cache_key)
        if cached:
            return {**cached, 'cache_hit': True}

        # Call LLM
        result = await self.llm.generate(prompt, model)

        # Cache and return
        await self.cache.set(cache_key, result)
        self.metrics.record(result, user_id)

        return {**result, 'cache_hit': False}


# integrations/llm_integration.py - Integration Layer
class LLMIntegration:
    """Abstraction over multiple LLM providers"""

    def __init__(self):
        self.providers = {
            'anthropic': AnthropicAdapter(),
            'openai': OpenAIAdapter(),
        }
        self.circuit_breaker = CircuitBreaker()

    async def generate(self, prompt: str, model: str) -> dict:
        """Generate with fallback logic"""
        primary = self._select_provider(model)
        fallback = 'openai' if primary == 'anthropic' else 'anthropic'

        # Try primary
        try:
            return await self.providers[primary].generate(prompt, model)
        except Exception as e:
            logger.warning(f"Primary failed: {e}, using fallback")
            return await self.providers[fallback].generate(prompt, model)


# adapters/anthropic_adapter.py - Adapter Pattern
class AnthropicAdapter:
    """Adapter for Anthropic API"""

    def __init__(self):
        self.client = anthropic.Client(api_key=os.getenv('ANTHROPIC_API_KEY'))

    async def generate(self, prompt: str, model: str) -> dict:
        """Standardized interface"""
        response = self.client.messages.create(
            model=f"claude-3-{model}-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            'text': response.content[0].text,
            'model': model,
            'provider': 'anthropic',
            'input_tokens': response.usage.input_tokens,
            'output_tokens': response.usage.output_tokens,
        }
```

---

## 10. AI Gateway Architecture

### 10.1 AI Gateway Pattern

An AI Gateway is a unified API layer that sits between your application and multiple LLM providers, providing abstraction, routing, fallback, rate limiting, and observability.

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Applications                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │   Web    │  │  Mobile  │  │  Backend │                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
└───────┼─────────────┼─────────────┼─────────────────────────┘
        │             │             │
        └─────────────┴─────────────┘
                      │
        ┌─────────────▼─────────────────────────────────────┐
        │              AI GATEWAY                           │
        │                                                   │
        │  ┌────────────────────────────────────────────┐  │
        │  │   Unified API Interface                    │  │
        │  │   POST /v1/chat/completions                │  │
        │  │   POST /v1/completions                     │  │
        │  │   POST /v1/embeddings                      │  │
        │  └────────────────────────────────────────────┘  │
        │                                                   │
        │  ┌────────────────────────────────────────────┐  │
        │  │   Core Gateway Features                    │  │
        │  │                                            │  │
        │  │  • Provider Routing                        │  │
        │  │  • Load Balancing                          │  │
        │  │  • Failover & Retry                        │  │
        │  │  • Rate Limiting                           │  │
        │  │  • Caching                                 │  │
        │  │  • Cost Tracking                           │  │
        │  │  • Token Counting                          │  │
        │  │  • Request/Response Logging                │  │
        │  └────────────────────────────────────────────┘  │
        │                                                   │
        │  ┌────────────────────────────────────────────┐  │
        │  │   Provider Adapters                        │  │
        │  │   (Convert unified format to provider      │  │
        │  │    specific format)                        │  │
        │  └────────────────────────────────────────────┘  │
        └───────────┬───────────┬───────────┬───────────────┘
                    │           │           │
        ┌───────────▼─┐   ┌─────▼─────┐   ┌▼──────────┐
        │  Anthropic  │   │  OpenAI   │   │   Azure   │
        │   Claude    │   │  GPT-4    │   │  OpenAI   │
        └─────────────┘   └───────────┘   └───────────┘
```

### 10.2 AI Gateway Implementation

```python
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
import time

class AIGateway:
    """
    Complete AI Gateway implementation
    Provides unified interface to multiple LLM providers
    """

    def __init__(self):
        self.providers = self._init_providers()
        self.router = ProviderRouter()
        self.cache = GatewayCache()
        self.rate_limiter = RateLimiter()
        self.cost_tracker = CostTracker()
        self.circuit_breakers = {}

    def _init_providers(self) -> Dict:
        """Initialize all LLM provider clients"""
        return {
            'anthropic': AnthropicProvider(
                api_key=os.getenv('ANTHROPIC_API_KEY'),
                models=['claude-3-haiku', 'claude-3-5-sonnet', 'claude-3-opus']
            ),
            'openai': OpenAIProvider(
                api_key=os.getenv('OPENAI_API_KEY'),
                models=['gpt-3.5-turbo', 'gpt-4-turbo', 'gpt-4']
            ),
            'azure': AzureOpenAIProvider(
                api_key=os.getenv('AZURE_OPENAI_KEY'),
                endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                models=['gpt-4']
            ),
        }

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
        user_id: str,
        api_key: str,
    ) -> ChatCompletionResponse:
        """
        Unified chat completion endpoint
        Compatible with OpenAI API format
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # 1. Rate limiting
            await self.rate_limiter.check(user_id, api_key)

            # 2. Check cache
            cache_key = self._get_cache_key(request)
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for request {request_id}")
                return cached

            # 3. Route to appropriate provider
            provider_name, model = self.router.select_provider(
                requested_model=request.model,
                user_preferences=self._get_user_preferences(user_id)
            )

            # 4. Check circuit breaker
            if not self._is_provider_available(provider_name):
                provider_name = self.router.get_fallback_provider(provider_name)

            # 5. Call provider
            provider = self.providers[provider_name]
            response = await provider.chat_completion(
                messages=request.messages,
                model=model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            # 6. Track cost
            cost = self.cost_tracker.calculate_cost(
                provider=provider_name,
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            await self.cost_tracker.record(user_id, cost, request_id)

            # 7. Cache response
            await self.cache.set(cache_key, response, ttl=3600)

            # 8. Log metrics
            elapsed = time.time() - start_time
            await self._log_request(
                request_id=request_id,
                user_id=user_id,
                provider=provider_name,
                model=model,
                latency=elapsed,
                cost=cost,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

            return response

        except Exception as e:
            # Record circuit breaker failure
            self._record_provider_failure(provider_name)

            # Try fallback provider
            fallback_provider = self.router.get_fallback_provider(provider_name)
            if fallback_provider:
                logger.warning(f"Falling back to {fallback_provider}")
                return await self._call_fallback(fallback_provider, request)

            raise HTTPException(status_code=503, detail=str(e))


class ProviderRouter:
    """Route requests to appropriate LLM provider"""

    def __init__(self):
        # Model to provider mapping
        self.model_mapping = {
            'claude-3-haiku': 'anthropic',
            'claude-3-5-sonnet': 'anthropic',
            'claude-3-opus': 'anthropic',
            'gpt-3.5-turbo': 'openai',
            'gpt-4-turbo': 'openai',
            'gpt-4': 'openai',
        }

        # Fallback chains
        self.fallback_chains = {
            'anthropic': ['openai', 'azure'],
            'openai': ['anthropic', 'azure'],
            'azure': ['openai', 'anthropic'],
        }

    def select_provider(
        self,
        requested_model: str,
        user_preferences: dict = None,
    ) -> tuple[str, str]:
        """
        Select provider and model based on:
        - Requested model
        - User preferences (cost, speed, quality)
        - Provider availability
        """
        # Direct model mapping
        if requested_model in self.model_mapping:
            provider = self.model_mapping[requested_model]
            return provider, requested_model

        # Smart routing based on preferences
        if user_preferences:
            if user_preferences.get('optimize') == 'cost':
                return 'anthropic', 'claude-3-haiku'
            elif user_preferences.get('optimize') == 'speed':
                return 'anthropic', 'claude-3-haiku'
            elif user_preferences.get('optimize') == 'quality':
                return 'anthropic', 'claude-3-opus'

        # Default
        return 'anthropic', 'claude-3-haiku'

    def get_fallback_provider(self, failed_provider: str) -> Optional[str]:
        """Get next provider in fallback chain"""
        chain = self.fallback_chains.get(failed_provider, [])
        return chain[0] if chain else None


class GatewayCache:
    """Caching layer for AI Gateway"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
        self.default_ttl = 3600  # 1 hour

    async def get(self, key: str) -> Optional[dict]:
        """Get cached response"""
        cached = self.redis.get(f"gateway:cache:{key}")
        if cached:
            return json.loads(cached)
        return None

    async def set(self, key: str, value: dict, ttl: int = None):
        """Cache response"""
        self.redis.setex(
            f"gateway:cache:{key}",
            ttl or self.default_ttl,
            json.dumps(value)
        )


class CostTracker:
    """Track costs across providers"""

    PRICING = {
        'anthropic': {
            'claude-3-haiku': {'input': 0.25, 'output': 1.25},
            'claude-3-5-sonnet': {'input': 3.00, 'output': 15.00},
            'claude-3-opus': {'input': 15.00, 'output': 75.00},
        },
        'openai': {
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-4': {'input': 30.00, 'output': 60.00},
        },
    }

    def calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Calculate cost in USD"""
        pricing = self.PRICING.get(provider, {}).get(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']

        return input_cost + output_cost

    async def record(self, user_id: str, cost: float, request_id: str):
        """Record cost for user"""
        await self.db.insert('cost_tracking', {
            'user_id': user_id,
            'request_id': request_id,
            'cost_usd': cost,
            'timestamp': datetime.utcnow(),
        })


# FastAPI Application
app = FastAPI(title="AI Gateway")
gateway = AIGateway()

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Header(..., alias="Authorization"),
    user: User = Depends(get_current_user),
):
    """
    OpenAI-compatible chat completions endpoint
    Works with any OpenAI SDK
    """
    return await gateway.chat_completion(request, user.id, api_key)

@app.post("/v1/completions")
async def completions(
    request: CompletionRequest,
    api_key: str = Header(..., alias="Authorization"),
    user: User = Depends(get_current_user),
):
    """Legacy completions endpoint"""
    return await gateway.completion(request, user.id, api_key)

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        'data': [
            {'id': 'claude-3-haiku', 'provider': 'anthropic'},
            {'id': 'claude-3-5-sonnet', 'provider': 'anthropic'},
            {'id': 'gpt-3.5-turbo', 'provider': 'openai'},
            {'id': 'gpt-4-turbo', 'provider': 'openai'},
        ]
    }
```

### 10.3 Using AI Gateway with OpenAI SDK

```python
# Your application code - works with any OpenAI-compatible SDK
import openai

# Point OpenAI SDK to your AI Gateway
openai.api_base = "https://your-ai-gateway.com/v1"
openai.api_key = "your-gateway-api-key"

# Now all OpenAI calls go through your gateway
response = openai.ChatCompletion.create(
    model="claude-3-haiku",  # Or any model your gateway supports
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

# Gateway automatically:
# - Routes to Anthropic
# - Handles rate limiting
# - Caches responses
# - Tracks costs
# - Provides fallback to OpenAI if Anthropic fails
```

### 10.4 AI Gateway Benefits

**For Developers:**
- ✅ Single API for all LLM providers
- ✅ Drop-in replacement for OpenAI API
- ✅ No vendor lock-in
- ✅ Easy provider switching
- ✅ Built-in retry and fallback

**For Operations:**
- ✅ Centralized rate limiting
- ✅ Cost tracking per user/project
- ✅ Request/response logging
- ✅ Circuit breaker protection
- ✅ Provider health monitoring

**For Business:**
- ✅ Lower costs (smart routing)
- ✅ Higher availability (multi-provider)
- ✅ Better observability
- ✅ Compliance enforcement
- ✅ Budget controls

### 10.5 Open Source AI Gateway Solutions

```yaml
LiteLLM:
  url: https://github.com/BerriAI/litellm
  features:
    - 100+ LLM providers support
    - OpenAI-compatible API
    - Load balancing
    - Fallback logic
    - Cost tracking
  deployment: Python package or Docker

Portkey:
  url: https://portkey.ai
  features:
    - Multi-provider gateway
    - Caching layer
    - Observability
    - Prompt management
    - Cost controls
  deployment: SaaS or self-hosted

Kong AI Gateway:
  url: https://konghq.com/products/kong-gateway
  features:
    - Enterprise API Gateway with AI plugins
    - Rate limiting
    - Authentication
    - Observability
  deployment: Self-hosted or Kong Cloud

Custom Gateway:
  features:
    - Full control
    - Custom routing logic
    - Company-specific features
  deployment: Self-hosted
  effort: High (build from scratch)
```

### 10.6 AI Gateway Architecture Decisions

```python
class GatewayArchitectureDecisions:
    """Key decisions when building AI Gateway"""

    DECISIONS = {
        'api_compatibility': {
            'options': ['OpenAI-compatible', 'Custom API', 'Both'],
            'recommendation': 'OpenAI-compatible',
            'reason': 'Works with existing SDKs and tools',
        },

        'routing_strategy': {
            'options': ['Static', 'Cost-based', 'Load-based', 'Smart'],
            'recommendation': 'Smart (cost + availability + latency)',
            'reason': 'Optimize across multiple dimensions',
        },

        'caching_layer': {
            'options': ['None', 'In-memory', 'Redis', 'CDN'],
            'recommendation': 'Redis',
            'reason': 'Shared across instances, persistent',
        },

        'deployment': {
            'options': ['Monolith', 'Microservice', 'Serverless'],
            'recommendation': 'Depends on scale',
            'reason': {
                'small': 'Monolith (simple)',
                'medium': 'Microservice (scalable)',
                'large': 'Multiple regions, edge deployment',
            },
        },

        'state_management': {
            'options': ['Stateless', 'Stateful'],
            'recommendation': 'Stateless',
            'reason': 'Easier to scale horizontally',
        },

        'observability': {
            'required': [
                'Request/response logging',
                'Cost per user/project',
                'Latency per provider',
                'Error rates',
                'Cache hit rates',
            ],
        },
    }
```

---

## 11. Architecture Checklist

### Pre-Implementation Checklist

```yaml
Architecture Design:
  - [ ] Architecture pattern selected (monolithic/layered/microservices)
  - [ ] Component boundaries defined
  - [ ] Service communication patterns chosen
  - [ ] Data flow documented
  - [ ] Scalability strategy planned

Integration Points:
  - [ ] LLM providers identified and prioritized
  - [ ] Fallback strategy defined
  - [ ] API contracts documented
  - [ ] Error handling strategy defined
  - [ ] Timeout values configured

Data Management:
  - [ ] Database architecture chosen
  - [ ] Caching strategy defined
  - [ ] Data retention policies set
  - [ ] Backup and recovery planned

Scalability:
  - [ ] Load balancing strategy
  - [ ] Horizontal scaling plan
  - [ ] Auto-scaling configured
  - [ ] Resource limits defined

High Availability:
  - [ ] Redundancy for critical components
  - [ ] Failover procedures documented
  - [ ] Circuit breaker implemented
  - [ ] Health checks configured

Security:
  - [ ] Authentication architecture
  - [ ] Authorization model defined
  - [ ] API key management strategy
  - [ ] Data encryption plan

Observability:
  - [ ] Logging strategy defined
  - [ ] Metrics collection configured
  - [ ] Distributed tracing set up
  - [ ] Alerting rules defined

Multi-tenancy (if applicable):
  - [ ] Tenant isolation strategy
  - [ ] Data segregation approach
  - [ ] Resource allocation model
  - [ ] Billing integration
```

---

**Version:** 1.0
**Last Updated:** February 9, 2026
**Status:** Active
