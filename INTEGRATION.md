# AI Integration Guide

## Overview

This guide covers practical patterns and best practices for integrating AI capabilities (LLMs, embeddings, etc.) into applications, including API design, SDK usage, error handling, and production deployment strategies.

---

## 1. Integration Patterns

### 1.1 Direct API Integration

```python
import anthropic
import openai
from typing import Optional

class LLMClient:
    """Direct API integration with fallback"""

    def __init__(self):
        self.anthropic_client = anthropic.Client(
            api_key=os.getenv('ANTHROPIC_API_KEY')
        )
        self.openai_client = openai.Client(
            api_key=os.getenv('OPENAI_API_KEY')
        )

    def generate_text(self, prompt: str, model: str = 'haiku') -> dict:
        """Generate text with primary and fallback providers"""
        try:
            # Try primary provider (Anthropic)
            if model in ['haiku', 'sonnet', 'opus']:
                return self._call_anthropic(prompt, model)
        except Exception as e:
            logger.warning(f"Anthropic API failed: {e}, falling back to OpenAI")

        # Fallback to OpenAI
        try:
            return self._call_openai(prompt)
        except Exception as e:
            logger.error(f"All LLM providers failed: {e}")
            raise

    def _call_anthropic(self, prompt: str, model: str) -> dict:
        """Call Anthropic API"""
        model_map = {
            'haiku': 'claude-3-haiku-20240307',
            'sonnet': 'claude-3-5-sonnet-20241022',
            'opus': 'claude-3-opus-20240229',
        }

        response = self.anthropic_client.messages.create(
            model=model_map[model],
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

    def _call_openai(self, prompt: str) -> dict:
        """Call OpenAI API"""
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )

        return {
            'text': response.choices[0].message.content,
            'model': 'gpt-3.5-turbo',
            'provider': 'openai',
            'input_tokens': response.usage.prompt_tokens,
            'output_tokens': response.usage.completion_tokens,
        }
```

### 1.2 Abstraction Layer

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base for LLM providers"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def embed(self, text: str) -> list:
        """Generate embeddings"""
        pass

class AnthropicProvider(LLMProvider):
    """Anthropic implementation"""

    def __init__(self, api_key: str):
        self.client = anthropic.Client(api_key=api_key)

    def generate(self, prompt: str, model: str = 'haiku', **kwargs) -> dict:
        response = self.client.messages.create(
            model=f'claude-3-{model}-20240307',
            max_tokens=kwargs.get('max_tokens', 1024),
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            'text': response.content[0].text,
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens,
            }
        }

    def embed(self, text: str) -> list:
        # Anthropic doesn't have embeddings API
        raise NotImplementedError()

class OpenAIProvider(LLMProvider):
    """OpenAI implementation"""

    def __init__(self, api_key: str):
        self.client = openai.Client(api_key=api_key)

    def generate(self, prompt: str, model: str = 'gpt-3.5-turbo', **kwargs) -> dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get('max_tokens', 1024),
        )

        return {
            'text': response.choices[0].message.content,
            'usage': {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
            }
        }

    def embed(self, text: str) -> list:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

class LLMService:
    """High-level service with provider abstraction"""

    def __init__(self, primary: LLMProvider, fallback: Optional[LLMProvider] = None):
        self.primary = primary
        self.fallback = fallback

    def generate(self, prompt: str, **kwargs) -> dict:
        """Generate with automatic fallback"""
        try:
            return self.primary.generate(prompt, **kwargs)
        except Exception as e:
            if self.fallback:
                logger.warning(f"Primary provider failed: {e}, using fallback")
                return self.fallback.generate(prompt, **kwargs)
            raise

# Usage
service = LLMService(
    primary=AnthropicProvider(api_key=ANTHROPIC_KEY),
    fallback=OpenAIProvider(api_key=OPENAI_KEY)
)

result = service.generate("Explain quantum computing")
```

---

## 2. SDK Integration Patterns

### 2.1 LangChain Integration

```python
from langchain.llms import OpenAI, Anthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

class LangChainIntegration:
    """LangChain-based integration"""

    def __init__(self):
        self.llm = Anthropic(
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            model='claude-3-haiku-20240307'
        )

        self.memory = ConversationBufferMemory()

    def create_chain(self, template: str) -> LLMChain:
        """Create a chain with prompt template"""
        prompt = PromptTemplate(
            input_variables=["input"],
            template=template
        )

        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            memory=self.memory
        )

        return chain

    def chat(self, message: str) -> dict:
        """Chat with memory"""
        chain = self.create_chain(
            template="You are a helpful assistant.\\n\\n{input}"
        )

        # Track costs
        with get_openai_callback() as cb:
            response = chain.run(input=message)

            return {
                'response': response,
                'tokens': cb.total_tokens,
                'cost': cb.total_cost,
            }

# Usage
integration = LangChainIntegration()
result = integration.chat("What is machine learning?")
```

### 2.2 LlamaIndex Integration (RAG)

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import Anthropic

class RAGIntegration:
    """Retrieval-Augmented Generation with LlamaIndex"""

    def __init__(self, documents_path: str):
        # Load documents
        self.documents = SimpleDirectoryReader(documents_path).load_data()

        # Configure LLM
        self.llm = Anthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            model='claude-3-haiku-20240307'
        )

        # Create service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            chunk_size=512,
        )

        # Build index
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            service_context=self.service_context
        )

    def query(self, question: str) -> dict:
        """Query with RAG"""
        query_engine = self.index.as_query_engine(
            similarity_top_k=3  # Retrieve top 3 relevant chunks
        )

        response = query_engine.query(question)

        return {
            'answer': str(response),
            'source_nodes': [
                {
                    'text': node.node.get_text()[:200],
                    'score': node.score,
                }
                for node in response.source_nodes
            ],
        }

# Usage
rag = RAGIntegration('./knowledge_base/')
result = rag.query("What is our refund policy?")
```

---

## 3. API Design Best Practices

### 3.1 RESTful API Endpoints

```python
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

def require_api_key(f):
    """Decorator for API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if not api_key or not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401

        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/v1/generate', methods=['POST'])
@require_api_key
def generate_text():
    """
    Generate text from prompt

    Request:
    {
        "prompt": "string",
        "model": "haiku|sonnet|opus",
        "max_tokens": 1024,
        "temperature": 0.7
    }

    Response:
    {
        "id": "req_123",
        "text": "generated text",
        "model": "haiku",
        "usage": {
            "input_tokens": 150,
            "output_tokens": 200,
            "total_tokens": 350
        },
        "cost_usd": 0.0008
    }
    """
    try:
        data = request.get_json()

        # Validate input
        if not data.get('prompt'):
            return jsonify({'error': 'Prompt is required'}), 400

        # Rate limiting
        user_id = get_user_from_api_key(request.headers.get('X-API-Key'))
        if not check_rate_limit(user_id):
            return jsonify({'error': 'Rate limit exceeded'}), 429

        # Generate
        result = llm_service.generate(
            prompt=data['prompt'],
            model=data.get('model', 'haiku'),
            max_tokens=data.get('max_tokens', 1024),
            temperature=data.get('temperature', 0.7),
        )

        # Calculate cost
        cost = calculate_cost(
            result['usage']['input_tokens'],
            result['usage']['output_tokens'],
            result['model']
        )

        response = {
            'id': generate_request_id(),
            'text': result['text'],
            'model': result['model'],
            'usage': result['usage'],
            'cost_usd': cost,
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/v1/chat', methods=['POST'])
@require_api_key
def chat():
    """
    Chat with conversation history

    Request:
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ],
        "model": "haiku"
    }

    Response:
    {
        "message": {
            "role": "assistant",
            "content": "generated response"
        },
        "usage": {...}
    }
    """
    try:
        data = request.get_json()
        messages = data.get('messages', [])

        if not messages:
            return jsonify({'error': 'Messages are required'}), 400

        # Limit history
        messages = messages[-10:]  # Only last 10 messages

        result = llm_service.chat(
            messages=messages,
            model=data.get('model', 'haiku')
        )

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500
```

### 3.2 Streaming Responses

```python
from flask import Response, stream_with_context

@app.route('/api/v1/generate/stream', methods=['POST'])
@require_api_key
def generate_stream():
    """
    Stream generated text (Server-Sent Events)

    Response format (SSE):
    data: {"type": "token", "content": "Hello"}
    data: {"type": "token", "content": " world"}
    data: {"type": "done", "usage": {...}}
    """
    def generate():
        try:
            data = request.get_json()
            prompt = data.get('prompt')

            # Stream from LLM
            with client.messages.stream(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'type': 'token', 'content': text})}\\n\\n"

                # Send completion
                usage = stream.get_final_message().usage
                yield f"data: {json.dumps({'type': 'done', 'usage': usage})}\\n\\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\\n\\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )
```

---

## 4. Error Handling and Retries

### 4.1 Retry Logic

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class LLMClientWithRetry:
    """LLM client with automatic retries"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
        ))
    )
    def generate_with_retry(self, prompt: str) -> dict:
        """Generate with automatic exponential backoff retry"""
        return self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

    def generate_with_fallback(self, prompt: str) -> dict:
        """Generate with manual retry and fallback"""
        max_retries = 3
        providers = [
            ('anthropic', self.call_anthropic),
            ('openai', self.call_openai),
        ]

        for provider_name, provider_func in providers:
            for attempt in range(max_retries):
                try:
                    result = provider_func(prompt)
                    logger.info(f"Success with {provider_name} on attempt {attempt + 1}")
                    return result

                except Exception as e:
                    logger.warning(f"{provider_name} attempt {attempt + 1} failed: {e}")

                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All retries failed for {provider_name}")

        raise Exception("All providers failed after retries")
```

### 4.2 Error Response Standards

```python
class ErrorResponse:
    """Standardized error responses"""

    @staticmethod
    def validation_error(message: str, field: str = None) -> tuple:
        """400 Bad Request"""
        return jsonify({
            'error': {
                'type': 'validation_error',
                'message': message,
                'field': field,
            }
        }), 400

    @staticmethod
    def authentication_error() -> tuple:
        """401 Unauthorized"""
        return jsonify({
            'error': {
                'type': 'authentication_error',
                'message': 'Invalid or missing API key',
            }
        }), 401

    @staticmethod
    def authorization_error() -> tuple:
        """403 Forbidden"""
        return jsonify({
            'error': {
                'type': 'authorization_error',
                'message': 'Insufficient permissions',
            }
        }), 403

    @staticmethod
    def rate_limit_error(retry_after: int = None) -> tuple:
        """429 Too Many Requests"""
        response = jsonify({
            'error': {
                'type': 'rate_limit_error',
                'message': 'Rate limit exceeded',
                'retry_after': retry_after,
            }
        })

        if retry_after:
            response.headers['Retry-After'] = str(retry_after)

        return response, 429

    @staticmethod
    def server_error(request_id: str = None) -> tuple:
        """500 Internal Server Error"""
        return jsonify({
            'error': {
                'type': 'server_error',
                'message': 'An internal error occurred',
                'request_id': request_id,
            }
        }), 500

    @staticmethod
    def service_unavailable(estimated_time: int = None) -> tuple:
        """503 Service Unavailable"""
        return jsonify({
            'error': {
                'type': 'service_unavailable',
                'message': 'Service temporarily unavailable',
                'estimated_time_seconds': estimated_time,
            }
        }), 503
```

---

## 5. Webhook Integration

### 5.1 Webhook Handler

```python
import hmac
import hashlib

class WebhookHandler:
    """Handle incoming webhooks from LLM providers"""

    def __init__(self, secret: str):
        self.secret = secret

    def verify_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature"""
        expected_signature = hmac.new(
            self.secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(signature, expected_signature)

    def handle_webhook(self, event_type: str, data: dict):
        """Route webhook events"""
        handlers = {
            'generation.completed': self.handle_generation_completed,
            'generation.failed': self.handle_generation_failed,
            'usage.threshold_reached': self.handle_usage_alert,
        }

        handler = handlers.get(event_type)
        if handler:
            handler(data)
        else:
            logger.warning(f"Unknown webhook event: {event_type}")

    def handle_generation_completed(self, data: dict):
        """Handle completed generation"""
        request_id = data['request_id']
        result = data['result']

        # Update database
        self.db.update_request(request_id, {
            'status': 'completed',
            'result': result,
            'completed_at': datetime.utcnow(),
        })

        # Notify user
        self.notify_user(data['user_id'], result)

@app.route('/webhooks/llm-provider', methods=['POST'])
def webhook_endpoint():
    """Receive webhooks from LLM provider"""
    # Verify signature
    signature = request.headers.get('X-Webhook-Signature')
    if not webhook_handler.verify_signature(request.data, signature):
        return jsonify({'error': 'Invalid signature'}), 401

    # Process event
    event = request.get_json()
    webhook_handler.handle_webhook(
        event_type=event['type'],
        data=event['data']
    )

    return jsonify({'received': True}), 200
```

---

## 6. Batch Processing

### 6.1 Batch API Integration

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

class BatchProcessor:
    """Process multiple requests efficiently"""

    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_batch(self, prompts: List[str]) -> List[dict]:
        """Process multiple prompts in parallel"""
        futures = {
            self.executor.submit(self.process_single, prompt): prompt
            for prompt in prompts
        }

        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                prompt = futures[future]
                logger.error(f"Failed to process prompt: {prompt[:50]}... Error: {e}")
                results.append({
                    'error': str(e),
                    'prompt': prompt[:50],
                })

        return results

    def process_single(self, prompt: str) -> dict:
        """Process a single prompt"""
        return llm_service.generate(prompt)

    def process_with_rate_limiting(self, prompts: List[str],
                                  requests_per_second: int = 10) -> List[dict]:
        """Process batch with rate limiting"""
        delay = 1.0 / requests_per_second
        results = []

        for prompt in prompts:
            result = self.process_single(prompt)
            results.append(result)
            time.sleep(delay)

        return results

# Usage
processor = BatchProcessor(max_workers=5)
prompts = ["Summarize: " + doc for doc in documents]
results = processor.process_batch(prompts)
```

---

## 7. Caching Integration

### 7.1 Redis Cache

```python
import redis
import json
import hashlib

class LLMCache:
    """Cache LLM responses in Redis"""

    def __init__(self, redis_url: str, ttl: int = 3600):
        self.redis = redis.from_url(redis_url)
        self.ttl = ttl

    def get_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key from request parameters"""
        cache_data = {
            'prompt': prompt,
            'model': model,
            **kwargs
        }

        key_str = json.dumps(cache_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()

        return f"llm:cache:{key_hash}"

    def get(self, prompt: str, model: str, **kwargs) -> Optional[dict]:
        """Get cached response"""
        key = self.get_cache_key(prompt, model, **kwargs)
        cached = self.redis.get(key)

        if cached:
            logger.info(f"Cache hit for key: {key[:16]}...")
            return json.loads(cached)

        return None

    def set(self, prompt: str, model: str, response: dict, **kwargs):
        """Cache response"""
        key = self.get_cache_key(prompt, model, **kwargs)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(response)
        )

        logger.info(f"Cached response for key: {key[:16]}...")

class CachedLLMService:
    """LLM service with caching"""

    def __init__(self, llm_service, cache: LLMCache):
        self.llm = llm_service
        self.cache = cache

    def generate(self, prompt: str, model: str = 'haiku', **kwargs) -> dict:
        """Generate with caching"""
        # Check cache
        cached = self.cache.get(prompt, model, **kwargs)
        if cached:
            cached['cache_hit'] = True
            return cached

        # Generate
        response = self.llm.generate(prompt, model=model, **kwargs)

        # Cache response
        self.cache.set(prompt, model, response, **kwargs)

        response['cache_hit'] = False
        return response

# Usage
cache = LLMCache(redis_url='redis://localhost:6379', ttl=3600)
cached_service = CachedLLMService(llm_service, cache)

result = cached_service.generate("What is AI?", model='haiku')
```

---

## 8. Async/Await Integration

### 8.1 Async Python

```python
import asyncio
import httpx

class AsyncLLMClient:
    """Async LLM client"""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={'Authorization': f'Bearer {API_KEY}'}
        )

    async def generate(self, prompt: str) -> dict:
        """Async generate"""
        response = await self.client.post(
            'https://api.anthropic.com/v1/messages',
            json={
                'model': 'claude-3-haiku-20240307',
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 1024,
            }
        )

        return response.json()

    async def generate_many(self, prompts: List[str]) -> List[dict]:
        """Generate multiple in parallel"""
        tasks = [self.generate(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return [
            r if not isinstance(r, Exception) else {'error': str(r)}
            for r in results
        ]

# Usage with FastAPI
from fastapi import FastAPI

app = FastAPI()
llm_client = AsyncLLMClient()

@app.post('/api/generate')
async def generate(request: dict):
    """Async endpoint"""
    result = await llm_client.generate(request['prompt'])
    return result

@app.post('/api/batch')
async def batch_generate(request: dict):
    """Batch async endpoint"""
    results = await llm_client.generate_many(request['prompts'])
    return {'results': results}
```

---

## 9. Client SDKs

### 9.1 Python SDK

```python
import requests
from typing import Optional, List

class AIServiceSDK:
    """Python SDK for AI service"""

    def __init__(self, api_key: str, base_url: str = 'https://api.example.com'):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': api_key})

    def generate(self, prompt: str, model: str = 'haiku',
                max_tokens: int = 1024) -> dict:
        """Generate text"""
        response = self.session.post(
            f'{self.base_url}/api/v1/generate',
            json={
                'prompt': prompt,
                'model': model,
                'max_tokens': max_tokens,
            },
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

    def chat(self, messages: List[dict], model: str = 'haiku') -> dict:
        """Chat with history"""
        response = self.session.post(
            f'{self.base_url}/api/v1/chat',
            json={'messages': messages, 'model': model},
            timeout=30,
        )

        response.raise_for_status()
        return response.json()

    def stream_generate(self, prompt: str):
        """Stream generated text"""
        with self.session.post(
            f'{self.base_url}/api/v1/generate/stream',
            json={'prompt': prompt},
            stream=True,
            timeout=60,
        ) as response:
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data: '):
                        yield json.loads(decoded[6:])

# Usage
sdk = AIServiceSDK(api_key='your-api-key')

# Simple generation
result = sdk.generate("Explain quantum computing")
print(result['text'])

# Streaming
for chunk in sdk.stream_generate("Write a story"):
    if chunk['type'] == 'token':
        print(chunk['content'], end='', flush=True)
```

### 9.2 JavaScript SDK

```javascript
class AIServiceSDK {
    constructor(apiKey, baseURL = 'https://api.example.com') {
        this.apiKey = apiKey;
        this.baseURL = baseURL;
    }

    async generate(prompt, options = {}) {
        const response = await fetch(`${this.baseURL}/api/v1/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey,
            },
            body: JSON.stringify({
                prompt,
                model: options.model || 'haiku',
                max_tokens: options.maxTokens || 1024,
            }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return response.json();
    }

    async *streamGenerate(prompt) {
        const response = await fetch(`${this.baseURL}/api/v1/generate/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey,
            },
            body: JSON.stringify({ prompt }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    yield JSON.parse(line.slice(6));
                }
            }
        }
    }

    async chat(messages, model = 'haiku') {
        const response = await fetch(`${this.baseURL}/api/v1/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': this.apiKey,
            },
            body: JSON.stringify({ messages, model }),
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        return response.json();
    }
}

// Usage
const sdk = new AIServiceSDK('your-api-key');

// Simple generation
const result = await sdk.generate('Explain quantum computing');
console.log(result.text);

// Streaming
for await (const chunk of sdk.streamGenerate('Write a story')) {
    if (chunk.type === 'token') {
        process.stdout.write(chunk.content);
    }
}
```

---

## 10. Testing Integration

### 10.1 Mocking LLM Responses

```python
from unittest.mock import Mock, patch
import pytest

class MockLLMResponse:
    """Mock LLM response for testing"""

    def __init__(self, text: str, input_tokens: int = 100, output_tokens: int = 200):
        self.content = [Mock(text=text)]
        self.usage = Mock(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

def test_llm_generation():
    """Test LLM generation with mock"""
    with patch('anthropic.Client') as mock_client:
        # Configure mock
        mock_response = MockLLMResponse("Mocked response")
        mock_client.return_value.messages.create.return_value = mock_response

        # Test
        client = LLMClient()
        result = client.generate("Test prompt")

        assert result['text'] == "Mocked response"
        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 200

@pytest.fixture
def mock_llm_service():
    """Fixture for mocked LLM service"""
    service = Mock()
    service.generate.return_value = {
        'text': 'Test response',
        'model': 'haiku',
        'usage': {'input_tokens': 10, 'output_tokens': 20}
    }
    return service

def test_api_endpoint(mock_llm_service):
    """Test API endpoint with mocked service"""
    app.dependency_overrides[get_llm_service] = lambda: mock_llm_service

    response = client.post('/api/generate', json={'prompt': 'test'})

    assert response.status_code == 200
    assert 'text' in response.json()
```

---

## 11. Integration Checklist

```yaml
Setup:
  - [ ] API keys configured (environment variables)
  - [ ] Provider SDKs installed
  - [ ] Error handling implemented
  - [ ] Retry logic with exponential backoff
  - [ ] Rate limiting configured
  - [ ] Timeout values set (30s recommended)

API Design:
  - [ ] RESTful endpoints defined
  - [ ] Request/response schemas documented
  - [ ] Authentication mechanism (API keys)
  - [ ] Versioning strategy (e.g., /api/v1/)
  - [ ] Error response standards
  - [ ] CORS configured (if web app)

Performance:
  - [ ] Caching layer implemented (Redis/in-memory)
  - [ ] Response streaming for long outputs
  - [ ] Batch processing for multiple requests
  - [ ] Async/await for non-blocking operations
  - [ ] Connection pooling configured

Monitoring:
  - [ ] Request/response logging
  - [ ] Token usage tracking
  - [ ] Cost calculation per request
  - [ ] Error rate monitoring
  - [ ] Latency metrics (p50/p95/p99)

Security:
  - [ ] Input validation on all endpoints
  - [ ] Output sanitization
  - [ ] API key rotation schedule
  - [ ] HTTPS/TLS enforced
  - [ ] Request signing (optional)

Testing:
  - [ ] Unit tests with mocked responses
  - [ ] Integration tests with real API (dev environment)
  - [ ] Load testing completed
  - [ ] Error handling tests
  - [ ] Fallback testing

Documentation:
  - [ ] API documentation (OpenAPI/Swagger)
  - [ ] SDK documentation and examples
  - [ ] Integration guide
  - [ ] Common errors and solutions
  - [ ] Rate limits documented
```

---

**Version:** 1.0
**Last Updated:** February 8, 2026
**Status:** Active
