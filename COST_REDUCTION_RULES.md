# AI Development Cost Reduction Rules

## Overview

This document outlines practical rules and guidelines to minimize costs in AI development projects while maintaining quality and functionality.

---

## 1. Model Selection Strategy

### Rule 1.1: Use the Smallest Capable Model
- **Always start with the cheapest model** that can handle the task
- Only upgrade to larger models when necessary
- Examples:
  - Text classification → Use Haiku or GPT-3.5
  - Complex reasoning → Use Sonnet
  - Critical decisions only → Use Opus/GPT-4

### Rule 1.2: Model-Specific Task Routing
```
Simple tasks (classification, extraction):     Haiku / GPT-3.5 Turbo
Medium complexity (summarization, analysis):   Sonnet / GPT-4
Complex reasoning (planning, architecture):    Opus / GPT-4 Turbo
```

### Rule 1.3: Avoid Default Model Selection
- Never default to the most expensive model
- Explicitly choose models based on task requirements
- Document why a more expensive model is needed

---

## 2. Token Optimization

### Rule 2.1: Minimize Input Tokens
- Keep prompts concise and focused
- Remove unnecessary context, examples, or formatting
- Use template variables instead of repeating text
- Strip whitespace and unnecessary line breaks

### Rule 2.2: Limit Output Tokens
- Set `max_tokens` parameter to reasonable limits
- Request structured outputs (JSON) instead of verbose text
- Use bullet points instead of full paragraphs when appropriate

### Rule 2.3: Avoid Redundant API Calls
- Cache responses for identical inputs
- Batch similar requests together
- Use streaming only when needed (costs same but better UX)

---

## 3. Caching Strategies

### Rule 3.1: Implement Response Caching
```python
# Cache API responses with TTL
cache = {
    "key": (response, timestamp),
}
TTL = 3600  # 1 hour
```

### Rule 3.2: Use Prompt Caching (Claude)
- Structure prompts to maximize cache hits
- Put static context at the beginning
- Keep variable content at the end
- Minimum 1024 tokens for cache eligibility (Claude)

### Rule 3.3: Cache Embeddings
- Store embeddings in database/cache
- Reuse embeddings for similar content
- Update only when source changes

---

## 4. Prompt Engineering

### Rule 4.1: Single-Shot Over Few-Shot
- Avoid providing multiple examples unless necessary
- Use clear instructions instead of examples
- Each example = more input tokens = higher cost

### Rule 4.2: Structured Outputs
- Request JSON/structured format to reduce verbose output
- Use schemas to enforce concise responses
- Example:
  ```json
  {
    "answer": "yes",
    "confidence": 0.95,
    "reason": "brief explanation"
  }
  ```

### Rule 4.3: Avoid Over-Engineering Prompts
- Start simple, add complexity only if needed
- Remove "politeness" phrases ("please", "kindly")
- Get straight to the point

---

## 5. Development Practices

### Rule 5.1: Test with Cheap Models First
- Develop and debug using Haiku/GPT-3.5
- Switch to expensive models only for final testing
- Use mocked responses for unit tests

### Rule 5.2: Implement Rate Limiting
```python
# Limit API calls per user/session
MAX_CALLS_PER_HOUR = 100
MAX_CALLS_PER_DAY = 1000
```

### Rule 5.3: Set Request Timeouts
- Prevent hanging requests that accumulate costs
- Use reasonable timeout values (10-30 seconds)
- Implement retry logic with exponential backoff

---

## 6. Monitoring and Optimization

### Rule 6.1: Track Token Usage
```python
# Log every API call
{
    "timestamp": "2024-01-01T10:00:00Z",
    "model": "claude-3-haiku",
    "input_tokens": 150,
    "output_tokens": 200,
    "cost": 0.0008,
    "endpoint": "/api/analyze"
}
```

### Rule 6.2: Set Budget Alerts
- Configure daily/monthly spending limits
- Alert when 80% of budget is reached
- Automatically throttle or disable if limit exceeded

### Rule 6.3: Regular Cost Reviews
- Weekly: Review top expensive endpoints
- Monthly: Analyze trends and optimization opportunities
- Identify and refactor inefficient patterns

---

## 7. Architecture Patterns

### Rule 7.1: Use Deterministic Logic First
- Don't use LLMs for tasks that can be solved with code
- Examples to avoid:
  - ❌ LLM for date formatting
  - ❌ LLM for simple math calculations
  - ❌ LLM for regex matching
  - ✅ LLM for natural language understanding
  - ✅ LLM for creative generation
  - ✅ LLM for complex reasoning

### Rule 7.2: Implement Fallback Strategies
```python
# Try cheap model first, fallback to expensive only if needed
response = try_haiku(prompt)
if confidence < 0.8:
    response = try_sonnet(prompt)
```

### Rule 7.3: Preprocess and Filter
- Filter invalid inputs before API calls
- Validate data locally first
- Use client-side validation when possible

---

## 8. Context Management

### Rule 8.1: Limit Conversation History
- Keep only last N messages in context
- Summarize old conversations instead of including full history
- Typical limit: 10-20 messages

### Rule 8.2: Smart Context Pruning
```python
# Keep only relevant context
def get_context(messages):
    recent = messages[-5:]  # Last 5 messages
    important = [m for m in messages if m.get('important')]
    return recent + important
```

### Rule 8.3: Avoid Context Duplication
- Don't repeat information already in context
- Reference previous messages instead of re-sending data

---

## 9. Feature Flags and Gradual Rollout

### Rule 9.1: Feature Flags for AI Features
```python
ENABLE_AI_FEATURE = os.getenv('ENABLE_AI_FEATURE', 'false') == 'true'
```
- Roll out AI features gradually
- Monitor costs before full deployment
- A/B test expensive features

### Rule 9.2: User Tier Limits
```python
LIMITS = {
    'free': 10,      # requests per day
    'basic': 100,
    'premium': 1000,
}
```

---

## 10. Data Efficiency

### Rule 10.1: Compress Long Documents
- Summarize long documents before processing
- Use chunking with overlap for context
- Extract only relevant sections

### Rule 10.2: Avoid Redundant Preprocessing
- Preprocess data once, reuse results
- Store cleaned/normalized data
- Cache intermediate results

### Rule 10.3: Smart Retrieval (RAG)
- Return only top K most relevant chunks (K=3-5)
- Don't send entire vector database results
- Filter by relevance score threshold

---

## 11. Testing Cost Reduction

### Rule 11.1: Mock Responses in Tests
```python
# Use mock responses for unit tests
mock_response = {
    "content": "test response",
    "usage": {"input_tokens": 10, "output_tokens": 20}
}
```

### Rule 11.2: Dedicated Test Environment
- Use separate API keys for dev/test/prod
- Set lower rate limits for test environments
- Use cheapest models for CI/CD tests

### Rule 11.3: Smoke Tests Only
- Don't run full AI pipeline in every test
- Test critical paths only
- Use cached responses for regression tests

---

## 12. Emergency Cost Controls

### Rule 12.1: Circuit Breaker Pattern
```python
if hourly_cost > THRESHOLD:
    disable_ai_features()
    alert_team()
    log_incident()
```

### Rule 12.2: Graceful Degradation
- Have non-AI fallback for critical features
- Show cached/default results when over budget
- Queue requests for later processing

### Rule 12.3: Manual Override
- Provide admin controls to disable expensive features
- Emergency shutdown procedure
- Clear escalation path

---

## Cost Calculation Reference

### Claude 3.5 (Anthropic)
```
Haiku:   $0.25 / 1M input tokens,  $1.25 / 1M output tokens
Sonnet:  $3.00 / 1M input tokens, $15.00 / 1M output tokens
Opus:   $15.00 / 1M input tokens, $75.00 / 1M output tokens
```

### OpenAI
```
GPT-3.5 Turbo: $0.50 / 1M input tokens, $1.50 / 1M output tokens
GPT-4:         $30.00 / 1M input tokens, $60.00 / 1M output tokens
GPT-4 Turbo:   $10.00 / 1M input tokens, $30.00 / 1M output tokens
```

---

## Quick Wins Checklist

- [ ] Use Haiku/GPT-3.5 for simple tasks
- [ ] Set max_tokens limits on all API calls
- [ ] Implement response caching with TTL
- [ ] Remove unnecessary examples from prompts
- [ ] Add token usage logging
- [ ] Set up budget alerts
- [ ] Use deterministic logic instead of LLM where possible
- [ ] Limit conversation history to 10 messages
- [ ] Mock API calls in tests
- [ ] Implement rate limiting per user
- [ ] Add circuit breaker for cost spikes
- [ ] Review and optimize top 5 most expensive endpoints

---

## Resources

- [Anthropic Pricing](https://www.anthropic.com/pricing)
- [OpenAI Pricing](https://openai.com/pricing)
- [Token Counting Tools](https://platform.openai.com/tokenizer)
- [Cost Optimization Guide](https://docs.anthropic.com/claude/docs/cost-optimization)

---

**Last Updated:** February 8, 2026
**Version:** 1.0
