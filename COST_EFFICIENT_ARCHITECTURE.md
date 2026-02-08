# AI Application Architecture for Cost Efficiency

## Overview

This document outlines architectural patterns and designs for building cost-efficient AI applications that minimize unnecessary LLM usage while maximizing functionality.

---

## Core Architecture Principle

**🎯 LLMs are expensive last-resort tools, not first-choice solutions.**

```
┌─────────────────────────────────────────────────────────────┐
│                     Request Pipeline                        │
│                                                             │
│  1. Deterministic Logic (FREE)                             │
│         ↓                                                   │
│  2. Rule-Based Systems (FREE)                              │
│         ↓                                                   │
│  3. Cache Lookup (CHEAP)                                   │
│         ↓                                                   │
│  4. Cheap Model (Haiku/GPT-3.5)                           │
│         ↓                                                   │
│  5. Expensive Model (Sonnet/Opus) - ONLY IF NECESSARY     │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Layered Decision Architecture

### Layer 0: Input Validation (Deterministic)

**Cost: $0.00 | Speed: Microseconds**

```python
class InputValidator:
    """Always validate before touching LLM"""

    def validate(self, input_data):
        # Length checks
        if len(input_data) > MAX_LENGTH:
            return ValidationError("Input too long")

        # Format validation
        if not self.is_valid_format(input_data):
            return ValidationError("Invalid format")

        # Content filtering
        if self.contains_banned_content(input_data):
            return ValidationError("Inappropriate content")

        # Language detection (using library)
        if detect_language(input_data) not in SUPPORTED_LANGUAGES:
            return ValidationError("Unsupported language")

        return ValidationSuccess()

# ❌ BAD: Send everything to LLM
response = llm.call("Validate this input: " + user_input)

# ✅ GOOD: Validate first, LLM only if needed
validation = validator.validate(user_input)
if not validation.is_valid():
    return validation.error
# Only now consider LLM if needed
```

### Layer 1: Rule-Based Logic (Deterministic)

**Cost: $0.00 | Speed: Milliseconds**

```python
class IntentRouter:
    """Route requests using rules before LLM"""

    def route(self, message: str):
        # Keyword matching
        if any(word in message.lower() for word in ['price', 'cost', 'how much']):
            return self.handle_pricing_query()

        # Regex patterns
        if re.match(r'^\d{10}$', message):  # Phone number
            return self.handle_phone_lookup()

        # Command detection
        if message.startswith('/'):
            return self.handle_command(message)

        # FAQ matching
        faq_match = self.fuzzy_match_faq(message)
        if faq_match and faq_match.confidence > 0.9:
            return faq_match.answer

        # Only use LLM for complex queries
        return self.llm_intent_classification(message)
```

### Layer 2: Cache Layer

**Cost: ~$0.00 | Speed: Milliseconds**

```python
class SmartCache:
    """Multi-level caching strategy"""

    def __init__(self):
        self.exact_cache = {}      # Exact match
        self.semantic_cache = {}   # Similar queries
        self.response_cache = {}   # Response templates

    def get(self, query: str):
        # Level 1: Exact match
        if query in self.exact_cache:
            return self.exact_cache[query]

        # Level 2: Semantic similarity (using embeddings)
        similar = self.find_similar(query, threshold=0.95)
        if similar:
            return self.semantic_cache[similar]

        # Level 3: Template match
        template = self.match_template(query)
        if template:
            return self.fill_template(template, query)

        return None  # Cache miss - proceed to LLM
```

### Layer 3: Model Selection Layer

**Cost: Varies | Speed: Seconds**

```python
class ModelRouter:
    """Choose cheapest model capable of handling task"""

    TASK_COMPLEXITY = {
        'extract_email': 1,        # Regex can do this
        'sentiment': 2,            # Haiku
        'summarize': 3,            # Haiku/Sonnet
        'reasoning': 4,            # Sonnet
        'creative_writing': 5,     # Sonnet/Opus
        'complex_analysis': 6,     # Opus
    }

    def route(self, task_type: str, content: str):
        complexity = self.TASK_COMPLEXITY.get(task_type, 3)

        # Try deterministic first
        if complexity == 1:
            return self.deterministic_handler(content)

        # Use cheap model for simple tasks
        if complexity <= 3:
            response = self.haiku.call(content)
            # Fallback to better model if confidence is low
            if response.confidence < 0.8:
                return self.sonnet.call(content)
            return response

        # Complex tasks - use expensive model
        return self.opus.call(content)
```

---

## 2. Component Architecture

### A. Request Processing Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   User Request                              │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Pre-Processing Layer (Deterministic)                       │
│  • Input sanitization                                       │
│  • Format validation                                        │
│  • Language detection                                       │
│  • Length normalization                                     │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Rule Engine (Deterministic)                                │
│  • Keyword matching                                         │
│  • Regex patterns                                           │
│  • FAQ lookup                                               │
│  • Command routing                                          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
              Can Handle? ───YES──→ Return Response
                     │
                     NO
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Cache Layer                                                │
│  • Exact match cache                                        │
│  • Semantic similarity cache                                │
│  • Response template cache                                  │
└────────────────────┬────────────────────────────────────────┘
                     ↓
              Cache Hit? ───YES──→ Return Cached Response
                     │
                     NO
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Model Router                                               │
│  • Classify task complexity                                 │
│  • Select cheapest capable model                            │
│  • Apply rate limiting                                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  LLM Execution                                              │
│  • Token counting                                           │
│  • API call with timeout                                    │
│  • Response validation                                      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Post-Processing                                            │
│  • Cache response                                           │
│  • Log token usage                                          │
│  • Update metrics                                           │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  Response                                                   │
└─────────────────────────────────────────────────────────────┘
```

### B. System Components

```
┌──────────────────────────────────────────────────────────────┐
│                    API Gateway Layer                         │
│  • Rate limiting per user/tier                              │
│  • Request validation                                        │
│  • Authentication/Authorization                              │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│              Application Service Layer                       │
│                                                              │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Rule Engine     │  │ Cache Service│  │ Model Router   │ │
│  │ (Deterministic) │  │              │  │                │ │
│  └─────────────────┘  └──────────────┘  └────────────────┘ │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│              LLM Abstraction Layer                           │
│  • Provider-agnostic interface                              │
│  • Fallback logic                                           │
│  • Token counting                                           │
│  • Cost tracking                                            │
└────────────────────┬─────────────────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────────────────┐
│              External LLM Providers                          │
│  [ Anthropic ]  [ OpenAI ]  [ Others ]                      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│              Observability Layer (Cross-cutting)             │
│  • Token usage metrics                                       │
│  • Cost tracking per endpoint                                │
│  • Performance monitoring                                    │
│  • Alert system                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Deterministic Logic Examples

### Example 1: Email Extraction

```python
# ❌ EXPENSIVE: Using LLM
def extract_email_expensive(text: str):
    prompt = f"Extract the email address from this text: {text}"
    response = llm.call(prompt)  # Cost: ~$0.001 per request
    return response

# ✅ CHEAP: Using regex
def extract_email_cheap(text: str):
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(pattern, text)
    return emails[0] if emails else None  # Cost: $0.00
```

**Savings:** 100% cost reduction, 1000x faster

### Example 2: Date Parsing

```python
# ❌ EXPENSIVE: Using LLM
def parse_date_expensive(date_string: str):
    prompt = f"Parse this date and return ISO format: {date_string}"
    response = llm.call(prompt)
    return response

# ✅ CHEAP: Using dateutil
from dateutil import parser

def parse_date_cheap(date_string: str):
    try:
        dt = parser.parse(date_string)
        return dt.isoformat()
    except ValueError:
        return None  # Or use LLM only for ambiguous cases
```

**Savings:** 100% cost reduction, instant response

### Example 3: Sentiment Analysis

```python
# ❌ EXPENSIVE: Always using LLM
def sentiment_expensive(text: str):
    prompt = f"Classify sentiment: {text}"
    return llm.call(prompt)  # Cost: $0.001-0.01 per request

# ✅ SMART: Hybrid approach
from textblob import TextBlob

def sentiment_smart(text: str):
    # Use free library for clear cases
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    # Clear positive
    if polarity > 0.5:
        return {"sentiment": "positive", "confidence": 0.95}
    # Clear negative
    elif polarity < -0.5:
        return {"sentiment": "negative", "confidence": 0.95}
    # Unclear - use LLM
    else:
        return llm.classify_sentiment(text)
```

**Savings:** 70-80% cost reduction (most cases handled by free library)

### Example 4: Language Detection

```python
# ❌ EXPENSIVE: Using LLM
def detect_language_expensive(text: str):
    prompt = f"What language is this: {text}"
    return llm.call(prompt)

# ✅ CHEAP: Using langdetect
from langdetect import detect

def detect_language_cheap(text: str):
    try:
        return detect(text)  # Cost: $0.00, accuracy: 95%+
    except:
        return 'unknown'
```

**Savings:** 100% cost reduction

### Example 5: Data Validation

```python
# ❌ EXPENSIVE: Using LLM
def validate_phone_expensive(phone: str):
    prompt = f"Is this a valid phone number: {phone}"
    return llm.call(prompt)

# ✅ CHEAP: Using regex and phonenumbers library
import phonenumbers

def validate_phone_cheap(phone: str, region: str = 'US'):
    try:
        parsed = phonenumbers.parse(phone, region)
        return phonenumbers.is_valid_number(parsed)
    except:
        return False
```

**Savings:** 100% cost reduction

### Example 6: Simple Classification

```python
# ❌ EXPENSIVE: Using LLM for everything
def classify_ticket_expensive(ticket: str):
    prompt = f"Classify this support ticket: {ticket}"
    return llm.call(prompt)

# ✅ CHEAP: Keywords + rules first, LLM for edge cases
def classify_ticket_smart(ticket: str):
    ticket_lower = ticket.lower()

    # Rule-based classification (handles 80% of cases)
    keywords = {
        'billing': ['invoice', 'payment', 'charge', 'refund', 'price'],
        'technical': ['error', 'bug', 'crash', 'not working', 'broken'],
        'account': ['login', 'password', 'reset', 'access', 'locked'],
    }

    for category, words in keywords.items():
        if any(word in ticket_lower for word in words):
            return {
                'category': category,
                'confidence': 0.9,
                'method': 'rule-based'
            }

    # LLM for complex/ambiguous cases (20%)
    return llm.classify(ticket)
```

**Savings:** 80% cost reduction

### Example 7: Text Normalization

```python
# ❌ EXPENSIVE: Using LLM
def normalize_text_expensive(text: str):
    prompt = f"Normalize and clean this text: {text}"
    return llm.call(prompt)

# ✅ CHEAP: String operations
def normalize_text_cheap(text: str):
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Trim
    text = text.strip()
    return text
```

**Savings:** 100% cost reduction

---

## 4. Decision Matrix: When to Use LLM

| Task Type | Use Deterministic | Use LLM | Justification |
|-----------|------------------|---------|---------------|
| Email extraction | ✅ | ❌ | Regex is 100% accurate and free |
| Date parsing | ✅ | ❌ | Libraries handle all formats |
| URL validation | ✅ | ❌ | Regex/validators work perfectly |
| Phone validation | ✅ | ❌ | phonenumbers library is comprehensive |
| Language detection | ✅ | ❌ | langdetect is 95%+ accurate |
| Simple math | ✅ | ❌ | eval() or math library |
| Sentiment (clear) | ✅ | ❌ | TextBlob/VADER for obvious cases |
| Sentiment (nuanced) | ❌ | ✅ | Sarcasm, context needs LLM |
| Intent (keywords) | ✅ | ❌ | Pattern matching sufficient |
| Intent (complex) | ❌ | ✅ | Natural language understanding |
| Translation | ❌ | ✅ | Quality matters |
| Summarization | ❌ | ✅ | Requires understanding |
| Creative writing | ❌ | ✅ | Human-like output needed |
| Code generation | ❌ | ✅ | Complex logic required |

---

## 5. Architecture Patterns

### Pattern 1: Cascade Pattern

```python
class CascadeProcessor:
    """Try cheap methods first, cascade to expensive"""

    def process(self, input_data):
        # Level 1: Free
        result = self.try_deterministic(input_data)
        if result.confidence > 0.95:
            return result

        # Level 2: Cheap
        result = self.try_haiku(input_data)
        if result.confidence > 0.90:
            return result

        # Level 3: Expensive
        return self.try_opus(input_data)
```

### Pattern 2: Hybrid Pattern

```python
class HybridProcessor:
    """Combine deterministic + LLM"""

    def process(self, input_data):
        # Step 1: Extract structured data (free)
        structured = self.extract_deterministic(input_data)

        # Step 2: Only send unstructured parts to LLM
        if structured.has_ambiguous_parts():
            structured.ambiguous = self.llm_process(
                structured.ambiguous_parts
            )

        return structured
```

### Pattern 3: Preprocessing Pattern

```python
class PreprocessingPipeline:
    """Reduce LLM input size with deterministic preprocessing"""

    def process(self, document: str):
        # Step 1: Extract relevant sections (free)
        relevant = self.extract_relevant_sections(document)

        # Step 2: Remove boilerplate (free)
        cleaned = self.remove_boilerplate(relevant)

        # Step 3: Chunk intelligently (free)
        chunks = self.smart_chunking(cleaned)

        # Step 4: Only process most relevant chunk with LLM
        most_relevant = self.rank_chunks(chunks)[0]
        return self.llm_process(most_relevant)  # Reduced tokens
```

---

## 6. Cost-Aware Design Checklist

### Before Writing Code

- [ ] Can this be solved with regex?
- [ ] Can this be solved with a library?
- [ ] Can this be solved with simple logic?
- [ ] Is there a rule-based approach?
- [ ] Can I cache the results?
- [ ] What's the simplest model that works?

### During Implementation

- [ ] Added input validation (deterministic)
- [ ] Implemented caching layer
- [ ] Set max_tokens limits
- [ ] Added token usage logging
- [ ] Implemented rate limiting
- [ ] Added confidence-based fallback
- [ ] Error handling with retries

### After Implementation

- [ ] Measured token usage
- [ ] Calculated cost per request
- [ ] Identified optimization opportunities
- [ ] Set up monitoring alerts
- [ ] Documented why LLM is necessary

---

## 7. Anti-Patterns to Avoid

### ❌ Anti-Pattern 1: LLM for Everything

```python
# BAD: Using LLM for simple tasks
def is_email_valid(email):
    return llm.call(f"Is {email} valid?")  # $0.001

def format_date(date):
    return llm.call(f"Format {date} as ISO")  # $0.001

def uppercase(text):
    return llm.call(f"Convert to uppercase: {text}")  # $0.001
```

**Cost:** $3.00 per 1000 requests
**Fix:** Use built-in string/regex/libraries → $0.00

### ❌ Anti-Pattern 2: No Caching

```python
# BAD: Calling LLM every time
def get_category(product):
    return llm.classify(product)  # Same products = repeated calls
```

**Fix:**
```python
# GOOD: Cache results
@cache(ttl=3600)
def get_category(product):
    return llm.classify(product)
```

### ❌ Anti-Pattern 3: Unbounded Context

```python
# BAD: Sending entire conversation history
def chat(message, history):
    full_context = "\n".join(history) + message  # Grows forever
    return llm.call(full_context)
```

**Fix:**
```python
# GOOD: Limit context window
def chat(message, history):
    recent = history[-10:]  # Only last 10 messages
    context = "\n".join(recent) + message
    return llm.call(context)
```

---

## 8. Reference Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                     Load Balancer                       │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│                   API Gateway                           │
│  • Rate Limiting: 100 req/min per user                 │
│  • Auth: JWT tokens                                     │
│  • Input Validation: Length, format                     │
└────────────────────┬────────────────────────────────────┘
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
┌──────────────────┐   ┌──────────────────┐
│ Rule Engine      │   │  Redis Cache     │
│ (Deterministic)  │←→│  TTL: 1 hour     │
│ • Regex          │   │  Hit Rate: 70%   │
│ • Keywords       │   │                  │
│ • FAQ Match      │   │                  │
└────────┬─────────┘   └──────────────────┘
         ↓
   Can Handle?
         │
    YES  │  NO
    ↓    ↓
┌────────┐  ┌──────────────────────────────┐
│ Return │  │   Model Router               │
└────────┘  │   ├─→ Haiku (70% of cases)   │
            │   ├─→ Sonnet (25% of cases)  │
            │   └─→ Opus (5% of cases)     │
            └────────────┬─────────────────┘
                         ↓
            ┌────────────────────────────┐
            │   LLM Provider Pool        │
            │   • Anthropic              │
            │   • OpenAI (fallback)      │
            │   • Timeout: 30s           │
            └────────────┬───────────────┘
                         ↓
            ┌────────────────────────────┐
            │   Observability            │
            │   • Prometheus metrics     │
            │   • Token usage logs       │
            │   • Cost per endpoint      │
            │   • Alert on budget 80%    │
            └────────────────────────────┘
```

---

## 9. Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Set up input validation layer
- [ ] Implement basic rule engine
- [ ] Add response caching (Redis/in-memory)
- [ ] Set up token usage logging

### Phase 2: Optimization (Week 2-3)
- [ ] Build model router
- [ ] Implement cascade pattern
- [ ] Add semantic similarity cache
- [ ] Set up monitoring dashboard

### Phase 3: Intelligence (Week 4)
- [ ] Add confidence-based fallback
- [ ] Implement smart context pruning
- [ ] Build cost prediction model
- [ ] Add A/B testing framework

### Phase 4: Scale (Ongoing)
- [ ] Optimize based on metrics
- [ ] Expand rule coverage
- [ ] Fine-tune cache TTLs
- [ ] Continuous cost optimization

---

## 10. Success Metrics

### Cost Efficiency
- **Target:** 70%+ requests handled without LLM
- **Target:** Average cost per request < $0.001
- **Target:** 50%+ cost reduction vs naive implementation

### Performance
- **Target:** p95 latency < 500ms for deterministic
- **Target:** p95 latency < 3s for LLM calls
- **Target:** Cache hit rate > 60%

### Quality
- **Target:** Accuracy > 95% for all methods
- **Target:** User satisfaction > 4.5/5
- **Target:** Error rate < 1%

---

**Version:** 1.0
**Last Updated:** February 8, 2026
**Status:** Active
