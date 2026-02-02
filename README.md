# RoBert

A sentiment analysis tool powered by Claude AI.

## Tool Overview

RoBert is a sentiment analysis tool that leverages Anthropic's Claude large language model to classify the emotional tone of text inputs. Rather than relying on traditional machine learning pipelines that require labeled training data and model fine-tuning, RoBert uses Claude's natural language understanding capabilities to perform zero-shot sentiment classification.

## Methodology

### 1. Input Collection

Text data is submitted to RoBert as raw, unstructured natural language. Inputs can range from single sentences to multi-paragraph passages such as product reviews, social media posts, or survey responses.

### 2. Preprocessing

Before analysis, the following preprocessing steps are applied:

- **Whitespace normalization** — leading/trailing whitespace is trimmed and internal whitespace is collapsed.
- **Encoding normalization** — text is standardized to UTF-8 to handle special characters and emoji consistently.
- **Length truncation** — inputs exceeding the context window limit are truncated with a warning to the user.

No stemming, lemmatization, or stopword removal is performed. Claude operates on natural language directly, so preserving the original phrasing, punctuation, and casing is important for accurate sentiment detection.

### 3. Prompt Construction

RoBert constructs a structured prompt that instructs Claude to:

1. Analyze the sentiment of the provided text.
2. Classify it into one of the following categories: **Positive**, **Negative**, **Neutral**, or **Mixed**.
3. Provide a confidence score (0.0–1.0) for the classification.
4. Return a brief rationale explaining the reasoning behind the classification.

The prompt uses a system message to establish Claude as a sentiment analysis specialist, followed by the user text embedded in a clearly delimited block.

### 4. Sentiment Classification

Claude processes the prompt and returns a structured response containing:

| Field        | Description                                                  |
|--------------|--------------------------------------------------------------|
| `sentiment`  | The classified sentiment: Positive, Negative, Neutral, Mixed |
| `confidence` | A float between 0.0 and 1.0 indicating model confidence      |
| `rationale`  | A short explanation of why the sentiment was assigned         |

### 5. Post-Processing

The raw Claude response is parsed and validated:

- The `sentiment` field is checked against the allowed categories.
- The `confidence` score is verified to be within the valid range.
- Malformed responses trigger a retry with an adjusted prompt.

### 6. Output

The final structured result is returned to the caller in JSON format:

```json
{
  "text": "The original input text...",
  "sentiment": "Positive",
  "confidence": 0.92,
  "rationale": "The text expresses satisfaction and uses strongly favorable language."
}
```

## Advantages of This Approach

- **No training data required** — Claude performs zero-shot classification, eliminating the need for labeled datasets.
- **Context-aware** — Claude understands sarcasm, negation, and nuanced language better than traditional lexicon-based or bag-of-words models.
- **Multilingual support** — Claude can analyze sentiment across multiple languages without separate models.
- **Explainable** — Every classification includes a human-readable rationale.

## Limitations

- **Latency** — API-based inference introduces network latency compared to local model inference.
- **Cost** — Each classification requires an API call to Anthropic.
- **Determinism** — Results may vary slightly between identical calls due to the probabilistic nature of LLM outputs. Temperature is set to 0 to minimize this.
- **Context window** — Very long texts must be truncated or chunked, which may lose broader context.
