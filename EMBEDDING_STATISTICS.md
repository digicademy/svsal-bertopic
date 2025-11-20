# Embedding Generation Statistics

This document describes the enhanced error handling and statistics tracking features added to `01-embeddings-create.20250503.ipynb`.

## Overview

The notebook now includes comprehensive logging and statistics tracking for the embedding generation process. This helps monitor the health and success rate of embeddings across different providers (OpenAI, Academic Cloud, Cohere).

## Features

### 1. Structured Logging

The notebook uses Python's `logging` module for structured output:

- **INFO**: Normal operations (provider start/complete, caching info)
- **WARNING**: Recoverable errors (rate limits, individual document failures)
- **ERROR**: Critical failures (empty embeddings, exceptions)

Example log output:
```
2025-11-20 01:46:38,732 - __main__ - INFO - Provider openai_text-embedding-3-small started processing
2025-11-20 01:46:39,145 - __main__ - WARNING - Rate limit (429) for document doc_123 with provider openai_text-embedding-3-small
2025-11-20 01:46:45,821 - __main__ - INFO - Provider openai_text-embedding-3-small completed processing
```

### 2. Per-Provider Statistics

The `EmbeddingStatistics` class tracks metrics for each provider:

| Metric | Description |
|--------|-------------|
| **Total** | Total documents processed |
| **Success** | Successfully generated embeddings |
| **Failed** | Failed embedding attempts |
| **Success Rate** | Percentage of successful embeddings |
| **Processing Time** | Time from start to completion (seconds) |
| **Rate Limit Hits** | Number of 429 (rate limit) errors encountered |
| **Failed Docs** | List of failed documents with error details |

### 3. Summary Reports

After all providers complete, the notebook displays:

#### Per-Provider Statistics
```
Provider Statistics:
================================================================================

openai_text-embedding-3-small:
  Total: 1000
  Success: 998 (99.8%)
  Failed: 2 (0.2%)
  Time: 125.3s

academic-cloud_e5-mistral-7b-instruct:
  Total: 1000
  Success: 950 (95.0%)
  Failed: 50 (5.0%)
  Time: 450.2s
  Rate limit hits: 3

cohere_embed-v4.0_clustering:
  Total: 1000
  Success: 1000 (100.0%)
  Failed: 0 (0.0%)
  Time: 89.7s
```

#### Overall Summary
```
Overall Summary:
================================================================================
Total documents: 1000
Critical failures (failed for all providers): 0
Documents with partial success: 50
Documents with full success: 950
```

#### Failure Analysis
- **Critical Failures**: Documents that failed for ALL providers
- **Partial Success**: Documents that succeeded with at least one provider but failed with others
- **Full Success**: Documents that succeeded with all providers

### 4. Statistics Export

Statistics are automatically saved to a JSON file with the pattern:
```
{date}_embedding_statistics.json
```

Example content:
```json
{
  "timestamp": "2025-11-20 12:34:56",
  "total_documents": 1000,
  "providers": {
    "openai_text-embedding-3-small": {
      "provider_name": "openai_text-embedding-3-small",
      "total": 1000,
      "success": 998,
      "failed": 2,
      "success_rate": 99.8,
      "processing_time": 125.3,
      "rate_limit_hits": 0,
      "failed_docs": [
        {
          "doc_id": "https://id.salamanca.school/texts/W0103:doc123",
          "error": "Connection timeout"
        }
      ]
    }
  },
  "summary": {
    "critical_failures": 0,
    "partial_success": 50,
    "full_success": 950,
    "critical_failure_docs": []
  }
}
```

## Modified Functions

### `retrieve_embeddings(provider, semaphore, doc_id, content, stats)`

Now accepts a `stats` parameter and:
- Records successful embeddings via `stats.record_success(doc_id)`
- Records failures via `stats.record_failure(doc_id, error_msg)`
- Detects and logs rate limit errors (429)
- Logs all errors with appropriate severity levels

### `control_embeddings_retrieval(provider, semaphore, dfi, progress, task_id, stats)`

Passes the `stats` parameter to `retrieve_embeddings()` for each document.

### `get_embeddings_for_provider(provider, semaphore, dfi, progress, task_id, stats)`

Enhanced with:
- `stats.start()` at the beginning
- `stats.complete()` at the end
- Tracks cached documents as successes
- Logs caching information

### `get_all_embeddings(providers, dfi)`

Now returns a tuple: `(embeddings_dict, stats_dict)`
- Creates `EmbeddingStatistics` instance for each provider
- Returns statistics alongside embeddings

## Usage

The statistics are automatically collected during normal notebook execution. After running the main embedding generation cell, the following cells will display and export the statistics.

## Benefits

1. **Visibility**: Clear understanding of which providers and documents succeed/fail
2. **Debugging**: Detailed error information for failed documents
3. **Monitoring**: Track rate limit hits and processing times
4. **Analysis**: Identify systematic failures across providers
5. **Reporting**: JSON export for further analysis or dashboards

## Backward Compatibility

The changes are fully backward compatible:
- Existing embedding generation functionality is unchanged
- Statistics collection happens transparently
- All original output files are still generated
- New statistics file is additional, not replacing anything
