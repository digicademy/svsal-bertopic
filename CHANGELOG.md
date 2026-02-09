# Changelog

## 02-embeddings-upload Recent Updates (February 2026)

This notebook has been updated to work with the refactored EmbAPI (v2):

### 1. New LLM Service Model
- **LLM Service Definitions**: Reusable templates (some pre-seeded by `_system`)
- **LLM Service Instances**: User-specific configurations that reference definitions
- Instances can be created from system definitions or as standalone configurations

### 2. Updated API Structure
- New endpoints: `/v1/llm-definitions/...` and `/v1/llm-instances/...`
- Projects now require both `instance_owner` and `instance_handle` to reference the LLM service
- Embeddings now require both `instance_owner` and `instance_handle` (replaces single `llm_service_handle`)

### 3. Smart Data Loading with Fallbacks
- Automatically detects and loads the most recent embedding files
- Tries multiple formats in order of preference: **parquet → pickle → CSV**
- Handles the nested provider structure from `01-embeddings-create` notebook
- Loads embeddings for multiple providers: `{provider_name: {doc_id: [embedding]}}`

### 4. Full Polars Support
- Replaced all pandas operations with polars for better performance
- Efficient data handling for large datasets
- Type-safe operations throughout

### 5. Multi-Provider Upload System
- **Automatically uploads embeddings from different models to different VDB projects**
- Provider-to-project mapping defined in configuration
- Each embedding provider (OpenAI, Cohere, Gemini, etc.) uploads to its designated project
- Parallel-ready architecture for future optimization

### 6. Enhanced Upload Management
- Comprehensive progress tracking per provider
- Detailed success/failure reporting
- Option to upload specific providers only
- Configurable retry logic and rate limiting

## 02-embeddings-upload Recent Updates (January 2026)

This notebook has been enhanced with the following improvements:

### 1. Smart Data Loading with Fallbacks
- Automatically detects and loads the most recent embedding files
- Tries multiple formats in order of preference: **parquet → pickle → CSV**
- Handles the nested provider structure from `01-embeddings-create` notebook
- Loads embeddings for multiple providers: `{provider_name: {doc_id: [embedding]}}`

### 2. Full Polars Support
- Replaced all pandas operations with polars for better performance
- Efficient data handling for large datasets
- Type-safe operations throughout

### 3. Multi-Provider Upload System
- **Automatically uploads embeddings from different models to different VDB projects**
- Provider-to-project mapping defined in configuration
- Each embedding provider (OpenAI, Cohere, Gemini, etc.) uploads to its designated project
- Parallel-ready architecture for future optimization

### 4. Enhanced Upload Management
- Comprehensive progress tracking per provider
- Detailed success/failure reporting
- Option to upload specific providers only
- Configurable retry logic and rate limiting
