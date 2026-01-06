# Monitoring and Troubleshooting Guide

## Monitoring Overview

NOMYO Router provides comprehensive monitoring capabilities to track performance, health, and usage patterns.

## Monitoring Endpoints

### Health Check

```bash
curl http://localhost:12434/health
```

Response:
```json
{
  "status": "ok" | "error",
  "endpoints": {
    "http://endpoint1:11434": {
      "status": "ok" | "error",
      "version": "string" | "detail": "error message"
    }
  }
}
```

**HTTP Status Codes**:
- `200`: All endpoints healthy
- `503`: One or more endpoints unhealthy

### Current Usage

```bash
curl http://localhost:12434/api/usage
```

Response:
```json
{
  "usage_counts": {
    "http://endpoint1:11434": {
      "llama3": 2,
      "mistral": 1
    },
    "http://endpoint2:11434": {
      "llama3": 0,
      "mistral": 3
    }
  },
  "token_usage_counts": {
    "http://endpoint1:11434": {
      "llama3": 1542,
      "mistral": 876
    }
  }
}
```

### Token Statistics

```bash
curl http://localhost:12434/api/token_counts
```

Response:
```json
{
  "total_tokens": 2418,
  "breakdown": [
    {
      "endpoint": "http://endpoint1:11434",
      "model": "llama3",
      "input_tokens": 120,
      "output_tokens": 1422,
      "total_tokens": 1542
    },
    {
      "endpoint": "http://endpoint1:11434",
      "model": "mistral",
      "input_tokens": 80,
      "output_tokens": 796,
      "total_tokens": 876
    }
  ]
}
```

### Model Statistics

```bash
curl http://localhost:12434/api/stats -X POST -d '{"model": "llama3"}'
```

Response:
```json
{
  "model": "llama3",
  "input_tokens": 120,
  "output_tokens": 1422,
  "total_tokens": 1542,
  "time_series": [
    {
      "endpoint": "http://endpoint1:11434",
      "timestamp": 1712345600,
      "input_tokens": 20,
      "output_tokens": 150,
      "total_tokens": 170
    }
  ],
  "endpoint_distribution": {
    "http://endpoint1:11434": 1542
  }
}
```

### Configuration Status

```bash
curl http://localhost:12434/api/config
```

Response:
```json
{
  "endpoints": [
    {
      "url": "http://endpoint1:11434",
      "status": "ok" | "error",
      "version": "string" | "detail": "error message"
    }
  ]
}
```

### Real-time Usage Stream

```bash
curl http://localhost:12434/api/usage-stream
```

This provides Server-Sent Events (SSE) with real-time updates:
```
data: {"usage_counts": {...}, "token_usage_counts": {...}}

data: {"usage_counts": {...}, "token_usage_counts": {...}}
```

## Monitoring Tools

### Prometheus Integration

Create a Prometheus scrape configuration:

```yaml
scrape_configs:
  - job_name: 'nomyo-router'
    metrics_path: '/api/usage'
    params:
      format: ['prometheus']
    static_configs:
      - targets: ['nomyo-router:12434']
```

### Grafana Dashboard

Create a dashboard with these panels:
- Endpoint health status
- Current connection counts
- Token usage (input/output/total)
- Request rates
- Response times
- Error rates

### Logging

The router logs important events to stdout:
- Configuration loading
- Endpoint connection issues
- Token counting operations
- Error conditions

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Endpoint Unavailable

**Symptoms**:
- Health check shows endpoint as "error"
- Requests fail with connection errors

**Diagnosis**:
```bash
curl http://localhost:12434/health
curl http://localhost:12434/api/config
```

**Solutions**:
- Verify Ollama endpoint is running
- Check network connectivity
- Verify firewall rules
- Check DNS resolution
- Test direct connection: `curl http://endpoint:11434/api/version`

#### 2. Model Not Found

**Symptoms**:
- Error: "None of the configured endpoints advertise the model"
- Requests fail with model not found

**Diagnosis**:
```bash
curl http://localhost:12434/api/tags
curl http://endpoint:11434/api/tags
```

**Solutions**:
- Pull the model on the endpoint: `curl http://endpoint:11434/api/pull -d '{"name": "llama3"}'`
- Verify model name spelling
- Check if model is available on any endpoint
- For OpenAI endpoints, ensure model exists in their catalog

#### 3. High Latency

**Symptoms**:
- Slow response times
- Requests timing out

**Diagnosis**:
```bash
curl http://localhost:12434/api/usage
curl http://localhost:12434/api/config
```

**Solutions**:
- Check if endpoints are overloaded (high connection counts)
- Increase `max_concurrent_connections`
- Add more endpoints to the cluster
- Monitor Ollama endpoint performance
- Check network latency between router and endpoints

#### 4. Connection Limits Reached

**Symptoms**:
- Requests queuing
- High connection counts
- Slow response times

**Diagnosis**:
```bash
curl http://localhost:12434/api/usage
```

**Solutions**:
- Increase `max_concurrent_connections` in config.yaml
- Add more Ollama endpoints
- Scale your Ollama cluster
- Use MOE system for critical queries

#### 5. Token Tracking Not Working

**Symptoms**:
- Token counts not updating
- Database errors

**Diagnosis**:
```bash
ls -la token_counts.db
curl http://localhost:12434/api/token_counts
```

**Solutions**:
- Verify database file permissions
- Check if database path is writable
- Restart router to rebuild database
- Check disk space
- Verify environment variable `NOMYO_ROUTER_DB_PATH`

#### 6. Streaming Issues

**Symptoms**:
- Incomplete responses
- Connection resets during streaming
- Timeout errors

**Diagnosis**:
- Check router logs for errors
- Test with non-streaming requests
- Monitor connection counts

**Solutions**:
- Increase timeout settings
- Reduce `max_concurrent_connections`
- Check network stability
- Test with smaller payloads

### Error Messages

#### "Failed to connect to endpoint"

**Cause**: Network connectivity issue
**Action**: Verify endpoint is reachable, check firewall, test DNS

#### "None of the configured endpoints advertise the model"

**Cause**: Model not pulled on any endpoint
**Action**: Pull the model, verify model name

#### "Timed out waiting for endpoint"

**Cause**: Endpoint slow to respond
**Action**: Check endpoint health, increase timeouts

#### "Invalid JSON format in request body"

**Cause**: Malformed request
**Action**: Validate request payload, check API documentation

#### "Missing required field 'model'"

**Cause**: Request missing model parameter
**Action**: Add model parameter to request

### Performance Tuning

#### Optimizing Connection Handling

1. **Adjust concurrency limits**:
   ```yaml
   max_concurrent_connections: 4
   ```

2. **Monitor connection usage**:
   ```bash
   curl http://localhost:12434/api/usage
   ```

3. **Scale horizontally**:
   - Add more Ollama endpoints
   - Deploy multiple router instances

#### Reducing Latency

1. **Keep models loaded**:
   - Use frequently accessed models
   - Monitor `/api/ps` for loaded models

2. **Optimize endpoint selection**:
   - Distribute models across endpoints
   - Balance load evenly

3. **Use caching**:
   - Models cache (300s TTL)
   - Loaded models cache (30s TTL)

#### Memory Management

1. **Monitor memory usage**:
   - Token buffer grows with usage
   - Time-series data accumulates

2. **Adjust flush interval**:
   - Default: 10 seconds
   - Can be increased for less frequent I/O

3. **Database maintenance**:
   - Regular backups
   - Archive old data periodically

### Database Management

#### Viewing Token Data

```bash
sqlite3 token_counts.db "SELECT * FROM token_counts;"
sqlite3 token_counts.db "SELECT * FROM time_series LIMIT 100;"
```

#### Aggregating Old Data

```bash
curl http://localhost:12434/api/aggregate_time_series_days \
  -X POST \
  -d '{"days": 30, "trim_old": true}'
```

#### Backing Up Database

```bash
cp token_counts.db token_counts.db.backup
```

#### Restoring Database

```bash
cp token_counts.db.backup token_counts.db
```

### Advanced Troubleshooting

#### Debugging Endpoint Selection

Enable debug logging to see endpoint selection decisions:
```bash
uvicorn router:app --host 0.0.0.0 --port 12434 --log-level debug
```

#### Testing Individual Endpoints

```bash
# Test endpoint directly
curl http://endpoint:11434/api/version

# Test model availability
curl http://endpoint:11434/api/tags

# Test model loading
curl http://endpoint:11434/api/ps
```

#### Network Diagnostics

```bash
# Test connectivity
nc -zv endpoint 11434

# Test DNS resolution
dig endpoint

# Test latency
ping endpoint
```

### Common Pitfalls

1. **Using localhost in Docker**:
   - Inside Docker, `localhost` refers to the container itself
   - Use `host.docker.internal` or Docker service names

2. **Incorrect model names**:
   - Ollama: `llama3:latest`
   - OpenAI: `gpt-4` (no version suffix)

3. **Missing API keys**:
   - Remote endpoints require authentication
   - Set keys in config.yaml or environment variables

4. **Firewall blocking**:
   - Ensure port 11434 is open for Ollama
   - Ensure port 12434 is open for router

5. **Insufficient resources**:
   - Monitor CPU/memory on Ollama endpoints
   - Adjust `max_concurrent_connections` accordingly

## Best Practices

### Monitoring Setup

1. **Set up health checks**:
   - Monitor `/health` endpoint
   - Alert on status "error"

2. **Track usage metrics**:
   - Monitor connection counts
   - Track token usage
   - Watch for connection limits

3. **Log important events**:
   - Configuration changes
   - Endpoint failures
   - Recovery events

4. **Regular backups**:
   - Backup token_counts.db
   - Schedule regular backups
   - Test restore procedure

### Performance Monitoring

1. **Baseline metrics**:
   - Establish normal usage patterns
   - Track trends over time

2. **Alert thresholds**:
   - Set alerts for high connection counts
   - Monitor error rates
   - Watch for latency spikes

3. **Capacity planning**:
   - Track growth in token usage
   - Plan for scaling needs
   - Monitor resource utilization

### Incident Response

1. **Quick diagnosis**:
   - Check health endpoint first
   - Review recent logs
   - Verify endpoint status

2. **Isolation**:
   - Identify affected endpoints
   - Isolate problematic components
   - Fallback to healthy endpoints

3. **Recovery**:
   - Restart router if needed
   - Rebalance load
   - Restore from backup if necessary

## Examples

See the [examples](examples/) directory for monitoring configuration examples.
