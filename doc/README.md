# NOMYO Router Documentation

Welcome to the NOMYO Router documentation! This folder contains comprehensive guides for using, configuring, and deploying the NOMYO Router.

## Documentation Structure

```
doc/
├── architecture.md          # Technical architecture overview
├── configuration.md         # Detailed configuration guide
├── usage.md                 # API usage examples
├── deployment.md            # Deployment scenarios
├── monitoring.md            # Monitoring and troubleshooting
└── examples/                # Example configurations and scripts
    ├── docker-compose.yml
    ├── sample-config.yaml
    └── k8s-deployment.yaml
```

## Getting Started

### Quick Start Guide

1. **Install the router**:

   ```bash
   git clone https://github.com/nomyo-ai/nomyo-router.git
   cd nomyo-router
   python3 -m venv .venv/router
   source .venv/router/bin/activate
   pip3 install -r requirements.txt
   ```
2. **Configure endpoints** in `config.yaml`:

   ```yaml
   endpoints:
     - http://localhost:11434
   max_concurrent_connections: 2
   ```
3. **Run the router**:

   ```bash
   uvicorn router:app --host 0.0.0.0 --port 12434
   ```
4. **Use the router**: Point your frontend to `http://localhost:12434` instead of your Ollama instance.

### Key Features

- **Intelligent Routing**: Model deployment-aware routing with load balancing
- **Multi-Endpoint Support**: Combine Ollama and OpenAI-compatible endpoints
- **Token Tracking**: Comprehensive token usage monitoring
- **Real-time Monitoring**: Server-Sent Events for live usage updates
- **OpenAI Compatibility**: Full OpenAI API compatibility layer
- **MOE System**: Multiple Opinions Ensemble for improved responses with smaller models

## Documentation Guides

### [Architecture](architecture.md)

Learn about the router's internal architecture, routing algorithm, caching mechanisms, and advanced features like the MOE system.

### [Configuration](configuration.md)

Detailed guide on configuring the router with multiple endpoints, API keys, and environment variables.

### [Usage](usage.md)

Comprehensive API reference with examples for making requests, streaming responses, and using advanced features.

### [Deployment](deployment.md)

Step-by-step deployment guides for bare metal, Docker, Kubernetes, and production environments.

### [Monitoring](monitoring.md)

Monitoring endpoints, troubleshooting guides, performance tuning, and best practices for maintaining your router.

## Examples

The [examples](examples/) directory contains ready-to-use configuration files:

- **docker-compose.yml**: Complete Docker Compose setup with multiple Ollama instances
- **sample-config.yaml**: Example configuration with comments
- **k8s-deployment.yaml**: Kubernetes deployment manifests

## Need Help?

### Common Issues

Check the [Monitoring Guide](monitoring.md) for troubleshooting common problems:

- Endpoint unavailable
- Model not found
- High latency
- Connection limits reached
- Token tracking issues

### Support

For additional help:

1. Check the [GitHub Issues](https://github.com/nomyo-ai/nomyo-router/issues)
2. Review the [Monitoring Guide](monitoring.md) for diagnostics
3. Examine the router logs for detailed error messages

## Best Practices

### Configuration

- Use environment variables for API keys
- Set appropriate `max_concurrent_connections` based on your hardware
- Monitor endpoint health regularly
- Keep models loaded on multiple endpoints for redundancy

### Deployment

- Use Docker for containerized deployments
- Consider Kubernetes for production at scale
- Set up monitoring and alerting
- Implement regular backups of token counts database

### Performance

- Balance load across multiple endpoints
- Keep frequently used models loaded
- Monitor connection counts and token usage
- Scale horizontally when needed

## Next Steps

1. **Read the [Architecture Guide](architecture.md)** to understand how the router works
2. **Configure your endpoints** in `config.yaml`
3. **Deploy the router** using your preferred method
4. **Monitor your setup** using the monitoring endpoints
5. **Scale as needed** by adding more endpoints

Happy routing! 🚀
