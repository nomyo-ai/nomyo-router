# Deployment Guide

## Deployment Options

NOMYO Router can be deployed in various environments depending on your requirements.

## 1. Bare Metal / VM Deployment

### Prerequisites

- Python 3.10+
- pip
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/nomyo-ai/nomyo-router.git
cd nomyo-router

# Create virtual environment
python3 -m venv .venv/router
source .venv/router/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# Configure endpoints
nano config.yaml
```

### Running the Router

```bash
# Basic startup
uvicorn router:app --host 0.0.0.0 --port 12434

# With custom configuration path
export NOMYO_ROUTER_CONFIG_PATH=/etc/nomyo-router/config.yaml
uvicorn router:app --host 0.0.0.0 --port 12434

# With custom database path
export NOMYO_ROUTER_DB_PATH=/var/lib/nomyo-router/token_counts.db
uvicorn router:app --host 0.0.0.0 --port 12434
```

### Systemd Service

Create `/etc/systemd/system/nomyo-router.service`:

```ini
[Unit]
Description=NOMYO Router - Ollama Proxy
After=network.target

[Service]
User=nomyo
Group=nomyo
WorkingDirectory=/opt/nomyo-router
Environment="NOMYO_ROUTER_CONFIG_PATH=/etc/nomyo-router/config.yaml"
Environment="NOMYO_ROUTER_DB_PATH=/var/lib/nomyo-router/token_counts.db"
ExecStart=/opt/nomyo-router/.venv/router/bin/uvicorn router:app --host 0.0.0.0 --port 12434
Restart=always
RestartSec=10
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=nomyo-router

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable nomyo-router
sudo systemctl start nomyo-router
sudo systemctl status nomyo-router
```

## 2. Docker Deployment

### Build the Image

```bash
docker build -t nomyo-router .
```

### Run the Container

```bash
docker run -d \
  --name nomyo-router \
  -p 12434:12434 \
  -v /absolute/path/to/config_folder:/app/config/ \
  -e CONFIG_PATH=/app/config/config.yaml \
  nomyo-router
```

### Advanced Docker Configuration

#### Custom Port

```bash
docker run -d \
  --name nomyo-router \
  -p 9000:12434 \
  -v /path/to/config:/app/config/ \
  -e CONFIG_PATH=/app/config/config.yaml \
  nomyo-router \
  -- --port 9000
```

#### Custom Host

```bash
docker run -d \
  --name nomyo-router \
  -p 12434:12434 \
  -v /path/to/config:/app/config/ \
  -e CONFIG_PATH=/app/config/config.yaml \
  -e UVICORN_HOST=0.0.0.0 \
  nomyo-router
```

#### Persistent Database

```bash
docker run -d \
  --name nomyo-router \
  -p 12434:12434 \
  -v /path/to/config:/app/config/ \
  -v /path/to/db:/app/token_counts.db \
  -e CONFIG_PATH=/app/config/config.yaml \
  -e NOMYO_ROUTER_DB_PATH=/app/token_counts.db \
  nomyo-router
```

### Docker Compose Example

See [examples/docker-compose.yml](examples/docker-compose.yml) for a complete Docker Compose example.

## 3. Kubernetes Deployment

### Prerequisites

- Kubernetes cluster
- kubectl configured
- Helm (optional)

### Basic Deployment

Create `nomyo-router-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nomyo-router
  labels:
    app: nomyo-router
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nomyo-router
  template:
    metadata:
      labels:
        app: nomyo-router
    spec:
      containers:
      - name: nomyo-router
        image: nomyo-router:latest
        ports:
        - containerPort: 12434
        env:
        - name: CONFIG_PATH
          value: "/app/config/config.yaml"
        - name: NOMYO_ROUTER_DB_PATH
          value: "/app/token_counts.db"
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
        - name: db-volume
          mountPath: /app/token_counts.db
          subPath: token_counts.db
      volumes:
      - name: config-volume
        configMap:
          name: nomyo-router-config
      - name: db-volume
        persistentVolumeClaim:
          claimName: nomyo-router-db-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: nomyo-router
spec:
  selector:
    app: nomyo-router
  ports:
    - protocol: TCP
      port: 80
      targetPort: 12434
  type: LoadBalancer
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: nomyo-router-config
data:
  config.yaml: |
    endpoints:
      - http://ollama-service:11434
    max_concurrent_connections: 2
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nomyo-router-db-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

Apply the deployment:

```bash
kubectl apply -f nomyo-router-deployment.yaml
```

### Horizontal Pod Autoscaler

Create `nomyo-router-hpa.yaml`:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nomyo-router-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nomyo-router
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

Apply the HPA:

```bash
kubectl apply -f nomyo-router-hpa.yaml
```

## 4. Production Deployment

### High Availability Setup

For production environments with multiple Ollama instances:

```yaml
# config.yaml
endpoints:
  - http://ollama-worker1:11434
  - http://ollama-worker2:11434
  - http://ollama-worker3:11434
  - https://api.openai.com/v1

max_concurrent_connections: 4

api_keys:
  "https://api.openai.com/v1": "${OPENAI_KEY}"
```

### Load Balancing

Deploy multiple router instances behind a load balancer:

```
┌───────────────────────────────────────────────────────────────┐
│                     Load Balancer (NGINX, Traefik)             │
└───────────────────────────────────────────────────────────────┘
                        │
                        ├─┬───────────────────────────────────────┐
                        │   │                                   │
                        ▼   ▼                                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│  Router Instance │ │  Router Instance │ │  Router Instance        │
│  (Pod 1)        │ │  (Pod 2)        │ │  (Pod 3)               │
└─────────────────┘ └─────────────────┘ └─────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────┐
│                     Ollama Cluster                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────────┐  │
│  │ Ollama      │ │ Ollama      │ │ OpenAI API                 │  │
│  │ Worker 1    │ │ Worker 2    │ │ (Fallback)                 │  │
│  └─────────────┘ └─────────────┘ └─────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Monitoring and Logging

#### Prometheus Monitoring

Create a Prometheus scrape configuration:

```yaml
scrape_configs:
  - job_name: 'nomyo-router'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['nomyo-router:12434']
```

#### Logging

Configure log aggregation:

```bash
# In Docker
docker run -d \
  --name nomyo-router \
  -p 12434:12434 \
  -v /path/to/config:/app/config/ \
  -e CONFIG_PATH=/app/config/config.yaml \
  --log-driver=fluentd \
  --log-opt fluentd-address=fluentd:24224 \
  nomyo-router
```

## Deployment Checklist

### Pre-Deployment

- [ ] Configure all Ollama endpoints
- [ ] Set appropriate `max_concurrent_connections`
- [ ] Configure API keys for remote endpoints
- [ ] Test configuration locally
- [ ] Set up monitoring and alerting
- [ ] Configure logging
- [ ] Set up backup for token counts database

### Post-Deployment

- [ ] Verify health endpoint: `curl http://<router>/health`
- [ ] Check endpoint status: `curl http://<router>/api/config`
- [ ] Monitor connection counts: `curl http://<router>/api/usage`
- [ ] Set up regular backups
- [ ] Configure auto-restart on failure
- [ ] Monitor performance metrics

## Scaling Guidelines

### Vertical Scaling

- Increase `max_concurrent_connections` for more parallel requests
- Add more CPU/memory to the router instance
- Monitor memory usage (token buffer grows with usage)

### Horizontal Scaling

- Deploy multiple router instances
- Use a load balancer to distribute traffic
- Each instance maintains its own connection tracking
- Database can be shared or per-instance

### Database Considerations

- SQLite is sufficient for single-instance deployments
- For multi-instance deployments, consider PostgreSQL
- Regular backups are recommended
- Database size grows with token usage history

## Security Best Practices

### Network Security

- Use TLS for all external connections
- Restrict access to router port (12434)
- Use firewall rules to limit access
- Consider using VPN for internal communications

### Configuration Security

- Store API keys in environment variables
- Restrict access to config.yaml
- Use secrets management for production deployments
- Rotate API keys regularly

### Runtime Security

- Run as non-root user
- Set appropriate file permissions
- Monitor for suspicious activity
- Keep dependencies updated

## Troubleshooting Deployment Issues

### Common Issues

**Problem**: Router not starting

- **Check**: Logs for configuration errors
- **Solution**: Validate config.yaml syntax

**Problem**: Endpoints showing as unavailable

- **Check**: Network connectivity from router to endpoints
- **Solution**: Verify firewall rules and DNS resolution

**Problem**: High latency

- **Check**: Endpoint health and connection counts
- **Solution**: Add more endpoints or increase concurrency limits

**Problem**: Database errors

- **Check**: Database file permissions
- **Solution**: Ensure write permissions for the database path

**Problem**: Connection limits being hit

- **Check**: `/api/usage` endpoint
- **Solution**: Increase `max_concurrent_connections` or add endpoints

## Examples

See the [examples](examples/) directory for ready-to-use deployment examples.
