# NOMYO Router

is a transparent proxy for [Ollama](https://github.com/ollama/ollama) with model deployment aware routing.

<img width="2490" height="1298" alt="Screenshot_NOMYO_Router_0-2-2_Dashboard" src="https://github.com/user-attachments/assets/ddacdf88-e3f3-41dd-8be6-f165b22d9879" /><br>

It runs between your frontend application and Ollama backend and is transparent for both, the front- and backend.

![arch](https://github.com/user-attachments/assets/1e0064ab-de54-4226-8a15-c0fcca64704c)

# Installation

Copy/Clone the repository, edit the config.yaml by adding your Ollama backend servers and the max_concurrent_connections setting per endpoint. This equals to your OLLAMA_NUM_PARALLEL config settings.

```
# config.yaml
endpoints:
  - http://ollama0:11434
  - http://ollama1:11434
  - http://ollama2:11434
  - https://api.openai.com/v1

# Maximum concurrent connections *per endpoint‑model pair*
max_concurrent_connections: 2

# API keys for remote endpoints
# Set an environment variable like OPENAI_KEY
# Confirm endpoints are exactly as in endpoints block
api_keys:
  "http://192.168.0.50:11434": "ollama"
  "http://192.168.0.51:11434": "ollama"
  "http://192.168.0.52:11434": "ollama"
  "https://api.openai.com/v1": "${OPENAI_KEY}"
```

Run the NOMYO Router in a dedicated virtual environment, install the requirements and run with uvicorn:

```
python3 -m venv .venv/router
source .venv/router/bin/activate
pip3 install -r requirements.txt
```

[optional] on the shell do:

```
export OPENAI_KEY=YOUR_SECRET_API_KEY
```

finally you can

```
uvicorn router:app --host 127.0.0.1 --port 12434
```

## Docker Deployment

Build the container image locally:

```sh
docker build -t nomyo-router .
```

Run the router in Docker with your own configuration file mounted from the host. The entrypoint script accepts a `--config-path` argument so you can point to a file anywhere inside the container:

```sh
docker run -d \
  --name nomyo-router \
  -p 12434:12434 \
  -v /absolute/path/to/config_folder:/app/config/ \
  -e CONFIG_PATH /app/config/config.yaml
  nomyo-router \
```

Notes:
- `-e CONFIG_PATH` sets the `NOMYO_ROUTER_CONFIG_PATH` environment variable under the hood; you can export it directly instead if you prefer.
- To override the bind address or port, export `UVICORN_HOST` or `UVICORN_PORT`, or pass the corresponding uvicorn flags after `--`, e.g. `nomyo-router --config-path /config/config.yaml -- --port 9000`.
- Use `docker logs nomyo-router` to confirm the loaded endpoints and concurrency settings at startup.

# Routing

NOMYO Router accepts any Ollama request on the configured port for any Ollama endpoint from your frontend application. It then checks the available backends for the specific request.
When the request is embed(dings), chat or generate the request will be forwarded to a single Ollama server, answered and send back to the router which forwards it back to the frontend.

If another request for the same model config is made, NOMYO Router is aware which model runs on which Ollama server and routes the request to an Ollama server where this model is already deployed.

If at the same time there are more than max concurrent connections than configured, NOMYO Router will route this request to another Ollama server serving the requested model and having the least connections for fastest completion.

This way the Ollama backend servers are utilized more efficient than by simply using a wheighted, round-robin or least-connection approach.

![routing](https://github.com/user-attachments/assets/ed05dfbb-fcc8-4ff2-b8ca-3cdce2660c9f)

NOMYO Router also supports OpenAI API compatible v1 backend servers.
