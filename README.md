# NOMYO Router

is a transparent proxy for ![Ollama](https://github.com/ollama/ollama) with model deployment aware routing.

It runs between your frontend application and Ollama backend and is transparent for both, the front- and backend.

![arch](https://github.com/user-attachments/assets/80b9feac-7d57-40f9-9cbc-78d0e76809c6)

Copy/Clone the repository, edit the config.yaml by adding your Ollama backend servers and the max_concurrent_connections setting per endpoint. This equals to your OLLAMA_NUM_PARALLEL config settings.

Run the NOMYO Router in a dedicated virtual environment, install the requirements and run with uvicorn:

```
python3 -m venv .venv/router
source .venv/router/bin/activate
pip3 install requirements.txt -r 
```
finally you can

```
uvicorn router:app --host 127.0.0.1 --port 12434
```

# Routing


