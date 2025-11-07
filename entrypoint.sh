#!/usr/bin/env sh
set -e

CONFIG_PATH_ARG=""
SHOW_HELP=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --config-path)
            if [ -z "${2:-}" ]; then
                echo "Error: --config-path requires a value." >&2
                exit 1
            fi
            CONFIG_PATH_ARG="$2"
            shift 2
            ;;
        --config-path=*)
            CONFIG_PATH_ARG="${1#*=}"
            shift 1
            ;;
        -h|--help)
            SHOW_HELP=1
            shift 1
            ;;
        --)
            shift 1
            break
            ;;
        *)
            break
            ;;
    esac
done

if [ "$SHOW_HELP" -eq 1 ]; then
    cat <<'EOF'
Usage: entrypoint.sh [--config-path /path/to/config.yaml] [uvicorn options...]

Options:
  --config-path PATH    Absolute or relative path to a NOMYO Router YAML config file.
  -h, --help            Show this help message and exit.

Any arguments that remain after the options above are passed directly to uvicorn.

Environment variables:
  CONFIG_PATH               Alternative way to specify the config path.
  NOMYO_ROUTER_CONFIG_PATH  Overrides the config path (same as --config-path).
  UVICORN_HOST              Host interface to bind to (default: 0.0.0.0).
  UVICORN_PORT              Port to listen on (default: 12434).
  UVICORN_RELOAD            If set, enables --reload for uvicorn (useful for local dev).
  UVICORN_BIN               Path to the uvicorn executable (default: uvicorn).
EOF
    exit 0
fi

if [ -z "$CONFIG_PATH_ARG" ] && [ -n "${NOMYO_ROUTER_CONFIG_PATH:-}" ]; then
    CONFIG_PATH_ARG="$NOMYO_ROUTER_CONFIG_PATH"
fi

if [ -z "$CONFIG_PATH_ARG" ] && [ -n "${CONFIG_PATH:-}" ]; then
    CONFIG_PATH_ARG="$CONFIG_PATH"
fi

if [ -n "$CONFIG_PATH_ARG" ]; then
    export NOMYO_ROUTER_CONFIG_PATH="$CONFIG_PATH_ARG"
fi

UVICORN_BIN="${UVICORN_BIN:-uvicorn}"
UVICORN_HOST="${UVICORN_HOST:-0.0.0.0}"
UVICORN_PORT="${UVICORN_PORT:-12434}"

ADD_DEFAULTS=0
if [ "$#" -eq 0 ]; then
    set -- "$UVICORN_BIN" "router:app"
    ADD_DEFAULTS=1
elif [ "${1#-}" != "$1" ]; then
    set -- "$UVICORN_BIN" "router:app" "$@"
    ADD_DEFAULTS=1
elif [ "$1" = "$UVICORN_BIN" ]; then
    ADD_DEFAULTS=1
fi

if [ "$ADD_DEFAULTS" -eq 1 ]; then
    NEED_HOST=1
    NEED_PORT=1
    for arg in "$@"; do
        case "$arg" in
            --host|--host=*)
                NEED_HOST=0
                ;;
            --port|--port=*)
                NEED_PORT=0
                ;;
        esac
    done
    if [ "$NEED_HOST" -eq 1 ]; then
        set -- "$@" "--host" "$UVICORN_HOST"
    fi
    if [ "$NEED_PORT" -eq 1 ]; then
        set -- "$@" "--port" "$UVICORN_PORT"
    fi
    if [ -n "${UVICORN_RELOAD:-}" ]; then
        set -- "$@" "--reload"
    fi
fi

exec "$@"
