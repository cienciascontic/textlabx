#!/usr/bin/env bash
uvicorn api:app --host 0.0.0.0 --port $PORT
