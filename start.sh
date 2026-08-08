#!/bin/bash
exec gunicorn \
  --workers 1 \
  --worker-class gthread \
  --threads 4 \
  --timeout 180 \
  --bind 0.0.0.0:${PORT:-5000} \
  app_server:app
