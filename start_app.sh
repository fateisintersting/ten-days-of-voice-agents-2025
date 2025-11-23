#!/bin/bash

# Start all services in background
livekit-server --dev &
(cd backend && uv run python src/agent.py dev) &
(cd frontend && npm run dev) &

# Wait for all background jobs
wait