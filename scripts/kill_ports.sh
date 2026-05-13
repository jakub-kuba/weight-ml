#!/bin/bash
# Kill all processes using ports needed by this project
for port in 8000 8501 5000 9090 3000; do
    pid=$(lsof -ti :$port)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)"
        kill -9 $pid
    fi
done
echo "All ports cleared."