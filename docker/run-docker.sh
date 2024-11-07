#!/bin/bash

docker build docker/ --tag="vectara-agentic-demo:latest"

if [ $? -eq 0 ]; then
  echo "Docker build successful."
else
  echo "Docker build failed. Please check the messages above. Exiting..."
  exit 4
fi

# remove old container if it exists
docker container inspect vectara-agentic-demo &>/dev/null && docker rm -f vectara-agentic-demo

# Run docker container
docker run -p 8000:8000 --name vectara-agentic-demo vectara-agentic-demo:latest

if [ $? -eq 0 ]; then
  echo "Success! vectara-agentic simple agent is running."
  echo "Go to http://localhost:8001 to access the vectara-agentic simple agent."
else
  echo "Vectara-agentic container failed to start. Please check the messages above."
fi