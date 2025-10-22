# Docker Quick Reference

A quick cheatsheet for running the Customer Segmentation pipeline with Docker.

## Common Commands

### Build & Setup
```bash
# Build the Docker image
docker build -t customer-segmentation:latest .

# Or use helper script
./docker-run.sh build     # Linux/Mac
docker-run.bat build      # Windows
make build                # Using Makefile
```

### Run Training
```bash
# Quick training run
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  customer-segmentation:latest

# Or use shortcuts
./docker-run.sh train     # Linux/Mac
docker-run.bat train      # Windows
make train                # Using Makefile
docker-compose up train   # Docker Compose
```

### Make Predictions
```bash
# Generate predictions
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  customer-segmentation:latest \
  python predict.py

# Or use shortcuts
./docker-run.sh predict
make predict
docker-compose up predict
```

### Jupyter Notebook
```bash
# Start Jupyter (access at http://localhost:8888)
docker run --rm -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/notebooks:/app/notebooks \
  customer-segmentation:latest \
  jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

# Or use shortcuts
./docker-run.sh jupyter
make jupyter
docker-compose up jupyter
```

### Development Shell
```bash
# Interactive bash shell
docker run --rm -it \
  -v $(pwd):/app \
  customer-segmentation:latest \
  bash

# Or use shortcuts
./docker-run.sh dev
make dev
docker-compose up dev
```

## Volume Mounts Explained

| Mount | Purpose |
|-------|---------|
| `-v $(pwd)/data:/app/data` | Input/output data files |
| `-v $(pwd)/models:/app/models` | Trained model files |
| `-v $(pwd)/results:/app/results` | Results and visualizations |
| `-v $(pwd)/notebooks:/app/notebooks` | Jupyter notebooks |
| `-v $(pwd)/src:/app/src` | Source code (dev only) |

## Container Management

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a running container
docker stop customer-segmentation-train

# Remove a container
docker rm customer-segmentation-train

# View container logs
docker logs customer-segmentation-train

# Follow logs in real-time
docker logs -f customer-segmentation-train

# Execute command in running container
docker exec -it customer-segmentation-train bash
```

## Image Management

```bash
# List images
docker images

# Remove image
docker rmi customer-segmentation:latest

# Check image size
docker images customer-segmentation:latest

# View image layers
docker history customer-segmentation:latest

# Tag image
docker tag customer-segmentation:latest myrepo/customer-segmentation:v1.0
```

## Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove all unused data
docker system prune

# Remove everything (including volumes)
docker system prune -a --volumes

# Or use helper scripts
./docker-run.sh clean
make clean
```

## Troubleshooting

```bash
# Build without cache
docker build --no-cache -t customer-segmentation:latest .

# Check resource usage
docker stats

# Inspect container
docker inspect customer-segmentation-train

# View disk usage
docker system df

# Pull fresh base image
docker pull python:3.11-slim
```

## Environment Variables

```bash
# Set environment variables
docker run --rm \
  -e TF_CPP_MIN_LOG_LEVEL=3 \
  -e PYTHONUNBUFFERED=1 \
  customer-segmentation:latest

# Or use env file
docker run --rm --env-file .env customer-segmentation:latest
```

## Windows-Specific Notes

```bash
# Use forward slashes for paths
docker run -v C:/Users/YourName/project/data:/app/data ...

# Or use PowerShell with ${PWD}
docker run -v ${PWD}/data:/app/data ...

# Batch script
docker-run.bat build
docker-run.bat train
```

## Docker Compose Commands

```bash
# Start service
docker-compose up train

# Run in background
docker-compose up -d jupyter

# View logs
docker-compose logs -f train

# Stop services
docker-compose down

# Rebuild and start
docker-compose up --build train

# Remove volumes
docker-compose down -v

# List services
docker-compose ps
```

## Makefile Targets

```bash
make help          # Show all available commands
make build         # Build image
make train         # Run training
make predict       # Run predictions
make jupyter       # Start Jupyter
make dev           # Development shell
make test          # Quick test
make clean         # Cleanup
make ps            # List containers
make logs          # View logs
make size          # Show image size
```

## Tips & Tricks

### Run as Non-Root User
```bash
docker run --user $(id -u):$(id -g) ...
```

### Use GPU (NVIDIA)
```bash
docker run --gpus all customer-segmentation:latest
```

### Limit Resources
```bash
docker run \
  --memory="4g" \
  --cpus="2.0" \
  customer-segmentation:latest
```

### Mount Current Directory
```bash
# Linux/Mac
docker run -v $(pwd):/app ...

# Windows PowerShell
docker run -v ${PWD}:/app ...

# Windows CMD
docker run -v %cd%:/app ...
```

### Keep Container Running
```bash
# For debugging
docker run -d --name debug customer-segmentation:latest tail -f /dev/null
docker exec -it debug bash
```

### Copy Files From Container
```bash
# Copy trained model out
docker cp customer-segmentation-train:/app/models/autoencoder.h5 ./
```

## One-Liners

```bash
# Complete pipeline
docker build -t customer-segmentation:latest . && docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/results:/app/results customer-segmentation:latest

# Quick test
docker run --rm customer-segmentation:latest python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} OK')"

# Check Python packages
docker run --rm customer-segmentation:latest pip list

# View help
docker run --rm customer-segmentation:latest python train_pipeline.py --help
```

## Common Workflows

### First Time Setup
```bash
./docker-run.sh build
./docker-run.sh train
ls results/
```

### Development Cycle
```bash
# Edit code on host
vim train_pipeline.py

# Test changes
docker run --rm -v $(pwd):/app customer-segmentation:latest python train_pipeline.py

# If satisfied, rebuild
docker build -t customer-segmentation:latest .
```

### Production Deployment
```bash
# Tag with version
docker tag customer-segmentation:latest customer-segmentation:1.0.0

# Save image
docker save customer-segmentation:1.0.0 | gzip > customer-segmentation-1.0.0.tar.gz

# Load on another machine
gunzip -c customer-segmentation-1.0.0.tar.gz | docker load
```

## Links

- [DOCKER.md](DOCKER.md) - Full Docker documentation
- [README.md](README.md) - Project documentation
- [Docker Documentation](https://docs.docker.com/)
