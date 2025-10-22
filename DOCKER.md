# Docker Guide for Customer Segmentation

This guide explains how to use Docker to run the customer segmentation pipeline in an isolated, reproducible environment.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Usage Methods](#usage-methods)
- [Available Commands](#available-commands)
- [Docker Compose](#docker-compose)
- [Volumes and Data Persistence](#volumes-and-data-persistence)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. **Docker**: Install Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. **Docker Compose** (optional): Usually included with Docker Desktop

Verify installation:
```bash
docker --version
docker-compose --version
```

## Quick Start

### Option 1: Using Helper Script (Recommended)

**Linux/Mac:**
```bash
# Build the image
./docker-run.sh build

# Run training
./docker-run.sh train

# View results
ls -l results/
```

**Windows:**
```cmd
REM Build the image
docker-run.bat build

REM Run training
docker-run.bat train

REM View results
dir results\
```

### Option 2: Using Makefile

```bash
# Build the image
make build

# Run training
make train

# View all available commands
make help
```

### Option 3: Using Docker Compose

```bash
# Build and run training
docker-compose up train

# Run predictions
docker-compose up predict

# Start Jupyter notebook
docker-compose up jupyter
```

### Option 4: Direct Docker Commands

```bash
# Build image
docker build -t customer-segmentation:latest .

# Run training
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  customer-segmentation:latest

# Run predictions
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  customer-segmentation:latest \
  python predict.py
```

## Usage Methods

### 1. Helper Scripts

The project includes helper scripts for easy Docker management:

**Linux/Mac: `docker-run.sh`**
```bash
./docker-run.sh [COMMAND]

Commands:
  build       Build the Docker image
  train       Run the training pipeline
  predict     Run predictions on test data
  jupyter     Start Jupyter notebook server
  dev         Start interactive shell for development
  clean       Remove containers and images
  help        Show help message
```

**Windows: `docker-run.bat`**
```cmd
docker-run.bat [COMMAND]

Same commands as Linux/Mac version
```

### 2. Makefile

For systems with `make` installed:

```bash
make help          # Show all available commands
make build         # Build Docker image
make train         # Run training
make predict       # Run predictions
make jupyter       # Start Jupyter notebook
make dev           # Start development shell
make clean         # Clean up containers and images
make test          # Run quick test
```

### 3. Docker Compose

Use `docker-compose.yml` for multi-service management:

```bash
# Individual services
docker-compose up train      # Run training
docker-compose up predict    # Run predictions
docker-compose up jupyter    # Start Jupyter
docker-compose up dev        # Development shell

# Stop services
docker-compose down

# View logs
docker-compose logs train
```

## Available Commands

### Build Image

Builds the Docker image with all dependencies:

```bash
# Using script
./docker-run.sh build

# Using make
make build

# Using docker directly
docker build -t customer-segmentation:latest .
```

**Image size:** Approximately 1.5-2GB (includes Python, TensorFlow, and all dependencies)

### Run Training Pipeline

Trains the Auto Encoder and generates customer segments:

```bash
# Using script
./docker-run.sh train

# Using make
make train

# Using docker-compose
docker-compose up train
```

**Output:**
- Trained models in `models/`
- Results in `results/`
- Visualizations in `results/plots/`

**Duration:** ~10-20 minutes depending on hardware

### Run Predictions

Generate predictions for test data:

```bash
# Using script
./docker-run.sh predict

# Using make
make predict

# Using docker-compose
docker-compose up predict
```

**Requirements:** Trained models must exist in `models/` directory

### Start Jupyter Notebook

Launch Jupyter for interactive analysis:

```bash
# Using script
./docker-run.sh jupyter

# Using make
make jupyter

# Using docker-compose
docker-compose up jupyter
```

**Access:** Open browser to http://localhost:8888

**Note:** Jupyter runs without authentication by default. For production, configure password protection.

### Development Shell

Start an interactive bash shell inside the container:

```bash
# Using script
./docker-run.sh dev

# Using make
make dev

# Using docker-compose
docker-compose up dev
```

**Usage:**
```bash
# Inside container
python train_pipeline.py
python predict.py
python -c "import src.autoencoder as ae; print(ae.__file__)"
exit  # to leave container
```

### Clean Up

Remove Docker containers and images:

```bash
# Using script
./docker-run.sh clean

# Using make
make clean

# Manual cleanup
docker rm -f $(docker ps -a -q --filter ancestor=customer-segmentation)
docker rmi customer-segmentation:latest
```

## Docker Compose

The `docker-compose.yml` file defines multiple services:

### Services

1. **train** - Runs the training pipeline
2. **predict** - Generates predictions
3. **jupyter** - Jupyter notebook server
4. **dev** - Development environment

### Usage Examples

```bash
# Start specific service
docker-compose up [service-name]

# Run in background
docker-compose up -d jupyter

# View logs
docker-compose logs -f train

# Stop all services
docker-compose down

# Rebuild and start
docker-compose up --build train

# Remove volumes
docker-compose down -v
```

## Volumes and Data Persistence

### Volume Mounts

The Docker setup uses volume mounts to persist data:

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Input datasets |
| `./models` | `/app/models` | Trained models |
| `./results` | `/app/results` | Output predictions and plots |
| `./notebooks` | `/app/notebooks` | Jupyter notebooks |
| `./src` | `/app/src` | Source code (dev mode) |

### Data Flow

```
Host Machine                Docker Container
===========                 ================

./data/raw/Train.csv   -->  /app/data/raw/Train.csv
./data/raw/Test.csv    -->  /app/data/raw/Test.csv

                       <--  /app/models/autoencoder.h5
                       <--  /app/models/preprocessor.pkl
                       <--  /app/models/clusterer.pkl

                       <--  /app/results/submission.csv
                       <--  /app/results/plots/*.png
```

### Ensuring Data Availability

**Option 1: Include data in image (during build)**
- Uncomment data copy lines in Dockerfile
- Data will be embedded in image (larger image size)

**Option 2: Mount data as volume (recommended)**
- Keep data on host machine
- Mount as volume at runtime (smaller image, flexible)

```bash
# Ensure data is in the right place
cp Kaggle/*.csv data/raw/

# Or use make command
make pull-data
```

## Troubleshooting

### Issue: "No such file or directory" when running container

**Solution:** Ensure data files exist in `data/raw/`:
```bash
ls -l data/raw/
# Should show Train.csv and Test.csv
```

### Issue: "Permission denied" errors

**Solution:** Fix file permissions:
```bash
# Linux/Mac
chmod -R 755 data/ models/ results/

# Or run container as current user
docker run --user $(id -u):$(id -g) ...
```

### Issue: Container runs out of memory

**Solution:** Increase Docker memory limit:
- Docker Desktop → Settings → Resources → Memory
- Recommended: At least 4GB RAM

### Issue: TensorFlow warnings/errors

**Solution:** TensorFlow warnings are usually harmless. To suppress:
```bash
# Already set in Dockerfile
ENV TF_CPP_MIN_LOG_LEVEL=2
```

### Issue: "Port 8888 is already in use" (Jupyter)

**Solution:** Stop other Jupyter instances or use different port:
```bash
docker run -p 8889:8888 ...  # Use port 8889 instead
```

### Issue: Slow performance on Windows

**Solution:**
- Use WSL2 backend for Docker Desktop
- Store project files in WSL2 filesystem (not Windows filesystem)
- See: https://docs.docker.com/desktop/windows/wsl/

### Issue: Image build fails

**Solution:** Common fixes:
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t customer-segmentation:latest .

# Check Docker disk space
docker system df
```

### Issue: Cannot access results after running container

**Solution:** Ensure volume mounts are correct:
```bash
# Use absolute paths if relative paths don't work
docker run -v /full/path/to/results:/app/results ...

# On Windows, use forward slashes
docker run -v C:/Users/YourName/project/results:/app/results ...
```

## Advanced Usage

### Custom Configuration

Modify training parameters:
```bash
# Edit train_pipeline.py, then rebuild
docker build -t customer-segmentation:latest .

# Or mount source code and edit on host
docker run -v $(pwd)/train_pipeline.py:/app/train_pipeline.py ...
```

### Using GPU (NVIDIA)

To use GPU acceleration:

1. Install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)

2. Run with GPU:
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  customer-segmentation:latest
```

3. Or update docker-compose.yml:
```yaml
services:
  train:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Pushing to Docker Registry

Share your image:

```bash
# Tag image
docker tag customer-segmentation:latest yourusername/customer-segmentation:latest

# Push to Docker Hub
docker login
docker push yourusername/customer-segmentation:latest

# Pull on another machine
docker pull yourusername/customer-segmentation:latest
```

### Multi-stage Build (Optimization)

The Dockerfile can be optimized with multi-stage builds:

```dockerfile
# Builder stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
WORKDIR /app
COPY . .
CMD ["python", "train_pipeline.py"]
```

## Best Practices

1. **Always use volume mounts** for data, models, and results
2. **Don't store sensitive data** in Docker images
3. **Use .dockerignore** to exclude unnecessary files
4. **Tag images** with version numbers for production
5. **Monitor resource usage** with `docker stats`
6. **Clean up regularly** with `docker system prune`
7. **Use specific Python version** (not `latest`) for reproducibility
8. **Document dependencies** in requirements.txt with versions

## Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [TensorFlow Docker Guide](https://www.tensorflow.org/install/docker)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

## Support

For issues related to:
- Docker setup: Check this guide and Docker documentation
- Code/model: Check main README.md
- Data: Check SETUP_INSTRUCTIONS.md
