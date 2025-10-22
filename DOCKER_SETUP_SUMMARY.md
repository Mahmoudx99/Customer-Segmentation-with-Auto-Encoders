# Docker Setup Summary

This document summarizes the Docker configuration for the Customer Segmentation project.

## What Was Created

### Core Docker Files

1. **Dockerfile**
   - Base image: `python:3.11-slim`
   - Installs all Python dependencies from `requirements.txt`
   - Copies project source code
   - Creates necessary directories
   - Default command: runs `train_pipeline.py`
   - Image size: ~1.5-2GB

2. **.dockerignore**
   - Excludes unnecessary files from Docker build context
   - Reduces build time and image size
   - Excludes: Python cache, IDE files, generated models, results

3. **docker-compose.yml**
   - Defines 4 services:
     - `train`: Runs training pipeline
     - `predict`: Generates predictions
     - `jupyter`: Jupyter notebook server on port 8888
     - `dev`: Interactive development shell
   - Configures volume mounts for data persistence
   - Sets environment variables

### Helper Scripts

4. **docker-run.sh** (Linux/Mac)
   - Executable bash script
   - Commands: build, train, predict, jupyter, dev, clean, help
   - Color-coded output
   - Automatic error handling

5. **docker-run.bat** (Windows)
   - Windows batch script
   - Same commands as Linux/Mac version
   - Compatible with Windows CMD

6. **Makefile**
   - Convenient make targets for Docker operations
   - 15+ commands including build, train, predict, jupyter, dev, clean
   - Works on Linux/Mac with `make` installed
   - Includes docker-compose shortcuts

### Documentation

7. **DOCKER.md** (11,000+ words)
   - Complete Docker guide
   - Sections:
     - Prerequisites
     - Quick Start (4 different methods)
     - Usage Methods
     - Available Commands (detailed)
     - Docker Compose
     - Volumes and Data Persistence
     - Troubleshooting (10+ common issues)
     - Advanced Usage (GPU, registry, optimization)
     - Best Practices

8. **DOCKER_QUICKREF.md**
   - Quick reference cheatsheet
   - Common commands with examples
   - Volume mounts explained
   - Container/image management
   - Cleanup commands
   - One-liners and workflows

9. **README.md** (Updated)
   - Added "Quick Start with Docker" section
   - Updated project structure to show Docker files
   - Links to detailed Docker documentation

## How It Works

### Docker Image Architecture

```
Base: python:3.11-slim (Debian)
├── System packages (build-essential, git, curl)
├── Python packages (from requirements.txt)
│   ├── TensorFlow 2.13+
│   ├── Scikit-learn
│   ├── Pandas, NumPy
│   ├── Matplotlib, Seaborn
│   └── Jupyter
├── Source code (src/)
├── Scripts (train_pipeline.py, predict.py)
├── Notebooks (notebooks/)
└── Directories (data/, models/, results/)
```

### Volume Mounts (Data Persistence)

```
Host Machine          Docker Container
============          ================
./data           -->  /app/data
./models         <->  /app/models
./results        <--  /app/results
./notebooks      <->  /app/notebooks
./src            <->  /app/src (dev mode only)
```

This ensures:
- Data files are accessible inside container
- Trained models are saved to host
- Results persist after container stops
- Code changes reflect immediately (dev mode)

## Usage Examples

### Method 1: Helper Scripts (Easiest)

**Linux/Mac:**
```bash
./docker-run.sh build     # Build image (5-10 min first time)
./docker-run.sh train     # Train model (10-20 min)
./docker-run.sh predict   # Generate predictions
./docker-run.sh jupyter   # Start Jupyter (port 8888)
./docker-run.sh dev       # Development shell
./docker-run.sh clean     # Remove containers/images
```

**Windows:**
```cmd
docker-run.bat build
docker-run.bat train
docker-run.bat predict
docker-run.bat jupyter
docker-run.bat dev
docker-run.bat clean
```

### Method 2: Makefile (Convenient)

```bash
make build         # Build Docker image
make train         # Run training
make predict       # Run predictions
make jupyter       # Start Jupyter
make dev           # Development shell
make clean         # Cleanup
make test          # Quick test
make help          # Show all commands
```

### Method 3: Docker Compose (Multi-service)

```bash
docker-compose up train      # Run training
docker-compose up predict    # Run predictions
docker-compose up jupyter    # Start Jupyter
docker-compose up dev        # Development shell
docker-compose down          # Stop services
docker-compose logs train    # View logs
```

### Method 4: Direct Docker Commands (Manual)

```bash
# Build
docker build -t customer-segmentation:latest .

# Train
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  customer-segmentation:latest

# Predict
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  customer-segmentation:latest \
  python predict.py

# Jupyter
docker run --rm -p 8888:8888 \
  -v $(pwd)/notebooks:/app/notebooks \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  customer-segmentation:latest \
  jupyter notebook --ip=0.0.0.0 --allow-root --no-browser

# Development
docker run --rm -it \
  -v $(pwd):/app \
  customer-segmentation:latest \
  bash
```

## Benefits of Docker Setup

### For Development
- ✅ Consistent environment across all machines
- ✅ No dependency conflicts
- ✅ Easy setup for new contributors
- ✅ Isolated from system Python
- ✅ Version control for entire environment

### For Deployment
- ✅ Reproducible builds
- ✅ Easy to deploy anywhere (cloud, on-premise)
- ✅ Portable across Linux, Mac, Windows
- ✅ Can be orchestrated with Kubernetes
- ✅ Resource limits and monitoring

### For Users
- ✅ One command to run everything
- ✅ No manual dependency installation
- ✅ No Python version conflicts
- ✅ Works out of the box
- ✅ Multiple interface options (CLI, Jupyter, shell)

## Key Features

1. **Multiple Entry Points**
   - Training pipeline (default)
   - Prediction script
   - Jupyter notebook
   - Interactive shell

2. **Data Persistence**
   - Volume mounts ensure data survives container restarts
   - Models saved to host machine
   - Results accessible outside container

3. **Platform Support**
   - Linux (native)
   - macOS (Docker Desktop)
   - Windows (Docker Desktop)
   - WSL2 support

4. **Flexible Usage**
   - 4 different ways to run (scripts, make, compose, docker)
   - Choose based on preference and use case
   - All methods produce same results

5. **Development Friendly**
   - Live code editing in dev mode
   - Jupyter for interactive exploration
   - Easy debugging

6. **Production Ready**
   - Clean, optimized Dockerfile
   - Proper layer caching
   - Small image size
   - Security best practices

## File Structure After Setup

```
Customer-Segmentation-with-Auto-Encoders/
├── Dockerfile                          # Docker image definition
├── .dockerignore                       # Build context exclusions
├── docker-compose.yml                  # Multi-service orchestration
├── docker-run.sh                       # Linux/Mac helper script
├── docker-run.bat                      # Windows helper script
├── Makefile                            # Make targets for Docker
├── DOCKER.md                           # Complete Docker documentation
├── DOCKER_QUICKREF.md                  # Quick reference guide
├── DOCKER_SETUP_SUMMARY.md            # This file
├── README.md                           # Updated with Docker section
├── ... (rest of project files)
```

## Testing Checklist

Before using in production, verify:

- [ ] Docker is installed and running
- [ ] Data files exist in `data/raw/`
- [ ] Image builds successfully
- [ ] Training runs and creates models
- [ ] Predictions generate correct output
- [ ] Jupyter notebook accessible
- [ ] Volume mounts work correctly
- [ ] Helper scripts are executable
- [ ] Results persist after container stops

## Common Issues and Solutions

1. **"docker: command not found"**
   - Install Docker Desktop
   - Ensure Docker daemon is running

2. **"Permission denied" on scripts**
   - Run: `chmod +x docker-run.sh`

3. **Port 8888 already in use**
   - Change port: `docker run -p 8889:8888 ...`

4. **Slow build on Windows**
   - Use WSL2 backend
   - Place project in WSL filesystem

5. **Out of disk space**
   - Run: `docker system prune -a`

## Next Steps

1. **Build the image:**
   ```bash
   ./docker-run.sh build
   # or
   make build
   ```

2. **Run training:**
   ```bash
   ./docker-run.sh train
   # or
   make train
   ```

3. **Check results:**
   ```bash
   ls results/
   ls results/plots/
   ```

4. **Explore with Jupyter:**
   ```bash
   ./docker-run.sh jupyter
   # Open browser to http://localhost:8888
   ```

## Support

- **Docker issues:** See DOCKER.md troubleshooting section
- **Project issues:** See main README.md
- **Quick reference:** See DOCKER_QUICKREF.md

## Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Dockerfile Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [TensorFlow Docker Guide](https://www.tensorflow.org/install/docker)

---

**Created:** October 2025
**Version:** 1.0
**Maintainer:** Customer Segmentation Project Team
