#!/bin/bash
# Helper script to run customer segmentation Docker container

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="customer-segmentation"
CONTAINER_NAME="customer-segmentation-app"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    build       Build the Docker image
    train       Run the training pipeline
    predict     Run predictions on test data
    jupyter     Start Jupyter notebook server (available at http://localhost:8888)
    dev         Start interactive shell for development
    clean       Remove containers and images
    help        Show this help message

Examples:
    $0 build          # Build the Docker image
    $0 train          # Train the model
    $0 jupyter        # Start Jupyter notebook
    $0 dev            # Start development shell

EOF
}

# Function to build Docker image
build_image() {
    print_info "Building Docker image: ${IMAGE_NAME}..."
    docker build -t ${IMAGE_NAME}:latest .
    print_success "Docker image built successfully!"
}

# Function to run training
run_training() {
    print_info "Running training pipeline..."
    docker run --rm \
        --name ${CONTAINER_NAME}-train \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/results:/app/results \
        ${IMAGE_NAME}:latest \
        python train_pipeline.py
    print_success "Training completed! Check results/ directory."
}

# Function to run predictions
run_prediction() {
    print_info "Running predictions..."
    docker run --rm \
        --name ${CONTAINER_NAME}-predict \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/results:/app/results \
        ${IMAGE_NAME}:latest \
        python predict.py --input data/raw/Test.csv --output results/predictions.csv
    print_success "Predictions completed! Check results/predictions.csv"
}

# Function to start Jupyter notebook
run_jupyter() {
    print_info "Starting Jupyter notebook server..."
    print_info "Access at: http://localhost:8888"
    docker run --rm \
        --name ${CONTAINER_NAME}-jupyter \
        -p 8888:8888 \
        -v $(pwd)/data:/app/data \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/results:/app/results \
        -v $(pwd)/notebooks:/app/notebooks \
        -v $(pwd)/src:/app/src \
        ${IMAGE_NAME}:latest \
        jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
        --NotebookApp.token='' --NotebookApp.password=''
}

# Function to start development shell
run_dev() {
    print_info "Starting development shell..."
    docker run --rm -it \
        --name ${CONTAINER_NAME}-dev \
        -v $(pwd):/app \
        ${IMAGE_NAME}:latest \
        bash
}

# Function to clean up
clean_up() {
    print_info "Cleaning up containers and images..."

    # Stop and remove containers
    docker ps -a | grep ${IMAGE_NAME} | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true

    # Remove image
    docker rmi ${IMAGE_NAME}:latest 2>/dev/null || true

    print_success "Cleanup completed!"
}

# Main script logic
case "${1:-help}" in
    build)
        build_image
        ;;
    train)
        run_training
        ;;
    predict)
        run_prediction
        ;;
    jupyter)
        run_jupyter
        ;;
    dev)
        run_dev
        ;;
    clean)
        clean_up
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        usage
        exit 1
        ;;
esac
