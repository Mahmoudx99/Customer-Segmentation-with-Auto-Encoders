# Makefile for Customer Segmentation with Auto Encoders
# Provides convenient commands for Docker operations

.PHONY: help build train predict jupyter dev clean test

# Variables
IMAGE_NAME := customer-segmentation
CONTAINER_NAME := customer-segmentation-app

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Customer Segmentation with Auto Encoders - Docker Commands"
	@echo ""
	@echo "Usage: make [TARGET]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make build     # Build the Docker image"
	@echo "  make train     # Train the model"
	@echo "  make jupyter   # Start Jupyter notebook"
	@echo ""

build: ## Build Docker image
	@echo "Building Docker image: $(IMAGE_NAME)..."
	docker build -t $(IMAGE_NAME):latest .
	@echo "✓ Docker image built successfully!"

train: ## Run training pipeline
	@echo "Running training pipeline..."
	docker run --rm \
		--name $(CONTAINER_NAME)-train \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/results:/app/results \
		$(IMAGE_NAME):latest \
		python train_pipeline.py
	@echo "✓ Training completed! Check results/ directory."

predict: ## Run predictions on test data
	@echo "Running predictions..."
	docker run --rm \
		--name $(CONTAINER_NAME)-predict \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/results:/app/results \
		$(IMAGE_NAME):latest \
		python predict.py --input data/raw/Test.csv --output results/predictions.csv
	@echo "✓ Predictions completed! Check results/predictions.csv"

jupyter: ## Start Jupyter notebook server
	@echo "Starting Jupyter notebook server..."
	@echo "Access at: http://localhost:8888"
	docker run --rm \
		--name $(CONTAINER_NAME)-jupyter \
		-p 8888:8888 \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/results:/app/results \
		-v $(CURDIR)/notebooks:/app/notebooks \
		-v $(CURDIR)/src:/app/src \
		$(IMAGE_NAME):latest \
		jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
		--NotebookApp.token='' --NotebookApp.password=''

dev: ## Start development shell
	@echo "Starting development shell..."
	docker run --rm -it \
		--name $(CONTAINER_NAME)-dev \
		-v $(CURDIR):/app \
		$(IMAGE_NAME):latest \
		bash

compose-up: ## Start services with docker-compose
	docker-compose up

compose-down: ## Stop services with docker-compose
	docker-compose down

compose-train: ## Run training with docker-compose
	docker-compose up train

compose-predict: ## Run predictions with docker-compose
	docker-compose up predict

compose-jupyter: ## Start Jupyter with docker-compose
	docker-compose up jupyter

test: ## Run quick test of the pipeline
	@echo "Running quick test..."
	docker run --rm \
		--name $(CONTAINER_NAME)-test \
		-v $(CURDIR)/data:/app/data \
		-v $(CURDIR)/models:/app/models \
		-v $(CURDIR)/results:/app/results \
		$(IMAGE_NAME):latest \
		python -c "import sys; sys.path.append('src'); from data_loader import load_data; train_df, test_df = load_data('data/raw'); print(f'✓ Test passed! Data loaded: {train_df.shape}, {test_df.shape}')"

clean: ## Remove containers and images
	@echo "Cleaning up containers and images..."
	-docker ps -a | grep $(IMAGE_NAME) | awk '{print $$1}' | xargs docker rm -f 2>/dev/null || true
	-docker rmi $(IMAGE_NAME):latest 2>/dev/null || true
	@echo "✓ Cleanup completed!"

logs: ## Show logs from last container run
	docker logs $(CONTAINER_NAME)-train 2>/dev/null || \
	docker logs $(CONTAINER_NAME)-predict 2>/dev/null || \
	echo "No container logs found"

ps: ## List running containers
	@docker ps -a | grep $(IMAGE_NAME) || echo "No containers running"

size: ## Show image size
	@docker images $(IMAGE_NAME):latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

pull-data: ## Copy Kaggle data to data/raw
	@echo "Copying data from Kaggle folder..."
	cp Kaggle/*.csv data/raw/ 2>/dev/null || echo "Kaggle folder not found"
	@echo "✓ Data copied"

# Docker Compose shortcuts
up: compose-up
down: compose-down
