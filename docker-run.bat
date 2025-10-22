@echo off
REM Helper script to run customer segmentation Docker container on Windows

setlocal enabledelayedexpansion

set IMAGE_NAME=customer-segmentation
set CONTAINER_NAME=customer-segmentation-app

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="--help" goto help
if "%1"=="-h" goto help

if "%1"=="build" goto build
if "%1"=="train" goto train
if "%1"=="predict" goto predict
if "%1"=="jupyter" goto jupyter
if "%1"=="dev" goto dev
if "%1"=="clean" goto clean

echo [ERROR] Unknown command: %1
echo.
goto help

:help
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo     build       Build the Docker image
echo     train       Run the training pipeline
echo     predict     Run predictions on test data
echo     jupyter     Start Jupyter notebook server (available at http://localhost:8888)
echo     dev         Start interactive shell for development
echo     clean       Remove containers and images
echo     help        Show this help message
echo.
echo Examples:
echo     %0 build          # Build the Docker image
echo     %0 train          # Train the model
echo     %0 jupyter        # Start Jupyter notebook
echo     %0 dev            # Start development shell
echo.
goto end

:build
echo [INFO] Building Docker image: %IMAGE_NAME%...
docker build -t %IMAGE_NAME%:latest .
if %errorlevel% equ 0 (
    echo [SUCCESS] Docker image built successfully!
) else (
    echo [ERROR] Failed to build Docker image
    exit /b 1
)
goto end

:train
echo [INFO] Running training pipeline...
docker run --rm ^
    --name %CONTAINER_NAME%-train ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\results:/app/results" ^
    %IMAGE_NAME%:latest ^
    python train_pipeline.py
if %errorlevel% equ 0 (
    echo [SUCCESS] Training completed! Check results\ directory.
) else (
    echo [ERROR] Training failed
    exit /b 1
)
goto end

:predict
echo [INFO] Running predictions...
docker run --rm ^
    --name %CONTAINER_NAME%-predict ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\results:/app/results" ^
    %IMAGE_NAME%:latest ^
    python predict.py --input data/raw/Test.csv --output results/predictions.csv
if %errorlevel% equ 0 (
    echo [SUCCESS] Predictions completed! Check results\predictions.csv
) else (
    echo [ERROR] Prediction failed
    exit /b 1
)
goto end

:jupyter
echo [INFO] Starting Jupyter notebook server...
echo [INFO] Access at: http://localhost:8888
docker run --rm ^
    --name %CONTAINER_NAME%-jupyter ^
    -p 8888:8888 ^
    -v "%cd%\data:/app/data" ^
    -v "%cd%\models:/app/models" ^
    -v "%cd%\results:/app/results" ^
    -v "%cd%\notebooks:/app/notebooks" ^
    -v "%cd%\src:/app/src" ^
    %IMAGE_NAME%:latest ^
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
goto end

:dev
echo [INFO] Starting development shell...
docker run --rm -it ^
    --name %CONTAINER_NAME%-dev ^
    -v "%cd%:/app" ^
    %IMAGE_NAME%:latest ^
    bash
goto end

:clean
echo [INFO] Cleaning up containers and images...
for /f "tokens=*" %%i in ('docker ps -a -q --filter "ancestor=%IMAGE_NAME%"') do docker rm -f %%i 2>nul
docker rmi %IMAGE_NAME%:latest 2>nul
echo [SUCCESS] Cleanup completed!
goto end

:end
endlocal
