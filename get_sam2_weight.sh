#!/bin/bash

# --- 설정 ---
WEIGHTS_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
WEIGHTS_DIR="checkpoints"
WEIGHTS_FILENAME="sam2.1_hiera_large.pt"
WEIGHTS_PATH="${WEIGHTS_DIR}/${WEIGHTS_FILENAME}"

CONFIG_URL="https://raw.githubusercontent.com/facebookresearch/sam2/main/configs/sam2.1/sam2.1_hiera_l.yaml"
CONFIG_DIR="configs/sam2.1"
CONFIG_FILENAME="sam2.1_hiera_l.yaml"
CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILENAME}"

# --- 메인 로직 ---
echo "==========================================="
echo " Setting up SAM2 Weights and Config File "
echo "==========================================="

# 1. 디렉토리 생성
echo -e "\n--- Creating directories (if they don't exist) ---"
mkdir -p "$WEIGHTS_DIR"
echo "Checked/Created directory: $WEIGHTS_DIR"
mkdir -p "$CONFIG_DIR"
echo "Checked/Created directory: $CONFIG_DIR"

# 2. 가중치 파일 확인 및 다운로드
echo -e "\n--- Checking Weights File ($WEIGHTS_FILENAME) ---"
if [ -f "$WEIGHTS_PATH" ]; then
    echo "Weights file already exists: $WEIGHTS_PATH"
else
    echo "Weights file not found. Downloading..."
    # wget [옵션] -O [저장될 파일 경로] [URL]
    wget --show-progress -O "$WEIGHTS_PATH" "$WEIGHTS_URL"
    if [ $? -eq 0 ]; then
        echo "Weights file downloaded successfully."
    else
        echo "Error downloading weights file. Please check the URL or network connection."
        # 실패 시 생성된 빈 파일이나 부분 파일을 삭제할 수 있습니다.
        rm -f "$WEIGHTS_PATH"
    fi
fi

# 3. 설정 파일 확인 및 다운로드
echo -e "\n--- Checking Config File ($CONFIG_FILENAME) ---"
if [ -f "$CONFIG_PATH" ]; then
    echo "Config file already exists: $CONFIG_PATH"
else
    echo "Config file not found. Downloading..."
    # wget [옵션] -O [저장될 파일 경로] [URL]
    wget --show-progress -O "$CONFIG_PATH" "$CONFIG_URL"
     if [ $? -eq 0 ]; then
        echo "Config file downloaded successfully."
    else
        echo "Error downloading config file. Please check the URL or network connection."
        rm -f "$CONFIG_PATH"
    fi
fi

echo -e "\n==========================================="
echo " Setup process finished."
echo "==========================================="

# 최종 확인
echo -e "\nFinal Check:"
if [ -f "$WEIGHTS_PATH" ]; then
    echo "✅ Weights file exists: $WEIGHTS_PATH"
else
    echo "❌ Weights file is missing: $WEIGHTS_PATH"
fi

if [ -f "$CONFIG_PATH" ]; then
    echo "✅ Config file exists: $CONFIG_PATH"
else
    echo "❌ Config file is missing: $CONFIG_PATH"
fi