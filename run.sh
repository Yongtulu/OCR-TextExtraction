#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "安装 Python 依赖..."
pip install -q -r requirements.txt

# 首次下载翻译模型（之后完全离线）
MODELS_FLAG=".models_ready"
if [ ! -f "$MODELS_FLAG" ]; then
  echo "首次运行：下载离线翻译语言包..."
  python setup_models.py
  touch "$MODELS_FLAG"
fi

# PaddleOCR 模型会在首次调用时自动下载到 ~/.paddleocr/
echo "启动应用..."


conda init
conda create -n ocr python=3.10 -y
conda activate ocr
pip install "numpy<2" paddlepaddle "paddleocr==2.7.3" Pillow langdetect argostranslate
python /Users/jungang/text_capture/app.py
