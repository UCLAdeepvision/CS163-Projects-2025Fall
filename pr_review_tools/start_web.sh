#!/bin/bash

# PR Review Web Interface 启动脚本

echo "================================"
echo "PR 审查工具 - Web界面"
echo "================================"
echo ""

# 检查Flask是否安装（使用 python 而不是 python3，适配 conda 环境）
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask 未安装"
    echo "正在安装 Flask..."
    pip install flask
    echo ""
fi

# 检查Docker是否运行
if ! docker info >/dev/null 2>&1; then
    echo "⚠️  警告: Docker 未运行"
    echo "请先启动 Docker Desktop"
    echo ""
fi

# 进入工具目录
cd "$(dirname "$0")"

echo "✅ 准备就绪"
echo ""
echo "启动Web服务器..."
echo "================================"
echo ""
echo "请在浏览器中访问: http://localhost:5001"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

# 启动服务器（使用 python 而不是 python3）
python pr_review_server.py

