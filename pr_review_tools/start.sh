#!/bin/bash

# PR Review Tool 启动脚本
# 包含Ctrl+C清理逻辑

echo "================================"
echo "PR 审查工具"
echo "================================"
echo ""

# 清理函数
cleanup() {
    echo ""
    echo "收到退出信号，正在清理..."
    
    # 停止Docker
    echo "停止Docker容器..."
    cd "$(dirname "$0")/.." && docker compose down 2>/dev/null
    
    # 清理Git状态
    current_branch=$(git branch --show-current 2>/dev/null)
    
    if [[ "$current_branch" == review-pr-* ]]; then
        echo "检测到临时分支: $current_branch"
        
        # 重置更改
        echo "重置未提交的更改..."
        git reset --hard 2>/dev/null
        
        # 清理未跟踪的文件
        echo "清理未跟踪的文件..."
        rm -rf _site/ 2>/dev/null
        git clean -fd 2>/dev/null
        
        # 返回main
        echo "返回main分支..."
        git checkout main 2>/dev/null
        
        # 删除临时分支
        echo "删除临时分支..."
        git branch -D "$current_branch" 2>/dev/null
        
        echo "✓ 清理完成"
    fi
    
    echo "再见！"
    exit 0
}

# 注册信号处理
trap cleanup SIGINT SIGTERM

# 检查Flask
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask 未安装"
    echo "正在安装 Flask..."
    pip install flask
    echo ""
fi

# 检查Docker
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
echo "按 Ctrl+C 停止服务器（会自动清理临时分支）"
echo ""

# 启动服务器
python pr_review_server.py

# 如果服务器正常退出，也执行清理
cleanup

