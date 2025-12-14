# PR Review Tools - 交互式审查系统

专为 CS163-Projects-2025Fall 设计的Pull Request审查与合并工具。

## 🚀 快速开始

```bash
cd pr_review_tools
./start.sh
```

在浏览器中打开：`http://localhost:5001`

## ✨ 特性

### 🔍 智能审查
- 自动获取GitHub PR列表
- 创建临时分支进行安全审查
- 实时Docker预览
- 自动文件过滤（只接受 `assets/` 和 `_posts/`）

### ✏️ Edit Mode（编辑模式）
- 在审查过程中手动编辑文件
- **全面追踪**：自动检测修改、添加、删除的文件
- 修改会包含在合并和GitHub评论中
- 分类显示：Modified ✏️ / Added ➕ / Deleted ➖

### 💬 GitHub评论
- 自动生成专业的英文审查报告
- 列出接受/拒绝/编辑的文件
- 支持添加自定义评论

### 🛡️ 安全机制
- Ctrl+C自动清理临时分支
- 仅本地合并，不自动push
- 冲突检测和处理

## 📋 工作流程

### 1. 选择PR
在左侧列表中点击要审查的PR

### 2. 等待准备
系统会自动：
- 创建临时分支 `review-pr-XX`
- 合并main和PR分支
- 启动Docker预览

### 3. 审查内容
点击 **🌐 在新标签页打开预览网站** 查看效果

### 4. 编辑文件（可选）
如果需要修改：
1. 点击 **✏️ Start Edit**
2. 使用你的编辑器修改 `assets/` 或 `_posts/` 下的文件
3. 点击 **✓ Finish Edit** 检测修改

### 5. 添加评论（可选）
在 **Additional Comment** 框中添加审查意见

### 6. 做出决定
- **✅ 接受并合并**：自动执行完整流程
  - 合并到main分支
  - 发布GitHub评论
  - **自动关闭GitHub PR**
  - **自动push到远程仓库**
  - 从列表中移除
- **⏭️ 跳过**：放弃审查，清理环境，PR保留在列表中

### 7. PR列表管理
- **自动移除**：接受PR后立即从本地列表移除
- **保留跳过**：跳过的PR继续保留在列表中
- **刷新过滤**：点击"🔄 刷新PR列表"强制从GitHub重新获取
  - ✅ **只加载 `state=open` 的PR**（已关闭的自动过滤）
  - ✅ 忽略本地缓存，获取最新状态
  - ✅ 已接受的PR自动push，不会重新出现

## 🔧 系统要求

- Python 3.x
- Flask (`pip install flask`)
- Docker Desktop
- GitHub Personal Access Token

## ⚙️ 配置

### 设置GitHub Token

```bash
export GITHUB_TOKEN="your_token_here"
```

或在 `~/.zshrc` / `~/.bashrc` 中添加：
```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
```

### 文件过滤规则

系统**只接受**以下目录的修改：
- `assets/`（图片、CSS等静态资源）
- `_posts/`（博客文章）

其他目录的修改会被**自动拒绝**并在GitHub评论中列出。

## 📁 文件结构

```
pr_review_tools/
├── pr_review_server.py    # Flask后端服务器
├── start.sh                # 启动脚本（包含Ctrl+C清理）
├── templates/
│   └── pr_review.html      # Web界面
├── README.md               # 详细文档
└── QUICKSTART.md           # 快速参考
```

## 🔒 Git忽略

所有审查工具文件已添加到 `.gitignore`，不会被提交到仓库。

## 🐛 故障排除

### Flask未找到
```bash
pip install flask
# 或在conda环境中
conda install flask
```

### Docker未运行
启动 Docker Desktop 应用

### 端口冲突（5001被占用）
修改 `pr_review_server.py` 中的端口号：
```python
app.run(debug=False, host='0.0.0.0', port=5002)  # 改为其他端口
```

### 临时分支未清理
手动清理：
```bash
git checkout main
git branch -D review-pr-XX
docker compose down
```

## 🆘 紧急清理

如果程序崩溃或异常退出：

```bash
# 1. 停止Docker
docker compose down

# 2. 检查当前分支
git branch

# 3. 如果在review-pr-*分支上
git reset --hard
git checkout main
git branch -D review-pr-XX  # 替换XX为实际的PR编号

# 4. 清理其他临时分支
git branch | grep review-pr | xargs git branch -D
```

## 📚 更多信息

- **快速开始**：见 `QUICKSTART.md`
- **编辑模式详解**：见 `EDIT_MODE_GUIDE.md`
- **完整使用指南**：见 `USAGE_GUIDE.md`

## 🔐 安全注意事项

1. **本地合并**：工具只在本地合并PR，需要手动push
2. **文件过滤**：自动拒绝非白名单目录的修改
3. **临时分支**：所有操作在临时分支进行，保护main分支
4. **自动清理**：Ctrl+C会自动清理环境

## 💡 提示

- 使用Edit Mode前先在预览中查看效果
- 添加Additional Comment让审查更清晰
- 记得定期手动push accepted的PR到远程

## 📄 License

仅供 CS163-Projects-2025Fall 课程使用
