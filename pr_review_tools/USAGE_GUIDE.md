# PR Review Tools - 完整使用指南

## 📖 目录

1. [系统架构](#系统架构)
2. [完整工作流](#完整工作流)
3. [功能详解](#功能详解)
4. [API接口](#api接口)
5. [Git操作](#git操作)
6. [Docker管理](#docker管理)
7. [故障排除](#故障排除)

## 🏗️ 系统架构

### 组件
- **Flask服务器**：提供Web界面和API
- **Git集成**：管理分支、合并、冲突检测
- **Docker**：预览Jekyll网站
- **GitHub API**：获取PR、发布评论

### 数据流
```
GitHub → Flask后端 → Git操作 → Docker预览 → 用户审查 → 合并 → GitHub评论
```

## 🔄 完整工作流

### Phase 1: 启动
```bash
cd pr_review_tools
./start.sh
```

**系统初始化**：
- 检查Flask安装
- 检查Docker状态
- 加载PR列表
- 启动Web服务器（5001端口）

### Phase 2: 选择PR

**用户操作**：点击左侧PR列表中的PR

**系统响应**：
1. 记录当前分支（通常是main）
2. 创建临时分支：`review-pr-{number}`
3. 切换到临时分支
4. Fetch远程仓库最新状态
5. 合并main分支
6. 合并PR分支
7. 检查合并冲突
8. 过滤文件（只保留assets/和_posts/）
9. 启动Docker预览

**状态转换**：
- idle → preparing → starting_docker → docker_ready

### Phase 3: 审查

**预览内容**：
- 点击"打开预览网站"
- 在 `http://localhost:4000/CS163-Projects-2025Fall/` 查看效果
- 检查文章内容、图片、格式等

**文件分类**：
- ✅ **Accepted Files**：`assets/`、`_posts/`下的修改
- ❌ **Rejected Files**：其他目录的修改

### Phase 4: 编辑（可选）

**启动编辑模式**：
1. 点击 **Start Edit** 按钮
2. 系统显示可编辑文件列表
3. 使用编辑器（VS Code等）编辑文件

**支持的操作**：
- 修改 `_posts/` 下的Markdown文件
- 添加/替换 `assets/images/` 下的图片
- 修改文件内容、格式等

**完成编辑**：
1. 点击 **Finish Edit** 按钮
2. 系统检测修改：`git status --porcelain`
3. 显示修改的文件列表
4. 记录到 `edited_files` 状态

### Phase 5: 决策

#### 选项A：接受并合并

**用户操作**：
1. （可选）填写Additional Comment
2. 点击 **接受并合并（本地）** 按钮
3. 确认对话框

**系统操作**：
1. 过滤并添加文件：
   ```bash
   git add assets/
   git add _posts/
   ```
2. 提交更改：
   ```bash
   git commit -m "Filtered changes from PR #X"
   ```
3. 停止Docker容器
4. 切换回main分支
5. 合并临时分支（fast-forward）：
   ```bash
   git merge review-pr-X --ff-only
   ```
6. 删除临时分支
7. 生成并发布GitHub评论

**GitHub评论格式**：
```markdown
## 🤖 Automated Review Report

### ✅ Accepted Files (X)
- `_posts/article.md`
- `assets/images/pic.png`

### ❌ Rejected Files (Y)
- `Gemfile.lock`
- `_site/index.html`

### ✏️ Edited Files (Z)
- `_posts/article.md`

## 💡 Additional Comments
{用户的自定义评论}
```

#### 选项B：跳过

**系统操作**：
1. 重置临时分支：`git reset --hard`
2. 停止Docker容器
3. 切换回main分支
4. 删除临时分支
5. 不发布GitHub评论

### Phase 6: Push（手动）

**重要**：系统不会自动push到远程！

```bash
# 检查本地状态
git log --oneline -5

# 确认无误后push
git push origin main
```

## ⚙️ 功能详解

### 文件过滤

**白名单目录**：
```python
ALLOWED_DIRS = ['assets/', '_posts/']
```

**过滤逻辑**：
1. 获取PR的所有改动：`git diff --name-only`
2. 分类文件：
   - 以 `assets/` 或 `_posts/` 开头 → accepted
   - 其他 → rejected
3. 只添加accepted files到暂存区

**特殊处理**：
- 二进制文件（图片）正常处理
- 新建文件和修改文件统一处理
- 删除文件会被忽略（不支持）

### Edit Mode

**实现原理**：
1. 用户触发 **Start Edit**
2. 系统记录初始状态：`git_initial_state = git status`
3. 用户在外部编辑器修改文件
4. 用户触发 **Finish Edit**
5. 系统对比：`git_current_state = git status`
6. 计算差异 → `edited_files`

**限制**：
- 只能编辑已过滤的文件
- 编辑其他文件会被忽略
- 不能删除文件

### GitHub集成

**获取PR列表**：
```bash
curl -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/OWNER/REPO/pulls
```

**发布评论**：
```bash
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"body": "comment content"}' \
  https://api.github.com/repos/OWNER/REPO/issues/PR_NUMBER/comments
```

### Docker管理

**构建**：
```bash
docker compose build --no-cache
```

**启动**：
```bash
docker compose up -d
```

**停止**：
```bash
docker compose down
```

**重启**：
```bash
docker compose down
docker compose up -d
```

**日志**：
```bash
docker compose logs --tail=50
```

## 🔌 API接口

### GET /api/prs
获取PR列表

**响应**：
```json
{
  "success": true,
  "prs": [
    {
      "index": 1,
      "number": 42,
      "title": "Add new article",
      "author": "username",
      "branch": "feature-branch"
    }
  ],
  "stats": {
    "total": 10,
    "reviewed": 3,
    "accepted": 2,
    "skipped": 1
  }
}
```

### POST /api/prepare_pr
准备审查PR

**请求**：
```json
{"index": 1}
```

### GET /api/status
获取当前状态

**响应**：
```json
{
  "success": true,
  "status": "docker_ready",
  "message": "Docker is ready",
  "docker_running": true,
  "preview_url": "http://localhost:4000/...",
  "current_pr": {...}
}
```

### POST /api/accept_pr
接受并合并PR

**请求**：
```json
{
  "additional_comment": "LGTM!"
}
```

### POST /api/skip_pr
跳过PR

### POST /api/refresh_prs
刷新PR列表

### POST /api/restart_docker
重启Docker

### GET /api/get_files
获取可编辑文件列表

### POST /api/check_edited_files
检测修改的文件

## 🐙 Git操作

### 分支命名
```
review-pr-{PR_NUMBER}
```

### 合并策略
1. **临时分支**：standard merge（可能有merge commit）
2. **main分支**：fast-forward only（保持线性历史）

### 冲突处理
如果检测到冲突：
- 状态设置为 `conflict`
- 显示错误信息
- 自动清理环境
- PR不会被合并

### 清理机制

**正常清理**（接受/跳过后）：
```bash
git reset --hard
git checkout main
git branch -D review-pr-X
docker compose down
```

**Ctrl+C清理**（信号处理）：
```bash
# start.sh和pr_review_server.py都实现了
trap cleanup SIGINT SIGTERM

cleanup() {
  current_branch=$(git branch --show-current)
  if [[ $current_branch == review-pr-* ]]; then
    git reset --hard
    git checkout main
    git branch -D $current_branch
  fi
  docker compose down
}
```

## 🐳 Docker管理

### 配置文件
- `Dockerfile`：Jekyll环境
- `docker-compose.yml`：容器编排

### 预览URL
```
http://localhost:4000/CS163-Projects-2025Fall/
```

### 常见问题

**端口冲突**：
```yaml
# docker-compose.yml
ports:
  - "4001:4000"  # 改为其他端口
```

**依赖缺失**：
```bash
docker compose build --no-cache
```

**容器未启动**：
```bash
docker compose logs
```

## 🔧 故障排除

### Flask启动失败

**症状**：`ModuleNotFoundError: No module named 'flask'`

**解决**：
```bash
pip install flask
# 或
conda install flask
```

### Git冲突

**症状**：状态显示 `conflict`

**手动解决**：
```bash
# 1. 查看冲突
git status

# 2. 解决冲突文件
# 编辑文件，移除冲突标记

# 3. 标记为已解决
git add <resolved_files>
git commit

# 4. 继续审查
```

### Docker不启动

**症状**：预览链接不可用

**检查**：
```bash
# Docker是否运行
docker info

# 容器状态
docker compose ps

# 容器日志
docker compose logs

# 重新构建
docker compose build --no-cache
docker compose up -d
```

### 端口占用

**症状**：`Address already in use`

**Flask端口**：
```python
# pr_review_server.py, line ~690
app.run(debug=False, host='0.0.0.0', port=5002)
```

**Docker端口**：
```yaml
# docker-compose.yml
ports:
  - "4001:4000"
```

### 临时分支残留

**症状**：多个 `review-pr-*` 分支

**清理**：
```bash
# 列出所有review分支
git branch | grep review-pr

# 批量删除
git branch | grep review-pr | xargs git branch -D

# 或单独删除
git branch -D review-pr-42
```

### GitHub Token问题

**症状**：无法获取PR或发布评论

**检查**：
```bash
# Token是否设置
echo $GITHUB_TOKEN

# Token权限
# 需要: repo (full access)
```

**设置**：
```bash
# 临时设置
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"

# 永久设置（~/.zshrc或~/.bashrc）
echo 'export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"' >> ~/.zshrc
source ~/.zshrc
```

## 🔒 安全最佳实践

1. **Token安全**：不要提交到Git
2. **本地优先**：不自动push
3. **分支隔离**：临时分支操作
4. **自动清理**：防止状态污染
5. **文件过滤**：防止意外修改

## 📊 监控和日志

### 系统状态
在Web界面右上角查看统计信息

### Git日志
```bash
git log --oneline --graph -10
```

### Docker日志
```bash
docker compose logs --tail=100 -f
```

### Flask日志
终端输出显示所有HTTP请求

## 💡 高级技巧

### 批量处理
依次选择、审查、接受多个PR

### 快速编辑
使用VS Code的文件监控实时查看效果

### 自定义过滤
修改 `ALLOWED_DIRS` 变量

### 自动化测试
```bash
# 测试Docker build
docker compose build

# 测试Jekyll
docker compose run --rm site jekyll build
```

## 📚 参考资料

- **Jekyll文档**：https://jekyllrb.com/docs/
- **GitHub API**：https://docs.github.com/en/rest
- **Flask文档**：https://flask.palletsprojects.com/
- **Docker Compose**：https://docs.docker.com/compose/

---

**维护者**：CS163-Projects-2025Fall Team  
**版本**：2.0  
**最后更新**：2025-12-14
