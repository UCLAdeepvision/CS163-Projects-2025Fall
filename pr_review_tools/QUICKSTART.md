# PR Review Tools - 快速参考

## ⚡ 一键启动

```bash
cd pr_review_tools
./start.sh
```

打开浏览器：**http://localhost:5001**

## 🎯 基本操作

### 审查PR
1. **选择PR** → 左侧列表点击
2. **等待** → 自动准备环境
3. **预览** → 点击"打开预览网站"
4. **决定** → 接受 or 跳过

### 编辑文件
1. **Start Edit** → 开启编辑模式
2. **编辑** → 用编辑器修改文件
3. **Finish Edit** → 检测修改

### 合并到main
1. **接受PR** → 点击"接受并合并"，自动完成：
   - 合并到main
   - 发布GitHub评论
   - **关闭GitHub PR**
   - **自动push到远程**
   - 从列表移除
2. **刷新** → 点击"🔄 刷新"从GitHub重新加载（只显示open状态的PR）

## ⌨️ 快捷命令

```bash
# 刷新PR列表
点击界面上的 "🔄 刷新PR列表"

# 重启Docker
点击界面上的 "🐳 重启Docker"

# 停止服务器（自动清理）
Ctrl + C

# 手动清理
git checkout main
git branch -D review-pr-XX
docker compose down
```

## 📂 文件过滤规则

✅ **接受**：`assets/`、`_posts/`  
❌ **拒绝**：其他所有目录

## 🛠️ 常见问题

### Flask未安装
```bash
pip install flask
```

### Docker未运行
启动 Docker Desktop

### 端口占用
修改 `pr_review_server.py` 第690行左右：
```python
app.run(..., port=5002)  # 改为其他端口
```

## 🔥 紧急清理

```bash
docker compose down
git reset --hard
git checkout main
git branch | grep review-pr | xargs git branch -D
```

## 🎨 界面说明

- **左侧**：PR列表
- **右上**：统计信息
- **右下**：审查控制面板

## 💡 小贴士

- 先预览，再编辑
- 添加Additional Comment更专业
- 记得手动push！

## 📞 需要帮助？

查看完整文档：`README.md`
