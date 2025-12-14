#!/usr/bin/env python3
"""
Interactive Pull Request Reviewer and Merger
这个脚本帮助您安全地审查和合并Pull Request
"""

import json
import subprocess
import sys
import time
import signal
import os

# 获取项目根目录（上一级目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PR_DATA_FILE = os.path.join(SCRIPT_DIR, 'pr_data.json')

# 切换到项目根目录
os.chdir(PROJECT_ROOT)

def run_command(cmd, capture=True, check=True):
    """运行shell命令并返回结果"""
    try:
        if capture:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=True, check=check)
            return None
    except subprocess.CalledProcessError as e:
        if check:
            print(f"错误: {cmd}")
            print(f"错误信息: {e.stderr if capture else e}")
            return None
        else:
            return e.stdout.strip() if capture else None

def load_pull_requests():
    """从JSON文件加载Pull Request列表"""
    try:
        with open(PR_DATA_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("错误: 找不到pr_data.json文件")
        print("正在从GitHub获取PR列表...")
        result = run_command(f'curl -s "https://api.github.com/repos/UCLAdeepvision/CS163-Projects-2025Fall/pulls?state=open&per_page=100" > {PR_DATA_FILE}', capture=False, check=False)
        try:
            with open(PR_DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            print("无法获取PR列表")
            sys.exit(1)

def display_pull_requests(prs):
    """显示所有Pull Request"""
    print(f"\n{'='*80}")
    print(f"发现 {len(prs)} 个待审核的Pull Request")
    print(f"{'='*80}\n")
    
    for i, pr in enumerate(prs, 1):
        print(f"{i}. PR #{pr['number']}: {pr['title']}")
        print(f"   作者: {pr['user']['login']}")
        print(f"   分支: {pr['head']['label']} -> {pr['base']['ref']}")
        print(f"   创建时间: {pr['created_at']}")
        print()

def cleanup_docker():
    """清理Docker容器"""
    print("\n正在停止Docker容器...")
    run_command("docker-compose down", capture=False, check=False)
    time.sleep(2)
    print("✓ Docker容器已停止")

def start_docker():
    """启动Docker预览网站"""
    print("\n正在启动Docker容器...")
    print("网站将在 http://localhost:4000 上运行")
    
    # 先停止现有容器
    run_command("docker-compose down", capture=False, check=False)
    time.sleep(1)
    
    # 启动新容器
    cmd = "docker-compose up --build -d"
    result = run_command(cmd, capture=False, check=False)
    
    if result is None:
        print("\n等待Jekyll构建网站...")
        time.sleep(5)
        
        # 检查容器状态
        status = run_command("docker-compose ps", capture=True, check=False)
        print(status)
        
        print("\n" + "="*80)
        print("✓ Docker已启动！")
        print("="*80)
        print(f"\n请在浏览器中访问: http://localhost:4000")
        print("检查网站是否正确显示了PR的内容")
        print("\n提示: 可能需要等待1-2分钟让Jekyll完成构建")
        return True
    return False

def get_current_branch():
    """获取当前Git分支"""
    return run_command("git branch --show-current", capture=True)

def review_pull_request(pr, pr_index, total_prs):
    """审查单个Pull Request"""
    pr_number = pr['number']
    pr_title = pr['title']
    pr_head_ref = pr['head']['ref']
    pr_head_label = pr['head']['label']
    pr_head_repo = pr['head']['repo']
    pr_author = pr['user']['login']
    temp_branch = f"review-pr-{pr_number}"
    
    print(f"\n{'='*80}")
    print(f"审查进度: {pr_index}/{total_prs}")
    print(f"PR #{pr_number}: {pr_title}")
    print(f"作者: {pr_author}")
    print(f"分支: {pr_head_label} -> {pr['base']['ref']}")
    print(f"{'='*80}\n")
    
    # 保存原始分支
    original_branch = get_current_branch()
    print(f"当前分支: {original_branch}")
    
    try:
        # 步骤1: 创建临时分支
        print(f"\n[1/7] 创建临时审查分支: {temp_branch}")
        run_command(f"git checkout -b {temp_branch}", capture=False)
        print(f"✓ 已创建临时分支")
        
        # 步骤2: 确保main是最新的
        print(f"\n[2/7] 更新main分支")
        run_command(f"git fetch origin main", capture=False)
        print(f"✓ main分支已更新")
        
        # 步骤3: 合并main到临时分支
        print(f"\n[3/7] 合并main到临时分支")
        result = run_command(f"git merge origin/main --no-edit", capture=False, check=False)
        if result is None:
            print("警告: 合并main时可能有冲突")
            print("正在尝试继续...")
        
        # 步骤4: 获取并合并PR
        print(f"\n[4/7] 获取并合并PR #{pr_number}")
        if pr_head_repo:
            head_repo_url = pr_head_repo['clone_url']
            run_command(f"git fetch {head_repo_url} {pr_head_ref}", capture=False)
            merge_result = run_command(f"git merge FETCH_HEAD --no-edit", capture=False, check=False)
        else:
            run_command(f"git fetch origin pull/{pr_number}/head", capture=False)
            merge_result = run_command(f"git merge FETCH_HEAD --no-edit", capture=False, check=False)
        
        if merge_result is None:
            print("⚠️  合并时可能有冲突")
            conflict_check = run_command("git diff --name-only --diff-filter=U", capture=True, check=False)
            if conflict_check:
                print(f"\n冲突文件:")
                print(conflict_check)
                print("\n由于有合并冲突，跳过此PR")
                cleanup_and_return(temp_branch, original_branch)
                return "skip"
        
        print(f"✓ PR已合并到临时分支")
        
        # 检查修改的文件，只保留 assets/ 和 _posts/ 目录下的修改
        print(f"\n检查并过滤修改的文件...")
        changed_files = run_command("git diff --name-only origin/main", capture=True, check=False)
        if changed_files:
            files = changed_files.strip().split('\n')
            valid_files = []
            invalid_files = []
            
            for f in files:
                f = f.strip()
                if not f:
                    continue
                if f.startswith('assets/') or f.startswith('_posts/'):
                    valid_files.append(f)
                else:
                    invalid_files.append(f)
            
            if invalid_files:
                print(f"\n⚠️  发现 {len(invalid_files)} 个不允许修改的文件:")
                for f in invalid_files[:5]:  # 只显示前5个
                    print(f"  - {f}")
                if len(invalid_files) > 5:
                    print(f"  ... 还有 {len(invalid_files) - 5} 个")
                
                if not valid_files:
                    print(f"\n❌ PR不包含允许的文件修改")
                    print(f"只允许修改 assets/ 和 _posts/ 目录")
                    print(f"\n自动跳过此PR")
                    cleanup_and_return(temp_branch, original_branch)
                    return "skip"
                
                print(f"\n🔧 正在过滤，只保留 {len(valid_files)} 个有效文件...")
                
                # 重置到origin/main，然后只保留允许的文件
                run_command("git reset --hard origin/main", capture=False, check=False)
                
                # 从PR获取并只checkout允许的文件
                if pr_head_repo:
                    head_repo_url = pr_head_repo['clone_url']
                    run_command(f"git fetch {head_repo_url} {pr_head_ref}", capture=False, check=False)
                else:
                    run_command(f"git fetch origin pull/{pr_number}/head", capture=False, check=False)
                
                # 只checkout允许的文件
                for vf in valid_files:
                    run_command(f"git checkout FETCH_HEAD -- {vf}", capture=False, check=False)
                
                # 提交过滤后的修改
                run_command("git add -A", capture=False, check=False)
                commit_msg = f"Filtered changes from PR #{pr_number} (only assets/ and _posts/)"
                run_command(f'git commit -m "{commit_msg}"', capture=False, check=False)
                
                print(f"✓ 已过滤不允许的文件，保留有效文件")
        
        print(f"✓ 文件检查完成")
        
        # 步骤5: 启动Docker预览
        print(f"\n[5/7] 启动Docker预览网站")
        if not start_docker():
            print("警告: Docker启动可能失败")
        
        # 步骤6: 等待用户审查
        print(f"\n[6/7] 等待您的审查决定")
        print("\n" + "="*80)
        print("请在浏览器中访问 http://localhost:4000 检查网站")
        print("="*80)
        
        while True:
            decision = input("\n您的决定 [a=接受并合并到main / s=跳过 / q=退出审查]: ").strip().lower()
            
            if decision == 'a':
                # 接受PR
                print("\n准备合并到main分支...")
                cleanup_docker()
                
                # 切换到main并合并
                print("切换到main分支...")
                run_command("git checkout main", capture=False)
                
                print("拉取最新的main...")
                run_command("git pull origin main", capture=False, check=False)
                
                print(f"合并PR #{pr_number}到main...")
                merge_cmd = f"git merge {temp_branch} --no-ff -m 'Merge pull request #{pr_number} from {pr_head_label}\\n\\n{pr_title}'"
                result = run_command(merge_cmd, capture=False, check=False)
                
                if result is None:
                    print(f"✓ PR #{pr_number} 已成功合并到main!")
                    print("\n提醒: 需要运行 'git push origin main' 来推送更改到GitHub")
                    
                    # 清理临时分支
                    print(f"\n清理临时分支 {temp_branch}...")
                    run_command(f"git branch -D {temp_branch}", capture=False, check=False)
                    
                    return "accepted"
                else:
                    print("合并到main时出错")
                    run_command("git merge --abort", capture=False, check=False)
                    cleanup_and_return(temp_branch, original_branch)
                    return "error"
            
            elif decision == 's':
                # 跳过PR
                print(f"\n跳过PR #{pr_number}")
                cleanup_docker()
                cleanup_and_return(temp_branch, original_branch)
                return "skipped"
            
            elif decision == 'q':
                # 退出
                print("\n退出审查流程...")
                cleanup_docker()
                cleanup_and_return(temp_branch, original_branch)
                return "quit"
            
            else:
                print("无效输入，请输入 a (接受), s (跳过), 或 q (退出)")
    
    except Exception as e:
        print(f"\n发生错误: {e}")
        cleanup_docker()
        cleanup_and_return(temp_branch, original_branch)
        return "error"

def cleanup_and_return(temp_branch, original_branch):
    """清理并返回原始分支"""
    try:
        # 检查是否有未提交的更改
        status = run_command("git status --porcelain", capture=True, check=False)
        if status:
            print("\n检测到未提交的更改，正在重置...")
            run_command("git reset --hard", capture=False, check=False)
        
        # 返回原始分支
        current = get_current_branch()
        if current != original_branch:
            print(f"返回到 {original_branch} 分支...")
            run_command(f"git checkout {original_branch}", capture=False, check=False)
        
        # 删除临时分支
        branches = run_command("git branch", capture=True, check=False)
        if temp_branch in branches:
            print(f"删除临时分支 {temp_branch}...")
            run_command(f"git branch -D {temp_branch}", capture=False, check=False)
        
        print("✓ 清理完成")
    except Exception as e:
        print(f"清理时出错: {e}")

def main():
    """主函数"""
    print("=" * 80)
    print("Pull Request 审查与合并工具")
    print("=" * 80)
    
    # 加载Pull Request列表
    prs = load_pull_requests()
    
    if not prs:
        print("没有找到待审核的Pull Request")
        sys.exit(0)
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == 'list':
            display_pull_requests(prs)
            sys.exit(0)
        
        try:
            # 审查单个PR
            pr_index = int(sys.argv[1]) - 1
            if 0 <= pr_index < len(prs):
                result = review_pull_request(prs[pr_index], pr_index + 1, len(prs))
                sys.exit(0 if result == "accepted" else 1)
            else:
                print(f"错误: 请提供1到{len(prs)}之间的数字")
                sys.exit(1)
        except ValueError:
            print("错误: 无效的参数")
            print(f"\n用法: python3 review_and_merge_pr.py [number|list|all]")
            sys.exit(1)
    
    # 显示所有PR
    display_pull_requests(prs)
    
    # 询问审查模式
    print("\n选择审查模式:")
    print("1. 审查单个PR (输入PR编号)")
    print("2. 批量审查所有PR (输入 'all')")
    print("3. 退出 (输入 'q')")
    
    choice = input("\n您的选择: ").strip().lower()
    
    if choice == 'q':
        print("退出")
        sys.exit(0)
    
    elif choice == 'all':
        # 批量审查模式
        print("\n开始批量审查模式...")
        print("您将依次审查每个PR\n")
        
        accepted_count = 0
        skipped_count = 0
        
        for i, pr in enumerate(prs, 1):
            result = review_pull_request(pr, i, len(prs))
            
            if result == "accepted":
                accepted_count += 1
            elif result == "skipped":
                skipped_count += 1
            elif result == "quit":
                break
            
            print("\n" + "="*80)
            print(f"已审查: {i}/{len(prs)} | 已接受: {accepted_count} | 已跳过: {skipped_count}")
            print("="*80)
            
            if i < len(prs) and result != "quit":
                cont = input("\n继续审查下一个PR? (y/n): ").strip().lower()
                if cont != 'y':
                    break
        
        print(f"\n审查完成!")
        print(f"总计: {len(prs)} | 已接受: {accepted_count} | 已跳过: {skipped_count}")
    
    else:
        # 单个PR审查
        try:
            pr_index = int(choice) - 1
            if 0 <= pr_index < len(prs):
                review_pull_request(prs[pr_index], pr_index + 1, len(prs))
            else:
                print(f"错误: 请输入1到{len(prs)}之间的数字")
        except ValueError:
            print("错误: 无效的输入")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在清理...")
        cleanup_docker()
        print("已退出")
        sys.exit(1)

