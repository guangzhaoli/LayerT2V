#!/bin/bash
# Copyright 2024-2025 LayerT2V Authors. All rights reserved.
#
# Wandb 离线日志同步脚本
#
# 使用说明：
#   在可以联网的 CPU 集群上运行此脚本，将 GPU 集群上离线保存的
#   Wandb 日志同步到 Wandb 服务器。
#
# 用法：
#   1. 同步单个运行：
#      ./scripts/sync_wandb.sh --dir ./logs/train/wandb/run_name
#
#   2. 同步目录下所有运行：
#      ./scripts/sync_wandb.sh --all ./logs/train/wandb
#
#   3. 指定项目和实体（可选）：
#      ./scripts/sync_wandb.sh --dir ./logs/train/wandb/run_name --project layert2v --entity your_team
#
# 前置条件：
#   - 已安装 wandb: pip install wandb
#   - 已登录 wandb: wandb login
#

set -e

# 默认值
SYNC_DIR=""
SYNC_ALL=""
WANDB_PROJECT=""
WANDB_ENTITY=""
DRY_RUN=false
CLEAN_AFTER_SYNC=false

# 帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --dir DIR          同步指定目录的 Wandb 日志"
    echo "  --all DIR          同步目录下所有子目录的 Wandb 日志"
    echo "  --project NAME     指定 Wandb 项目名称（可选）"
    echo "  --entity NAME      指定 Wandb 实体/团队名称（可选）"
    echo "  --dry-run          仅显示将要同步的目录，不实际执行"
    echo "  --clean            同步成功后删除本地日志"
    echo "  -h, --help         显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --dir ./logs/train/wandb/my_run"
    echo "  $0 --all ./logs/train/wandb"
    echo "  $0 --all ./logs --project layert2v --entity my_team"
    echo ""
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dir)
            SYNC_DIR="$2"
            shift 2
            ;;
        --all)
            SYNC_ALL="$2"
            shift 2
            ;;
        --project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --entity)
            WANDB_ENTITY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --clean)
            CLEAN_AFTER_SYNC=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查 wandb 是否安装
if ! command -v wandb &> /dev/null; then
    echo "错误: wandb 未安装。请运行: pip install wandb"
    exit 1
fi

# 检查是否已登录
if ! wandb verify 2>/dev/null; then
    echo "警告: wandb 可能未登录。如果同步失败，请运行: wandb login"
fi

# 构建 wandb sync 命令的额外参数
EXTRA_ARGS=""
if [[ -n "$WANDB_PROJECT" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --project $WANDB_PROJECT"
fi
if [[ -n "$WANDB_ENTITY" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --entity $WANDB_ENTITY"
fi

# 同步单个目录
sync_single() {
    local dir="$1"

    if [[ ! -d "$dir" ]]; then
        echo "错误: 目录不存在: $dir"
        return 1
    fi

    # 查找 wandb-* 目录（离线运行目录）
    local wandb_runs=$(find "$dir" -maxdepth 2 -type d -name "offline-run-*" -o -name "run-*" 2>/dev/null)

    if [[ -z "$wandb_runs" ]]; then
        # 尝试直接同步目录
        echo "同步: $dir"
        if [[ "$DRY_RUN" == "true" ]]; then
            echo "  [dry-run] wandb sync $EXTRA_ARGS $dir"
        else
            wandb sync $EXTRA_ARGS "$dir" && {
                if [[ "$CLEAN_AFTER_SYNC" == "true" ]]; then
                    echo "  清理: $dir"
                    rm -rf "$dir"
                fi
            }
        fi
    else
        for run_dir in $wandb_runs; do
            echo "同步: $run_dir"
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  [dry-run] wandb sync $EXTRA_ARGS $run_dir"
            else
                wandb sync $EXTRA_ARGS "$run_dir" && {
                    if [[ "$CLEAN_AFTER_SYNC" == "true" ]]; then
                        echo "  清理: $run_dir"
                        rm -rf "$run_dir"
                    fi
                }
            fi
        done
    fi
}

# 同步所有子目录
sync_all() {
    local base_dir="$1"

    if [[ ! -d "$base_dir" ]]; then
        echo "错误: 目录不存在: $base_dir"
        exit 1
    fi

    echo "扫描目录: $base_dir"
    echo ""

    # 查找所有 wandb 相关目录
    local count=0

    # 查找 wandb 目录
    for wandb_dir in $(find "$base_dir" -type d -name "wandb" 2>/dev/null); do
        for run_dir in $(find "$wandb_dir" -maxdepth 2 -type d \( -name "offline-run-*" -o -name "run-*" \) 2>/dev/null); do
            echo "找到: $run_dir"
            sync_single "$run_dir"
            ((count++)) || true
        done
    done

    # 也尝试直接查找 offline-run-* 目录
    for run_dir in $(find "$base_dir" -type d -name "offline-run-*" 2>/dev/null); do
        echo "找到: $run_dir"
        sync_single "$run_dir"
        ((count++)) || true
    done

    echo ""
    echo "同步完成: 共 $count 个运行"
}

# 主逻辑
if [[ -n "$SYNC_DIR" ]]; then
    sync_single "$SYNC_DIR"
elif [[ -n "$SYNC_ALL" ]]; then
    sync_all "$SYNC_ALL"
else
    echo "错误: 请指定 --dir 或 --all 参数"
    echo ""
    show_help
    exit 1
fi

echo ""
echo "完成！"
echo ""
echo "提示："
echo "  - 登录 https://wandb.ai 查看同步的实验"
echo "  - 如需重新同步，可再次运行此脚本"
