import argparse
import json
import os
import torch
import torch.distributed as dist
from tqdm import tqdm
import sys

# 将上级目录加入路径，确保能 import wan
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wan.layered_t2v import create_layered_pipeline

def setup_distributed():
    """初始化分布式环境，用于获取 global_rank 和 world_size 进行数据切分"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank

def main():
    parser = argparse.ArgumentParser(description="Layered Video Batch Inference")
    # 模型路径相关
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to input jsonl file")
    
    # Mask 相关设置 (保持与 run_inference.sh 一致)
    parser.add_argument("--mask_mode", type=str, default="vae", 
                        choices=["vae", "downsample", "downsample-project", "vae-project", "vae-lora", "mask-vae-project", "mask-vae-joint"])
    parser.add_argument("--mask_vae_lora_path", type=str, default=None)
    parser.add_argument("--mask_vae_path", type=str, default=None)
    parser.add_argument("--mask_vae_proj_path", type=str, default=None)
    
    # 4D Rope 设置
    parser.add_argument("--use_4d_rope", action="store_true")
    parser.add_argument("--no_4d_rope", action="store_false", dest="use_4d_rope")
    parser.set_defaults(use_4d_rope=True)
    parser.add_argument("--rope_dim_ratios", type=str, default=None)
    
    # 生成参数
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--use_ema", action="store_true")

    args = parser.parse_args()

    # 1. 初始化分布式环境
    global_rank, world_size, local_rank = setup_distributed()
    torch.cuda.set_device(local_rank)

    # 2. 读取并切分数据 (Data Parallelism)
    all_data = []
    with open(args.jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    # 根据 rank 切分数据：例如 8 张卡，rank 0 处理 0, 8, 16...
    my_data = all_data[global_rank::world_size]
    
    print(f"[Rank {global_rank}] Processing {len(my_data)}/{len(all_data)} samples on Device {local_rank}")

    # 3. 解析 Rope 参数
    rope_dim_ratios = None
    if args.rope_dim_ratios:
        rope_dim_ratios = tuple(int(x) for x in args.rope_dim_ratios.split(","))

    # 4. 初始化 Pipeline
    # 注意：这里强制传入 rank=0。
    # 原因：LayeredWanT2V.generate 方法中判断 self.rank == 0 才会解码输出。
    # 在数据并行模式下，我们希望每张卡都解码自己负责的样本。
    pipeline = create_layered_pipeline(
        model_path=args.model_path,
        lora_path=args.lora_path,
        use_ema=args.use_ema,
        mask_mode=args.mask_mode,
        use_4d_rope=args.use_4d_rope,
        rope_dim_ratios=rope_dim_ratios,
        device_id=local_rank, 
        mask_vae_path=args.mask_vae_path,
        mask_vae_proj_path=args.mask_vae_proj_path,
        mask_vae_lora_path=args.mask_vae_lora_path,
        # 这里的 rank 参数如果不传默认为 0，显式传 0 确保每张卡都作为“主卡”工作
    )
    # 修改 pipeline 的 rank 属性，再次确保 generate 时通过检查
    pipeline.rank = 0 

    # 5. 循环推理
    for item in tqdm(my_data, desc=f"Rank {global_rank}", position=local_rank):
        # 提取 Prompt
        prompt = item.get("overall_description", "")
        fg_prompt = item.get("foreground_prompt", "")
        bg_prompt = item.get("background_prompt", "")
        
        # 提取文件名作为 ID
        vid_path = item.get("vid", "")
        video_name = os.path.splitext(os.path.basename(vid_path))[0]
        
        # 构建当前样本的输出目录
        sample_output_dir = os.path.join(args.output_dir, video_name)
        
        # 如果已经存在，可以选择跳过 (可选)
        if os.path.exists(sample_output_dir) and len(os.listdir(sample_output_dir)) > 0:
            print(f"[Rank {global_rank}] Skipping {video_name}, already exists.")
            continue

        try:
            outputs = pipeline.generate(
                input_prompt=prompt,
                fg_prompt=fg_prompt,
                bg_prompt=bg_prompt,
                size=(args.width, args.height),
                frame_num=args.frames,
                sampling_steps=args.steps,
                seed=args.seed,
            )
            
            # 保存
            pipeline.save_outputs(outputs, sample_output_dir, fps=args.fps, prefix=video_name)
            
        except Exception as e:
            print(f"[Rank {global_rank}] Error processing {video_name}: {e}")

    print(f"[Rank {global_rank}] Finished.")

if __name__ == "__main__":
    main()