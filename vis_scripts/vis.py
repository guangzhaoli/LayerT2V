# import os
# import cv2
# import numpy as np
# from PIL import Image
# from loguru import logger

# # ================= 配置区域 =================
# # 指定目录路径
# TARGET_DIR = "/home/notebook/data/group/ckr/Data/layer/layert2v_new/outputs/1"

# # 需要可视化的文件名列表（按顺序排列）
# FILE_LIST = [
#     "output_background.mp4",
#     "output_fg_masked.mp4",
#     "output_foreground.mp4",
#     "output_full_video.mp4",
#     "output_mask.mp4"
# ]

# # 输出 GIF 的设置
# OUTPUT_GIF_NAME = "combined_visualization.gif"
# TARGET_FPS = 8
# # 统一缩放高度（宽度自适应），限制高度有助于控制 GIF 体积
# # 如果原视频分辨率很高，建议设置在 240-360 之间
# TARGET_HEIGHT = 256 
# # ==========================================


# def read_and_resize_video_frames(video_path, target_height):
#     """
#     读取视频文件，将每一帧从 BGR 转为 RGB，并按比例缩放到指定高度。
#     """
#     cap = cv2.VideoCapture(video_path)
#     frames = []
    
#     if not cap.isOpened():
#         logger.error(f"无法打开视频文件: {video_path}")
#         return []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # OpenCV 默认读取为 BGR，需转换为 RGB 供 PIL 使用
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # 计算缩放比例，保持宽高比
#         h, w = frame_rgb.shape[:2]
#         aspect_ratio = w / h
#         target_width = int(target_height * aspect_ratio)
        
#         # 使用区域插值进行缩放（适合缩小图像）
#         resized_frame = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
#         frames.append(resized_frame)
    
#     cap.release()
#     return frames

# def create_horizontal_gif(source_dir, file_list, output_name, fps=8, height=256):
#     output_path = os.path.join(source_dir, output_name)
#     logger.info(f"开始处理目录: {source_dir}")

#     # 1. 读取所有视频的帧数据
#     all_videos_frames = []
#     valid_files_count = 0

#     for filename in file_list:
#         file_path = os.path.join(source_dir, filename)
#         if not os.path.exists(file_path):
#             logger.warning(f"文件不存在，跳过: {file_path}")
#             continue
        
#         logger.info(f"正在读取: {filename} ...")
#         frames = read_and_resize_video_frames(file_path, height)
#         if frames:
#             all_videos_frames.append(frames)
#             valid_files_count += 1
#         else:
#              logger.warning(f"视频 {filename} 读取为空或失败。")

#     if valid_files_count == 0:
#         logger.error("没有找到有效的视频文件，无法生成 GIF。")
#         return

#     # 2. 帧数对齐 (强制同步)
#     # 找出所有视频中最小的帧数，以该长度截断所有视频，确保拼接时不会错位
#     min_frames = min(len(vs) for vs in all_videos_frames)
#     logger.info(f"所有视频已读取完毕。将基于最小帧数 [{min_frames}] 进行对齐拼接。")

#     # 3. 逐帧拼接并转换为 PIL 图像
#     gif_frames = []
#     for i in range(min_frames):
#         # 取出每个视频的第 i 帧
#         current_frame_row = [video[i] for video in all_videos_frames]
        
#         # 使用 numpy 横向拼接 (Horizontal Stack)
#         combined_image_np = np.hstack(current_frame_row)
        
#         # 转换为 PIL Image
#         pil_img = Image.fromarray(combined_image_np)
        
#         # --- 关键优化 ---
#         # 将图像转换为 'P' 模式 (调色板模式)，能显著减小 GIF 体积
#         # 使用自适应调色板 (ADAPTIVE) 确保颜色尽可能还原
#         pil_img_optimized = pil_img.convert("P", palette=Image.ADAPTIVE, colors=256)
#         gif_frames.append(pil_img_optimized)

#     # 4. 保存为 GIF
#     if gif_frames:
#         logger.info(f"正在保存 GIF 到: {output_path} (FPS={fps}, Loop=Inf)")
#         # 计算每帧持续时间 (毫秒)
#         duration_ms = int(1000 / fps)
        
#         gif_frames[0].save(
#             output_path,
#             save_all=True,
#             append_images=gif_frames[1:],
#             optimize=True,   # 开启 PIL 内部优化
#             duration=duration_ms,
#             loop=0           # 0 表示无限循环
#         )
#         logger.success("GIF 生成成功！")
#     else:
#         logger.error("GIF 生成失败，没有可用的帧数据。")

# if __name__ == "__main__":
#     # 检查目录是否存在
#     if os.path.exists(TARGET_DIR):
#         create_horizontal_gif(
#             source_dir=TARGET_DIR,
#             file_list=FILE_LIST,
#             output_name=OUTPUT_GIF_NAME,
#             fps=TARGET_FPS,
#             height=TARGET_HEIGHT
#         )
#     else:
#         logger.error(f"目标目录不存在: {TARGET_DIR}")


import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

# ================= 配置区域 =================
TARGET_DIR = "/home/notebook/data/group/ckr/Data/layer/layert2v_0116/test_outputs/16k-7000step/9"

# 文件名与对应的显示标签（配对处理，确保修正了 typo）
FILES_AND_LABELS = [
    ("output_background.mp4", "background"),
    ("output_fg_masked.mp4", "fg_mask"),
    ("output_foreground.mp4", "foreground"),
    ("output_full_video.mp4", "full_video"),
    ("output_mask.mp4", "mask")
]

OUTPUT_GIF_NAME = "labeled_visualization.gif"
TARGET_FPS = 8
TARGET_HEIGHT = 256  # 视频部分的高度
TEXT_AREA_HEIGHT = 40 # 底部文字区的高度
# ==========================================

def read_and_resize_video_frames(video_path, target_height):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        logger.error(f"无法打开视频文件: {video_path}")
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
        resized_frame = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        frames.append(resized_frame)
    cap.release()
    return frames

def create_labeled_frame(frame_np, label_text, text_h):
    """
    在视频帧下方拼接一个黑色区域并写入文字
    """
    v_h, v_w = frame_np.shape[:2]
    # 创建一个新的画布：高度 = 视频高 + 文字区高
    canvas = Image.new("RGB", (v_w, v_h + text_h), (0, 0, 0))
    # 将视频帧贴在上方
    video_img = Image.fromarray(frame_np)
    canvas.paste(video_img, (0, 0))
    
    # 绘制文字
    draw = ImageDraw.Draw(canvas)
    # 计算文字位置（水平居中，垂直在底部区域居中）
    # 注意：这里使用默认字体，如果需要特定字体可使用 ImageFont.truetype
    bbox = draw.textbbox((0, 0), label_text)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    tx = (v_w - tw) // 2
    ty = v_h + (text_h - th) // 2 - 2 # 微调向上移动2像素看起来更居中
    
    draw.text((tx, ty), label_text, fill=(255, 255, 255))
    return np.array(canvas)

def create_horizontal_labeled_gif(source_dir, file_info, output_name, fps, v_height, t_height):
    output_path = os.path.join(source_dir, output_name)
    logger.info(f"开始处理目录: {source_dir}")

    all_videos_frames = []
    active_labels = []

    # 1. 读取视频
    for filename, label in file_info:
        file_path = os.path.join(source_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"跳过不存在的文件: {filename}")
            continue
        
        frames = read_and_resize_video_frames(file_path, v_height)
        if frames:
            all_videos_frames.append(frames)
            active_labels.append(label)

    if not all_videos_frames:
        logger.error("无有效视频数据。")
        return

    # 2. 对齐帧数
    min_frames = min(len(vs) for vs in all_videos_frames)
    
    # 3. 逐帧合成
    gif_frames = []
    for i in range(min_frames):
        row_segments = []
        for v_idx in range(len(all_videos_frames)):
            # 获取该视频的第 i 帧
            raw_frame = all_videos_frames[v_idx][i]
            # 加上底部标签
            labeled_seg = create_labeled_frame(raw_frame, active_labels[v_idx], t_height)
            row_segments.append(labeled_seg)
        
        # 横向拼接所有带标签的段
        combined_row = np.hstack(row_segments)
        
        # 转换并压缩
        pil_img = Image.fromarray(combined_row)
        pil_img_opt = pil_img.convert("P", palette=Image.ADAPTIVE, colors=256)
        gif_frames.append(pil_img_opt)

    # 4. 保存
    if gif_frames:
        logger.info(f"正在保存带标签的 GIF...")
        gif_frames[0].save(
            output_path,
            save_all=True,
            append_images=gif_frames[1:],
            optimize=True,
            duration=int(1000 / fps),
            loop=0
        )
        logger.success(f"完成！保存至: {output_path}")

if __name__ == "__main__":
    if os.path.exists(TARGET_DIR):
        create_horizontal_labeled_gif(
            TARGET_DIR, FILES_AND_LABELS, OUTPUT_GIF_NAME, 
            TARGET_FPS, TARGET_HEIGHT, TEXT_AREA_HEIGHT
        )