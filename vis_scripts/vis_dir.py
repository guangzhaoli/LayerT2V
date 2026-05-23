import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from tqdm import tqdm

# ================= 配置区域 =================
# 根目录
ROOT_DIR = "/home/notebook/data/group/ckr/Data/layer/layert2v_0116/test_outputs/16k-6kstep-infer"

# 视频片段对应的后缀与显示标签
SUFFIX_AND_LABELS = [
    ("_background.mp4", "background"),
    ("_fg_masked.mp4", "fg_mask"),
    ("_foreground.mp4", "foreground"),
    ("_full_video.mp4", "full_video"),
    ("_mask.mp4", "mask")
]

TARGET_FPS = 8
TARGET_HEIGHT = 256   # 视频部分的高度
TEXT_AREA_HEIGHT = 50 # 底部文字区的高度（调高一点以容纳大字体）
FONT_SIZE = 24        # 字体大小
# ==========================================

def get_font(size):
    """
    尝试加载系统字体，如果失败则使用默认字体
    """
    # Linux/Notebook 环境下常见的字体路径
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "arial.ttf", # Windows 或本地
        "Arial.ttf"
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    
    # 如果都找不到，尝试直接通过名字加载（依赖系统配置）
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except:
        pass

    logger.warning("未找到常用字体文件，将使用默认较小字体。建议在系统中安装 DejaVuSans 或指定字体路径。")
    return ImageFont.load_default()

# 初始化字体对象
FONT = get_font(FONT_SIZE)

def read_and_resize_video_frames(video_path, target_height):
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        logger.debug(f"无法打开或找不到视频文件: {video_path}")
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

def create_labeled_frame(frame_np, label_text, text_h, font):
    """
    在视频帧下方拼接一个黑色区域并写入文字
    """
    v_h, v_w = frame_np.shape[:2]
    canvas = Image.new("RGB", (v_w, v_h + text_h), (0, 0, 0))
    video_img = Image.fromarray(frame_np)
    canvas.paste(video_img, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # 使用指定字体计算文字大小
    try:
        # Pillow >= 9.2.0
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # 旧版 Pillow
        tw, th = draw.textsize(label_text, font=font)
    
    tx = (v_w - tw) // 2
    # 垂直居中
    ty = v_h + (text_h - th) // 2 - 4 
    
    draw.text((tx, ty), label_text, fill=(255, 255, 255), font=font)
    return np.array(canvas)

def process_single_folder(source_dir):
    """
    处理单个文件夹：自动探测前缀 -> 读取视频 -> 合成 GIF
    """
    # 1. 获取 VID (文件夹名称) 并构建输出文件名
    # 使用 rstrip 处理路径末尾可能存在的 '/'
    vid_name = os.path.basename(source_dir.rstrip(os.sep))
    output_filename = f"{vid_name}_vis.gif"
    output_path = os.path.join(source_dir, output_filename)
    
    if os.path.exists(output_path):
        return

    # 2. 自动探测文件前缀
    mp4_files = [f for f in os.listdir(source_dir) if f.endswith('.mp4')]
    ref_suffix = SUFFIX_AND_LABELS[0][0] # e.g. "_background.mp4"
    ref_file = next((f for f in mp4_files if f.endswith(ref_suffix)), None)
    
    prefix = ""
    if ref_file:
        prefix = ref_file[:-len(ref_suffix)]
    else:
        # Fallback 逻辑
        if os.path.exists(os.path.join(source_dir, "output" + ref_suffix)):
            prefix = "output"
        else:
            # logger.warning(f"跳过目录 {vid_name}: 无法识别文件前缀")
            return

    # 3. 读取并处理视频
    all_videos_frames = []
    active_labels = []

    for suffix, label in SUFFIX_AND_LABELS:
        filename = prefix + suffix
        file_path = os.path.join(source_dir, filename)
        
        frames = read_and_resize_video_frames(file_path, TARGET_HEIGHT)
        if frames:
            all_videos_frames.append(frames)
            active_labels.append(label)

    if not all_videos_frames:
        return

    # 4. 合成 GIF
    min_frames = min(len(vs) for vs in all_videos_frames)
    gif_frames = []
    
    for i in range(min_frames):
        row_segments = []
        for v_idx in range(len(all_videos_frames)):
            raw_frame = all_videos_frames[v_idx][i]
            # 传入全局 FONT 对象
            labeled_seg = create_labeled_frame(raw_frame, active_labels[v_idx], TEXT_AREA_HEIGHT, FONT)
            row_segments.append(labeled_seg)
        
        combined_row = np.hstack(row_segments)
        pil_img = Image.fromarray(combined_row)
        pil_img_opt = pil_img.convert("P", palette=Image.ADAPTIVE, colors=256)
        gif_frames.append(pil_img_opt)

    # 5. 保存
    if gif_frames:
        try:
            gif_frames[0].save(
                output_path,
                save_all=True,
                append_images=gif_frames[1:],
                optimize=True,
                duration=int(1000 / TARGET_FPS),
                loop=0
            )
        except Exception as e:
            logger.error(f"保存失败 {source_dir}: {e}")

def main():
    if not os.path.exists(ROOT_DIR):
        logger.error(f"根目录不存在: {ROOT_DIR}")
        return

    subdirs = [os.path.join(ROOT_DIR, d) for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    subdirs.sort() # 排序一下看起来舒服
    
    logger.info(f"发现 {len(subdirs)} 个子目录，开始处理...")
    logger.info(f"输出格式示例: {subdirs[0]}/{os.path.basename(subdirs[0])}_vis.gif")

    for subdir in tqdm(subdirs, desc="Generating GIFs"):
        process_single_folder(subdir)
    
    logger.success("所有任务处理完成！")

if __name__ == "__main__":
    main()