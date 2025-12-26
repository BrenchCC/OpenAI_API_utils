import os
import io
import cv2
import ast
import math
import json
import time
import base64
import numpy as np
from tqdm import tqdm
from PIL import Image
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Tuple
from volcenginesdkarkruntime import Ark
MODEL_NAME = ""
chat_client = Ark()

def build_chapter_summary_messages():
    system_content = (
        "你是一个用于生成章节剧情概要的助手。**输入变量**：\n\n"
        "* `chapter_frames`：当前待生成概要的视频帧列表（每一章的 frames 输入前会标明对应的chapter_rank）。\n"
        "* `chapters_info`：用于参考的历史章节概要（如有）。\n\n"
        "你的任务：**为 `chapter_frames` 中的每个章节生成一条单行、简洁、准确的剧情 summary**，并**严格只返回一个 JSON 数组**（顶层为数组），数组内对象按 `chapter_frames` 对应的 chapter_rank 顺序与数量一一对应。**绝对禁止**输出除 JSON 以外的任何文字、标记、注释或格式（包括但不限于 Markdown 标题 ###、加粗 **、前导说明、示例文本、回车外的空白行等）。\n\n"
        "生成规则（必须严格遵守）：\n\n"
        "1. **输出格式**：返回完全可解析的 JSON（顶层为数组）。数组元素示例：\n\n"
        "   [{\"chapter_rank\": 1, \"summary\": \"单行纯文本概要\"}, {\"chapter_rank\": 2, \"summary\": \"另一章的纯文本概要\"}]\n\n"
        "   - 必须使用双引号（\"）包裹字段名和字符串。\n"
        "   - 顶层只能是数组，不能返回对象或其它类型。\n"
        "   - 不能包含任何注释、代码块标记或额外文字。\n\n"
        "2. **字段要求**：\n\n"
        "   - 每个对象必须包含且仅包含两个字段：`chapter_rank`（整数）和 `summary`（字符串）。\n"
        "   - `summary` 必须为单行纯文本：不得包含换行符（\\n）、不得包含 Markdown 标记（例如 ###、**、*、>）、不得包含代码字符包裹（例如 ```）或表格符号。\n"
        "   - `summary` 建议长度 20–50 个汉字，必须概括该章的核心冲突/转折或主要事件，语言简洁，不得包含台词格式或括注性说明。\n\n"
        "3. **内容约束**：\n\n"
        "   - 不得在 `summary` 中使用双引号 `\"`；若文本自然包含双引号，请用中文引号或直接删除双引号。\n"
        "   - 不要在 `summary` 中出现：时间戳、章节小标题、序号列表、脚本式对白（如 甲：...）、内心独白标注（如 （内心OS））、注释或编辑指令。\n"
        "   - 可以参考 `chapters_info`，但不得直接复制历史原文的多行或格式化内容，仅做语义参考。\n\n"
        "4. **顺序与覆盖**：\n\n"
        "   - 返回数组顺序必须与 `chapter_frames` 保持一致，每个 `chapter_rank` 精确对应。\n"
        "   - 若 `chapter_frames` 为空，必须返回空数组： []。\n\n"
        "5. **错误处理**：\n\n"
        "   - 如果无法生成任何合法概要，也必须返回空数组 []，不要返回错误信息或任何非 JSON 文本。\n\n"
        "6. **最终校验要求（请在生成时自检并确保）**：\n\n"
        "   - 输出首字符应为 `[`，末字符应为 `]`；且能被标准 JSON 解析库解析为数组。\n"
        "   - 输出中绝对没有 Markdown 标题、加粗符号、额外换行（除了 JSON 必需的格式化空白外可有紧凑一行也可有缩进）。\n"
        "   - 不要在 `summary` 中包含双引号 `\"`。\n\n"
        "示例（仅示范格式，调用时绝对禁止额外文字）：\n\n"
        '[{"chapter_rank":1,"summary":"陆鸿宇得知秦朗少主现身东城，下令清空江北机场并动员大秦集团寻人。秦朗（装傻）与楚月心乘车，楚月心告知家族为利益逼其离婚嫁叶少阳，叮嘱秦朗拿笔钱远离东城保安全。"}, {"chapter_rank":2,"summary":"楚家筹备招待大秦集团，楚家父子计划让秦朗签离婚协议，宣布楚月心与叶少阳订婚。大秦集团陆总抵达，楚家众人热情接待。"}]'
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": []}
    ]
    return messages

def build_summary_messages(chapters):
    return [
        {
            "role": "system",
            "content": "你是一位擅长将长篇剧情浓缩为结构化概要的助手。你必须按照指定JSON结构输出，并严格遵守格式要求。不要有额外文字、解释或非JSON内容。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"以下是该短剧的章节剧情概要：\n{json.dumps(chapters, ensure_ascii=False, indent=2)}\n\n"
                        "请根据这些章节概要，生成一个精炼且完整的前期剧情总结。\n"
                        "要求：\n"
                        "1. 输出必须是有效JSON：{\"full_summary\": \"总结文本\"}\n"
                        "2. full_summary应逻辑清晰，按剧情发展顺序总结，不超过800字。\n"
                        "3. 内容应涵盖主要人物关系、关键事件、冲突与转折、结局走向。\n"
                        "4. 精炼描述，去掉重复或不重要的细节。\n"
                        "5. 不要在JSON外输出任何内容，包括解释或附加文字。\n"
                    )
                }
            ]
        }
    ]

def aggregate_by_scene(image_list):

    agg = {}

    for item in image_list:
        key = item.get('scene_index')
        if key is None:
            continue

        # ensure entry exists
        if key not in agg:
            agg[key] = {
                'scene_meta': {
                    'scene_frame_interval': item.get('scene_frame_interval'),
                    'scene_time_interval': item.get('scene_time_interval'),
                    'scene_duration': item.get('scene_duration')
                },
                'images': []
            }

        # append frame info (keep cut_time as string to be consistent)
        agg[key]['images'].append({
            'cut_time': item.get('cut_time'),
            'frame_idx': item.get('frame_idx'),
            # 'image_path': os.path.join("/mnt/bn/yg-video-volume/comment_gen_pipeline/", f"{item.get('image')}"),
            'image_path': item.get('image')
        })

    # sort images inside each scene by numeric cut_time
    for key, val in agg.items():
        try:
            val['images'] = sorted(val['images'], key=lambda x: float(x.get('cut_time', 0)))
        except Exception:
            # if conversion fails, keep original order
            pass

    return agg

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

def image_to_resized_base64(image_path, max_pixels: int = 14 * 14 * 4 * 1280):
    img = Image.open(image_path).convert('RGB')
    original_w, original_h = img.size

    # smart resize expects (height, width)
    new_h, new_w = smart_resize(original_h, original_w)
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # encode to base64
    buffered = io.BytesIO()
    resized.save(buffered, format="JPEG", quality=90)
    b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return b64

def Get_playlet_summary(ChapterRank2Scenes, max_chapters=10, max_prev_summarie=3, max_pixels=128 * 28 * 28, system_prompt="", total_frames_per_batch=500, MAX_RETRIES=3):
    chapter_buffer, chapters_info = [], []
    total_frames = 0

    if '.md' in system_prompt:
        with open(system_prompt, 'r', encoding='utf-8') as f:
            system_prompt = f.read()

    for chapter_rank in range(1, max_chapters+1):
        scenes = ChapterRank2Scenes.get(str(chapter_rank), {})
        image_fps = []
        # collect representative frame per scene (mid)
        for scene_index, scene_info in scenes.items():
            images = scene_info.get('images', [])
            if not images:
                continue
            images_sampled = images[len(images) // 2]
            image_fps.append(images_sampled.get('image_path'))

        frames_b64 = []
        for image_path in image_fps:
            try:
                image_bs64 = image_to_resized_base64(image_path, max_pixels=max_pixels)
            except Exception as e:
                # skip problematic image, continue
                continue
            frames_b64.append(image_bs64)

        total_frames += len(frames_b64)
        chapter_buffer.append({
            "chapter_rank": chapter_rank,
            "frames_b64": frames_b64
        })

        # batching logic preserved from original
        if total_frames >= total_frames_per_batch or len(chapter_buffer) == 5 or chapter_rank == max_chapters:
            # build messages
            messages = build_chapter_summary_messages()
            for ch in chapter_buffer:
                messages[-1]['content'].append({
                    "type": "text",
                    "text": f"下面是第 {ch['chapter_rank']} 集的视频帧："
                })
                for idx, img_b64 in enumerate(ch["frames_b64"]):
                    messages[-1]['content'].append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_b64}",
                        "image_pixel_limit": {"max_pixels": 100352}
                    })
            
            prev_info_text = ""
            if chapters_info:
                prev_summaries = chapters_info[-max_prev_summarie:]
                prev_info_text = "以下是前面剧集的概要信息，以供参考：\n"
                for prev in prev_summaries:
                    prev_info_text += f"第{prev.get('chapter_rank')}章节概要：{prev.get('summary')}\n"
            # append prev info as text block if exists
            if prev_info_text:
                messages[-1]['content'].append({"type": "text", "text": prev_info_text + '\n'})
            
            # call API with retry/backoff
            backoff = 2
            output = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = chat_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        timeout=120  # optional: set per-request timeout
                    )
                    output = response.choices[0].message.content
                    break
                except Exception as e:
                    # simple backoff and retry
                    try:
                        if hasattr(e, 'response') and e.response is not None:
                            print(f"Response body: {getattr(e, 'response')}")
                    except Exception:
                        pass
                    time.sleep(backoff * (attempt + 1))

            if output is None:
                # skip this batch on persistent failure
                chapter_buffer = []
                total_frames = 0
                continue

            output = output.replace('```json', '').replace('```', '').strip()

            try:
                parsed = json.loads(output)
            except json.JSONDecodeError:
                # if parsing fails, skip this batch
                chapter_buffer = []
                total_frames = 0
                continue

            if isinstance(parsed, (list, tuple)):
                for rec in parsed:
                    if isinstance(rec, dict):
                        chapters_info.append(rec)
                    else:
                        chapters_info.append({"raw": rec})
            # reset buffer
            chapter_buffer = []
            total_frames = 0

    messages = build_summary_messages(chapters_info)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = chat_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
            )
            # 解析返回
            content = response.choices[0].message.content
            json_response = json.loads(content)
            full_summary = json_response.get('full_summary')
            if full_summary is None:
                raise ValueError("response JSON missing 'full_summary'")
            break
        except Exception as e:
            last_exception = e
            print(f"Attempt {attempt} failed for full_summary generation: {e}")
            time.sleep(2.0 * attempt)
    
    # return {'chapters_info': chapters_info, 'full_summary': full_summary}
    return full_summary

def Get_playlet_characters(ChapterRank2Scenes, max_chapters=5, max_pixels=128 * 28 * 28, total_frames_per_batch=500, MAX_RETRIES=3):
    chapter_buffer, characters_info = [], []
    total_frames = 0
    Chapter2Idx2Path = defaultdict(dict)
    for chapter_rank in range(1, max_chapters+1, 1):
        scenes = ChapterRank2Scenes.get(str(chapter_rank), {})
        image_fps, chapter_frames_idx = [], []
        idx = 0
        for scene_index, scene_info in scenes.items():
            scene_duration = scene_info['scene_meta']['scene_duration']
            images = scene_info['images']
            if float(scene_duration) > 3.0:
                image_fps.append(images[0]['image_path'])
                chapter_frames_idx.append(idx)
                Chapter2Idx2Path[chapter_rank][idx] = images[0]['image_path']
                idx+=1
                image_fps.append(images[-1]['image_path'])
                chapter_frames_idx.append(idx)
                Chapter2Idx2Path[chapter_rank][idx] = images[-1]['image_path']
                idx+=1
            else:
                images_sampled = images[len(images) // 2]
                image_fps.append(images_sampled['image_path'])
                chapter_frames_idx.append(idx)
                Chapter2Idx2Path[chapter_rank][idx] = images_sampled['image_path']
                idx+=1
        
        frames_b64 = []
        for image_path in image_fps:
            image_bs64 = image_to_resized_base64(image_path, max_pixels=max_pixels)
            frames_b64.append(image_bs64)

        total_frames += len(frames_b64)
        chapter_buffer.append({
            "chapter_rank": chapter_rank,
            "frames_b64": frames_b64,
            'chapter_frames_idx': chapter_frames_idx})
        
        if total_frames >= total_frames_per_batch or len(chapter_buffer) == max_chapters or chapter_rank == max_chapters:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是角色识别与定位助手。任务：从下面提供的视频帧中识别出最多3位主要人物，"
                        "并为每位人物返回不超过2个标志性帧（chapter_rank 与 frame_idx）。\n\n"
                        "输出要求（严格）：仅返回一个有效的 JSON 对象，格式如下：\n"
                        "{\n  \"characters\": [\n    {\n      \"name\": \"主要人物名\",\n      \"role_title\": \"角色头衔或人物背景（简洁）\",\n      \"iconic_frames\": [\n        {\"chapter_rank\": 7, \"frame_idx\": \"2\"},\n        {\"chapter_rank\": 8, \"frame_idx\": \"35\"}\n      ]\n    },\n    ...\n  ]\n}\n\n"
                        "约束：\n"
                        "1) 每个 characters 元素必须包含 name, role_title, iconic_frames 三个字段；\n"
                        "2) name应使用剧中人名，例如：\"张旭东\"、\"赵德柱\"，配角若没有提及姓名的，其名字带上相关主角的名字，例如：\"张旭东母亲\"等；\n"
                        "3) 每个 iconic_frames 最多包含 2 条；frame_idx 必须为字符串；\n"
                        "4) 只返回 JSON，不要有任何多余的说明、解释或文本；\n"
                        "5) 选择标志性帧时，优先满足：人物在画面清晰可见、正脸、无遮挡；若帧含有明显字幕/场景提示更优。\n"
                    )
                },
                {
                    "role": "user",
                    "content": []
                }
            ]
            for ch in chapter_buffer:
                # 章节提示文字
                messages[-1]['content'].append({
                    "type": "text",
                    "text": f"下面是第 {ch['chapter_rank']} 集的视频帧："})
                for idx, img_b64 in enumerate(ch["frames_b64"]):
                    # 短剧帧信息
                    frame_idx_token = f"chapter_rank: {ch['chapter_rank']}, frame_index: {ch['chapter_frames_idx'][idx]}"
                    messages[-1]['content'].append({
                        "type": "text",
                        "text": frame_idx_token
                    })
                    messages[-1]['content'].append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_b64}", 
                        'image_pixel_limit': {"max_pixels": max_pixels}})
            
            # API call with retries
            backoff = 2
            output = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = chat_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                    )
                    output = response.choices[0].message.content
                    break
                except Exception as e:
                    time.sleep(backoff * (attempt + 1))

            if output is None:
                # skip this album if persistent failure
                chapter_buffer = []
                total_frames = 0
                break

            output = output.replace('```json', '').replace('```', '').strip()
            try:
                parsed = eval(output)
            except Exception:
                # parsing failed - skip
                chapter_buffer = []
                total_frames = 0
                break

            characters_info.append(parsed)
            # break out of chapter loop after first successful batch (consistent with original)
            break
    
    try:
        characters = characters_info[0].get('characters', [])
        # prepare tos upload client created earlier
        for ch in characters:
            # only handle first iconic frame (if exists)
            iconic_list = ch.get('iconic_frames', [])
            if not iconic_list:
                continue

            for iconic_frame in iconic_list:
                chapter_rank = iconic_frame.get('chapter_rank')
                frame_idx = iconic_frame.get('frame_idx')
                try:
                    image_path = Chapter2Idx2Path[chapter_rank][int(frame_idx)]
                    iconic_frame['image'] = image_path
                except Exception:
                    # try str/int normalization
                    image_path = None
                    for key in (chapter_rank, str(chapter_rank), int(chapter_rank) if isinstance(chapter_rank, str) and chapter_rank.isdigit() else None):
                        if key in Chapter2Idx2Path:
                            try:
                                image_path = Chapter2Idx2Path[key][int(frame_idx)]
                                break
                            except Exception:
                                image_path = None
                    if image_path is None:
                        continue

        return characters

    except Exception as e:
        print(f"主要人物信息解析失败: {e}")
        return None

def generate_playlet_info(data, summary_kwargs=None, characters_kwargs=None):
    """
    生成短剧的剧情概要和主要人物信息
    
    Args:
        data: 输入数据，包含短剧的帧信息
        summary_kwargs: 剧情概要生成的参数配置
        characters_kwargs: 人物信息生成的参数配置
    
    Returns:
        tuple: (playlet_summary: str, playlet_characters: list)
    """
    if len(data) < 10:
        print('[warn] playlet chapter info <= 10') 

    # 设置默认参数
    if summary_kwargs is None:
        summary_kwargs = {
            "max_chapters": 10,
            "max_prev_summarie": 3,
            "max_pixels": 128 * 28 * 28,
        }
    
    if characters_kwargs is None:
        characters_kwargs = {
            "max_chapters": 6,
            "max_pixels": 128 * 28 * 28,
        }

    # gather images by scene_index | Get rank2scenes
    ChapterRank2Scenes = defaultdict(dict)
    if len(data) >= 10:
        data = data[:10]
    for item in data:
        playlet_id = item['playlet_id']
        chapter_gid = item['episode_gid']
        episode_rank = item['episode_rank']
        frames = item['frames']
        agg = aggregate_by_scene(frames)
        ChapterRank2Scenes[episode_rank] = agg

    start_ts = time.time()
    wait_timeout = 300
    
    # 并发执行两个任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        fut_summary = executor.submit(Get_playlet_summary, ChapterRank2Scenes, **summary_kwargs)
        fut_characters = executor.submit(Get_playlet_characters, ChapterRank2Scenes, **characters_kwargs)

    # 获取结果
    try:
        playlet_summary = fut_summary.result(timeout=wait_timeout)
    except Exception as e:
        print(f"[WARN] Get_playlet_summary failed: {e}")
        playlet_summary = ''
    
    try:
        playlet_characters = fut_characters.result(timeout=wait_timeout)
    except Exception as e:
        print(f"[WARN] Get_playlet_characters failed: {e}")
        playlet_characters = []

    elapsed = time.time() - start_ts
    print(f"[INFO] tasks finished in {elapsed:.2f}s")
    
    playlet_id = data[0]['playlet_id']

    playlet_background = construct_playlet_info(playlet_id, playlet_summary, playlet_characters)
    return playlet_background

def construct_playlet_info(playlet_id, playlet_summary: str, playlet_characters: list):
    """
    构造短剧的信息字典
    
    Args:
        playlet_id: 短剧的ID
        playlet_summary: 短剧的剧情概要
        playlet_characters: 短剧的主要人物信息
    
    Returns:
        dict: 短剧的信息字典
    """
    playlet_id = str(playlet_id)
    temp_characters = []
    for c in playlet_characters:
        role = c.get("name", "")
        role_description = c.get("role_title", "")
        role_img = c.get("iconic_frames", [])
        temp_characters.append({
            "role": role,
            "role_description": role_description,
            "role_img": role_img,
        })
    playlet_info = {
        "playlet_id": playlet_id,
        "playlet_description": playlet_summary,
        "playlet_role_list": temp_characters,
    }
    return playlet_info
    

# 使用示例
if __name__ == '__main__':

    """
    自定义参数配置
    custom_summary_kwargs = {
        "max_chapters": 8,
        "max_prev_summarie": 2,
        "max_pixels": 128 * 28 * 28,
    }
    
    custom_characters_kwargs = {
        "max_chapters": 5,
        "max_pixels": 128 * 28 * 28,
    }
    """
    # 加载数据
    with open('./examples/frames_demo.json') as f:
        data = json.load(f)
    
    # 调用接口函数
    playlet_background = generate_playlet_info(data)
    print('playlet_background:', playlet_background)
    
    with open('./examples/playlet_background_demo.json', 'w') as f:
        json.dump(playlet_background, f, ensure_ascii=False, indent=4)
