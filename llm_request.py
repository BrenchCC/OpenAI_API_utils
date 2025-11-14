import os
import base64
from openai import OpenAI
from pyarrow.fs import FileSystem

# ===== HDFS 依然保留 =====
hdfs_client, _ = FileSystem.from_uri('hdfs:')

# ===== 设置 OpenAI Key =====
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

client = OpenAI()


# ======================
#  Chat LLM
# ======================
def call_llm_on_openai(input_query, model_name, system_prompt=None, stream=False, reasoning_option=None):
    try:
        messages = [{"role": "user", "content": input_query}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        extra_args = {}
        if reasoning_option:
            # OpenAI reasoning models（o3 / o1）写法
            extra_args["reasoning"] = {"type": reasoning_option}

        # ===== 调用 =====
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=stream,
            **extra_args
        )

        # ===== 流式 =====
        if stream:
            result = ""
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="")
                    result += delta.content
        else:
            result = completion.choices[0].message.content

        # ===== Token 信息 =====
        try:
            reasoning_content = completion.choices[0].message.reasoning_content
            prompt_tok = completion.usage.prompt_tokens
            completion_tok = completion.usage.completion_tokens
        except:
            reasoning_content = ""
            prompt_tok, completion_tok = "", ""

        if not isinstance(result, str):
            result = "dummy_result"

        return reasoning_content, result, prompt_tok, completion_tok

    except Exception as e:
        print("OpenAI error:", e)
        return None, "dummy_result", "", ""


# ======================
#  Vision + Chat
# ======================
def vision_inference(user_prompt, images, model_name, system_prompt=None, messages=None, image_type="jpeg", reasoning_option=None):

    if not messages:
        if system_prompt:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": []}
            ]
        else:
            messages = [{"role": "user", "content": []}]

        # ===== 构造内容 =====
        if user_prompt:
            messages[-1]["content"].append({"type": "text", "text": user_prompt})

        for idx, image in enumerate(images):
            if image.startswith("hdfs"):
                with hdfs_client.open_input_file(image) as f:
                    img_bytes = f.read()
            else:
                img_bytes = open(image, "rb").read()

            b64 = base64.b64encode(img_bytes).decode()

            if not user_prompt:
                messages[-1]["content"].append({"type": "text", "text": f"第 {idx} 帧图"})

            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_type};base64,{b64}"
                },
            })

    try:
        extra_args = {}
        if reasoning_option:
            extra_args["reasoning"] = {"type": reasoning_option}

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            **extra_args
        )

        result = resp.choices[0].message.content
        prompt_tok = resp.usage.prompt_tokens
        completion_tok = resp.usage.completion_tokens
        return result, prompt_tok, completion_tok

    except Exception as e:
        print("vision_inference error:", e)
        return "", "", ""


# ======================
#  Vision Embedding
# ======================
def call_vision_embedding(image_path, model_name, image_type="jpeg"):
    try:
        img_bytes = open(image_path, "rb").read()
        b64 = base64.b64encode(img_bytes).decode()

        # ===== OpenAI embeddings =====
        resp = client.embeddings.create(
            model=model_name,
            input=[{
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_type};base64,{b64}"
                }
            }]
        )

        return resp.data[0].embedding

    except Exception as e:
        print("embedding error:", e)
        return []
