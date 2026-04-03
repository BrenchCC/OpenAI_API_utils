import os
import base64

from volcenginesdkarkruntime import Ark


def _ensure_ark_api_key() -> None:
    """EnsureK LLM_API_KEY is available in environment variables."""
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError(
            "Missing LLM_API_KEY in environment. Please export LLM_API_KEY before running."
        )


def call_llm_on_volcengine(input_query, end_point, system_prompt = None, stream = False, reasoning_option = None):
    _ensure_ark_api_key()
    client = Ark(
        base_url = os.environ.get("LLM_API_BASE_URL"),
        api_key = os.environ.get("LLM_API_KEY"),
    )
    try:
        messages = [{"role": "user", "content": input_query}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        if reasoning_option:
            completion = client.chat.completions.create(
                model = end_point,
                messages = messages,
                timeout = 300,
                stream = stream,
                extra_body = {
                    "thinking": {
                        "type": reasoning_option
                    }
                }
            )
        else:
            completion = client.chat.completions.create(
                model = end_point,
                messages = messages,
                timeout = 300,
                stream = stream
            )
        if stream:
            result = ""
            for tok in completion:
                if not tok.choices:
                    continue
                result += tok.choices[0].delta.content
                print(tok.choices[0].delta.content, end="")
        else:
            result = completion.choices[0].message.content
        try:
            reasoning_content = completion.choices[0].message.reasoning_content
            prompt_tok = completion.usage.prompt_tokens
            completion_tok = completion.usage.completion_tokens
        except Exception as e:
            print(e)
            reasoning_content = ""
            prompt_tok, completion_tok = "", ""
            # print(completion.choices[0].message.content)
        if not isinstance(result, str):
            result = "dummy_result"
            reasoning_content = ""
            prompt_tok, completion_tok = "", ""
        return reasoning_content, result, prompt_tok, completion_tok

    except Exception as e:
        print(e)
        return None, "dummy_result", "", ""


def call_vision_embedding(image_path, end_point, image_type="jpeg"):
    _ensure_ark_api_key()
    client = Ark()

    # print("----- multimodal embeddings request -----")
    img_bytes = open(image_path, 'rb').read()
    base64_image = base64.b64encode(img_bytes).decode()

    resp = client.multimodal_embeddings.create(
        model=end_point,
        input=[
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_type};base64,{base64_image}"
                }
            }
        ]
    )
    try:
        return resp.data["embedding"]
    except Exception as e:
        print("Error: ", e)
        return []
