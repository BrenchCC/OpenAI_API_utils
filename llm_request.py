import base64
import logging
from openai import OpenAI
# from pyarrow.fs import FileSystem

# hdfs_client, _ = FileSystem.from_uri("hdfs:")

logger = logging.getLogger("LLMs_Server")


class LLMServer:
    """
    Unified interface for multiple LLM providers (doubao / qwen / normal).
    Supports: chat, vision chat, embedding.
    Thinking ability is model-specific.
    """

    def __init__(self, base_url: str, api_key: str, model_type: str):
        """
        model_type: "doubao" | "qwen" | "normal" | etc.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model_type = model_type.lower()

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        logger.info(f"LLMServer initialized: model_type={self.model_type}")

    # ---------------------------
    # Thinking 能力控制
    # ---------------------------
    def _build_extra_body(self, enable_thinking: bool = False):
        """Return model-specific extra body."""
        # ----- Doubao -----
        if self.model_type == "doubao":
            return {
                "extra_body": {
                    "thinking": {
                        "type": "enabled" if enable_thinking else "disabled"
                    }
                }
            }

        # ----- Qwen -----
        if self.model_type == "qwen":
            return {
                "extra_body": {
                    "enable_thinking": bool(enable_thinking)
                }
            }

        # ----- Normal (预留扩展接口) -----
        return {
            "extra_body": {
                "thinking_mode": "none"  # placeholder for future extension
            }
        }

    # =====================================================
    # Chat
    # =====================================================
    def chat(self, query, model_name,
             system_prompt=None,
             stream=False,
             enable_thinking=False,
             **kwargs):
        """
        Unified chat interface.
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})

            extra_body = self._build_extra_body(enable_thinking)

            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=stream,
                **extra_body,
                **kwargs
            )

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

            # Token 信息
            try:
                reasoning_content = completion.choices[0].message.reasoning_content
                prompt_tok = completion.usage.prompt_tokens
                completion_tok = completion.usage.completion_tokens
            except:
                reasoning_content = ""
                prompt_tok = ""
                completion_tok = ""

            return reasoning_content, result, prompt_tok, completion_tok

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return None, "dummy_result", "", ""

    # =====================================================
    # Vision Chat
    # =====================================================
    def vision_chat(self,
                    user_prompt,
                    image_paths,
                    model_name,
                    system_prompt=None,
                    image_type="jpeg",
                    enable_thinking=False,
                    **kwargs):
        """
        Unified Vision + Chat inference.
        """
        try:
            # ===== Build Messages =====
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": []}
                ]
            else:
                messages = [{"role": "user", "content": []}]

            # Text prompt
            if user_prompt:
                messages[-1]["content"].append({"type": "text", "text": user_prompt})

            # Each image
            for idx, img_path in enumerate(image_paths):

                if img_path.startswith("hdfs"):
                    with hdfs_client.open_input_file(img_path) as f:
                        img_bytes = f.read()
                else:
                    img_bytes = open(img_path, "rb").read()

                b64 = base64.b64encode(img_bytes).decode()

                if not user_prompt:
                    messages[-1]["content"].append(
                        {"type": "text", "text": f"image {idx}"}
                    )

                messages[-1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{b64}"
                    }
                })

            # thinking
            extra_body = self._build_extra_body(enable_thinking)

            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **extra_body,
                **kwargs
            )

            result = response.choices[0].message.content
            ptok = response.usage.prompt_tokens
            ctok = response.usage.completion_tokens
            return result, ptok, ctok

        except Exception as e:
            logger.error(f"Vision chat error: {e}")
            return "", "", ""

    # =====================================================
    # Embedding（不涉及 thinking）
    # =====================================================
    def embedding(self, image_path, model_name, image_type="jpeg"):
        try:
            img_bytes = open(image_path, "rb").read()
            b64 = base64.b64encode(img_bytes).decode()

            response = self.client.embeddings.create(
                model=model_name,
                input=[{
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{b64}"
                    }
                }]
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    doubao_base_url = "https://ark.cn-beijing.volces.com/api/v3/"
    doubao_api_key = ""
    qwen_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_api_key = ""

    doubao_llm = LLMServer(doubao_base_url, doubao_api_key, model_type = "doubao")
    qwen_llm = LLMServer(qwen_base_url, qwen_api_key, model_type = "qwen")
    # qwen test
    reasoning_content, result, prompt_tok, completion_tok = qwen_llm.chat("你好", model_name = "qwen-plus")
    print(result)

    # doubao test
    
