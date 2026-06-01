"""Llama 3.2 11B Vision Instruct — vLLM on Modal (A10, FP16).

OpenAI-compatible multimodal endpoint for table image description.
Accepts base64-encoded images in the OpenAI chat completions format.

Usage:
    modal deploy modal/vision_serve.py        # deploy
    modal run modal/vision_serve.py           # test locally
"""

import modal

MINUTES = 60

MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MODEL_REVISION = "7c7feb5c3f0448f247a8aebac87e5d4b1b6a5f6e"

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "5",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("prism-vision-serve")

VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu="A10",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=50)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--served-model-name",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--async-scheduling",
        "--no-enforce-eager",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.92",
        "--limit-mm-per-prompt",
        "image=5",
    ]

    print("Starting vLLM Vision:", " ".join(cmd))
    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
async def main():
    import base64
    import io

    import aiohttp
    from PIL import Image

    url = await serve.get_web_url.aio()
    print(f"Server URL: {url}")

    async with aiohttp.ClientSession(base_url=url) as session:
        print("Health check...")
        async with session.get("/health", timeout=10 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed: {resp.status}"
        print("Health check passed!")

        img = Image.new("RGB", (200, 100), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64_img = base64.b64encode(buf.getvalue()).decode()

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_img}"
                            },
                        },
                    ],
                }
            ],
            "model": MODEL_NAME,
            "max_tokens": 100,
        }
        async with session.post(
            "/v1/chat/completions", json=payload
        ) as resp:
            result = await resp.json()
            content = result["choices"][0]["message"]["content"]
            print(f"Response: {content}")
