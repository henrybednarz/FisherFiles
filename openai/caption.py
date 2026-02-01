from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
import asyncio

from dotenv import load_dotenv

load_dotenv()

from openai import AsyncOpenAI


def _file_to_data_url(image_path: Path) -> str:
    """Convert an image file to a base64-encoded data URL: data:<mime>;base64,<...>."""
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None or not mime_type.startswith("image/"):
        mime_type = "image/png"  # fallback

    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


async def caption(
    image_path: str | Path,
    prompt: str = "Write a concise, vivid caption for this image (1 sentence).",
    model: str = "gpt-4.1-mini",
    max_output_tokens: int = 80,
    api_key: str | None = None,
) -> str:
    """
    Async: Generate a caption for a local image using the OpenAI Responses API.

    Args:
        image_path: Path to a local image file (jpg/png/webp/etc).
        prompt: Instruction for how to caption the image.
        model: Vision-capable model name.
        max_output_tokens: Upper bound for generated tokens.
        api_key: Optional API key. If omitted, uses OPENAI_API_KEY env var.

    Returns:
        The caption text (str).
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (and api_key was not provided).")

    client = AsyncOpenAI(api_key=key)

    path = Path(image_path).expanduser().resolve()
    data_url = _file_to_data_url(path)

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        max_output_tokens=max_output_tokens,
    )

    return (resp.output_text or "").strip()


# Example:
# print(caption("cat.jpg"))
# print(caption("cat.jpg", prompt="Make it a funny caption."))

async def main():
    print(await caption("soccer.webp"))

if __name__ == "__main__":
    asyncio.run(main())