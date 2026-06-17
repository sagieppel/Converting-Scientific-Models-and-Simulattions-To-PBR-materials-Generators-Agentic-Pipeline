import time

from sympy.polys.galoistools import gf_exquo

import API_KEYS

from openai import OpenAI



import cv2
import re



import os

import ast
from json.decoder import JSONDecodeError
import json

################################################################################################################3
def get_mime_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')  # default to jpe
###################################################################################################################################
def normalize_to_json(raw: str) -> str:
    """
    Take a string that might be either:
      • valid JSON, or
      • a Python literal (single‑quoted dict, tuple, list, etc.)
    and return a *valid* JSON string.
    """
    try:
        # If it already parses as JSON, just re-dump to normalize formatting
        obj = json.loads(raw)
    except JSONDecodeError:
        # Fallback: safely evaluate it as a Python literal
        obj = ast.literal_eval(raw)
    # Now dump as JSON with double‑quotes, no trailing commas, etc.
    return json.dumps(obj)
##########################################################################################
def get_reponse(data=None,text=None,messages=[], model="", as_json=False) -> list[str]:
    if text is not None: print("\n\n\nQuery:\n"+text+"\n")
    result=get_response_image_txt_json(text=text,model=model,as_json=as_json,messages=messages)
    if data==None:
        return  result
    data['messages'].append({"role": "user", "content": text})
    data['messages'].append({"role": "system", "content": str(result)})
    return result, data
##############################################################################################
def get_response_image_txt_json(
    text: str=None,
    img_path: list[str] | None = None,         # or pass file paths
    model: str = "gpt-5-mini",
    as_json: bool = True,
    messages: list=[]
) -> str:
    # Send querty
    openai_models=["gpt-5-mini","gpt-5", "gpt-oss-120b", "gpt-oss-20b","o4-mini","gpt-5.2-2025-12-11"]
    together_models=["zai-org/GLM-5","google/gemma-3n-E4B-it","meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8","meta-llama/Llama-4-Scout-17B-16E-Instruct","Qwen/Qwen2.5-VL-72B-Instruct","deepseek-ai/DeepSeek-R1-0528"]
    gemini_models=["gemini-3-pro-preview","gemini-2.5-pro","gemini-2.5-flash"]
    grok_models = ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning","grok-4","grok-2-vision","grok-2-vision-1212"]
    claude_models = ["claude-3-5-sonnet-latest","claude-sonnet-4-5-20250929","claude-sonnet-4-5-latest"]
    open_router=["minimax/minimax-m2.5","google/gemini-3.1-flash-lite-preview","google/gemini-3.1-pro-preview","qwen/qwen3.5-plus-02-15","z-ai/glm-5","qwen/qwen3.5-plus-02-15","minimax/minimax-m2","deepseek/deepseek-v3.2","z-ai/glm-4.7","google/gemini-3-flash-preview", "z-ai/glm-4.7-flash","moonshotai/kimi-k2.5","openai/gpt-5.2","qwen/qwen3-vl-32b-instruct","openai/gpt-5","z-ai/glm-4.6v","qwen/qwen3-vl-32b-instruct","qwen/qwen3-vl-8b-instruct"]
    l = len(messages)
    for i in range(4):
        messages = messages[:l]
        try:
                if model in openai_models:
                    return get_response_image_txt_json_openai(text,img_path,model,as_json,messages)
                elif model in together_models:
                    return get_response_image_txt_json_together(text, img_path, model, as_json,messages)
                elif model in gemini_models:
                    return get_response_image_txt_json_gemini(text, img_path, model, as_json,messages)
                elif model in grok_models:
                    return get_response_image_txt_json_grok(text, img_path, model, as_json,messages)
                elif model in claude_models:
                    return get_response_image_txt_json_claude(text, img_path, model, as_json,messages)
                elif model == "human":
                    return get_response_image_txt_json_human(text, img_path)
                else:
                    return get_response_image_txt_json_openrouter(text, img_path, model, as_json, messages)

        except Exception as e:
             print("Error getting answer:\n",e,"\nModel:",model)
             print("sleeping 6 seconds")
             time.sleep(6)



###############################################################################################################################
import base64
import json
import mimetypes
from typing import Optional, Union, List, Dict, Any

import requests


def get_mime_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def normalize_to_json(s: str) -> str:
    """
    Best-effort cleanup in case the model returns JSON wrapped in code fences.
    If your previous normalize_to_json already exists, you can remove this and use yours.
    """
    s = s.strip()
    if s.startswith("```"):
        # remove ```json ... ```
        s = s.split("```", 1)[1]
        s = s.rsplit("```", 1)[0]
    return s.strip()

#############################################################################################################
def get_response_image_txt_json_openrouter(
    text: str,
    img_path: Optional[Union[List[str], Dict[str, str]]] = None,  # list of paths OR dict(label->path)
    model: str = "qwen/qwen2.5-vl-72b-instruct",  # OpenRouter model slug (adjust if needed)
    as_json: bool = True,
    messages: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    site_url: Optional[str] = None,      # optional: OpenRouter leaderboard attribution
    app_name: Optional[str] = None,      # optional: OpenRouter leaderboard attribution
    timeout: int = 120,
) -> Any:
    api_key=API_KEYS.openrouter_api_key
    """


    Notes:
    - OpenRouter uses an OpenAI-compatible Chat Completions API.
    - Images are passed via content blocks with {"type":"image_url","image_url":{"url":...}}.
    - If as_json=True, we request JSON mode: response_format={"type":"json_object"}.
      (Model support varies; if the model doesn't support it, you may need to enforce JSON via prompting.)

    Returns:
      - Parsed JSON (dict) if as_json=True
      - Raw string otherwise
    """
    if messages is None:
        messages = []
    if api_key is None:
        raise ValueError("api_key is required (your OpenRouter API key).")

    # 1) Build the user 'content' as a list of blocks
    content: List[Dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})

    # 2) Attach images as base64 data URLs
    # Accept either list[str] or dict[label->path]
    if img_path:
        if isinstance(img_path, dict):
            items = list(img_path.items())  # (label, path)
        else:
            # if list, create simple labels: image_0, image_1, ...
            items = [(f"image_{i}", p) for i, p in enumerate(img_path)]

        for label, pth in items:
            with open(pth, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            mime = get_mime_type(pth)
            content.append({"type": "text", "text": f"This is image {label}:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )

    if content:
        messages.append({"role": "user", "content": content})

    # 3) OpenRouter request (OpenAI-compatible)
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Optional attribution headers (recommended by OpenRouter)
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if as_json:
        payload["response_format"] = {"type": "json_object"}
        payload['messages'].append({"role": "user", "content": "Respond with raw JSON only. Do not use Markdown.Do not wrap the response in code fences. Output must be directly parsable by JSON.parse."})

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    r.raise_for_status()
    data = r.json()

    msg = data["choices"][0]["message"]["content"]

    if as_json:
        return json.loads(normalize_to_json(msg))
    return msg





#########################################################################################################################
def get_response_image_txt_json_openai(
    text: str,
    img_path: list[str] | None = None,         # or pass file paths
    model: str = "gpt-5-mini",
    as_json: bool = True,
    messages: list =[]
):
            client = OpenAI(api_key=API_KEYS.open_AI_api_key)
            """
            Send a text prompt with multiple images (each with a label) to a multimodal model.
            Returns the assistant's message content.
            """

            # 1) Build the user 'content' as a list of content blocks

            if text is not None and len(text) > 0:
                content = [{"type": "text", "text": text}]
            else:
                content = None

            # 2) Prepare images (from paths or a dict of base64 strings)
            #    If both provided, we combine them.
            prepared = {}

            if img_path:
                for label in img_path:
                    pth = img_path[label]
                    with open(pth, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")

                    mime=get_mime_type(pth)
                    content.append({"type": "text", "text": f"This is image {label}:"})
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })
            if content is not None:
                messages.append({
                            "role": "user",
                            "content": content   # <-- list of objects, not nested inside another list
                        })
            # 4) Call the API with a single user message whose 'content' is that list of blocks
            resp = client.chat.completions.create(
                model=model,
                messages=messages
              ###  response_format={"type": "json_object"} if as_json else None
            )

            msg = resp.choices[0].message.content
            if as_json:
                              results = json.loads(normalize_to_json(msg))
            else:
                results=msg
            return results
##########################################################################################
def get_response_image_txt_json_together(
    text: str,
    img_path: list[str] | None = None,         # or pass file paths
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    as_json: bool = True,
    messages: list =[]
):
            from together import Together
            client = Together(api_key=API_KEYS.together_api_key)
            """
            Send a text prompt with multiple images (each with a label) to a multimodal model.
            Returns the assistant's message content.
            """

            # 1) Build the user 'content' as a list of content blocks

            if text is not None and len(text) > 0:
                content = [{"type": "text", "text": text}]
            else:
                content = None


            if img_path:
                for label in img_path:
                    pth = img_path[label]
                    with open(pth, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")

                    mime = get_mime_type(pth)
                    content.append({"type": "text", "text": f"This is image {label}:"})
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })

            # 4) Call the API with a single user message whose 'content' is that list of blocks
            if content is not None:
                messages.append({
                    "role": "user",
                    "content": content  # <-- list of objects, not nested inside another list
                })
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"} if as_json else None
            )

            msg = resp.choices[0].message.content
            if as_json:

                results = json.loads(normalize_to_json(msg))

            else:
                results=msg

            return results
#########################################################################################################################
##########################################################################################
def get_response_image_txt_json_gemini(
    text: str,
    img_path: list[str] | None = None,         # or pass file paths
    model: str = "gemini-2.5-pro",
    as_json: bool = True,
    messages: list =[]
):
            import google.generativeai as genai
            genai.configure(api_key=API_KEYS.gemini_api_key)


            # ---- Helpers ----
            def load_image_part(path: str) -> dict:
                """Return a Gemini image part from a local file."""
                with open(path, "rb") as f:
                    return {"mime_type": "image/jpeg", "data": f.read()}
            """
            Send a text prompt with multiple images (each with a label) to a multimodal model.
            Returns the assistant's message content.
            """

            # 1) Build the user 'content' as a list of content blocks
            content=[]
            if len(messages)>0:
                 for msg in messages:
                     content.append(str(msg))
            if text is not None and len(text) > 0:
                 content.append(text)


            if img_path:
                for label in img_path:
                    pth = img_path[label]
                    content.append("Image "+label)
                    content.append(load_image_part(pth))



            model = genai.GenerativeModel(model,generation_config = {"response_mime_type": "application/json",})


            resp = model.generate_content(content)

            if as_json:
                    results = json.loads(normalize_to_json(resp.text))
            else:
                results = resp.text

            return results
#########################################################################################################################
##########################################################################################
def get_response_image_txt_json_grok(
    text: str,
    img_path: list[str] | None = None,         # or pass file paths
    model: str = "grok-2-vision",
    as_json: bool = True,
    messages: list =[]
) -> str:
            client = OpenAI(api_key=API_KEYS.grok_api_key, base_url="https://api.x.ai/v1")
            """
            Send a text prompt with multiple images (each with a label) to a multimodal model.
            Returns the assistant's message content.
            """

            # 1) Build the user 'content' as a list of content blocks

            if text is not None and len(text) > 0:
                content = [{"type": "text", "text": text}]
            else:
                content = None

            # 2) Prepare images (from paths or a dict of base64 strings)
            #    If both provided, we combine them.
            prepared = {}

            if img_path:
                for label in img_path:
                    pth = img_path[label]
                    with open(pth, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")

                    mime = get_mime_type(pth)
                    content.append({"type": "text", "text": f"This is image {label}:"})
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"}
                    })


            if content is not None:
                messages.append({
                    "role": "user",
                    "content": content  # <-- list of objects, not nested inside another list
                })

            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"} if as_json else None
            )

            msg = resp.choices[0].message.content
            if as_json:
                   results = json.loads(normalize_to_json(msg))
            else:
                results=msg

            return results
###########################################################################################3
def load_json_claude(text):
    try:
        return json.loads(normalize_to_json(text))
    except:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

                # Remove markdown code fences
            text = re.sub(r'^```json\s*\n', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n```\s*$', '', text, flags=re.MULTILINE)

            # Try parsing again
            return json.loads(text.strip())
##########################################################################################
def get_response_image_txt_json_claude(
    text: str,
    img_path: list[str] | None = None,         # or pass file paths
    model: str = "",
    as_json: bool = True,
    messages: list =[]
):
            from anthropic import Anthropic
            client = Anthropic(api_key=API_KEYS.claude_api_key)
            """
            Send a text prompt with multiple images (each with a label) to a multimodal model.
            Returns the assistant's message content.
            """


            # 1) Build the user 'content' as a list of content blocks
            content = []
            if len(messages) > 0:
                    content.append({"type": "text", "text":str(messages)})
            if text is not None and len(text) > 0:
                content.append({"type": "text", "text":text})

            # 2) Prepare images (from paths or a dict of base64 strings)
            #    If both provided, we combine them.
            prepared = {}

            if img_path:
                for label in img_path:
                    pth = img_path[label]
                    with open(pth, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("utf-8")
                    # guess mime by extensio
                    mime = get_mime_type(pth)#"image/png" if ext == ".png" else "image/jpeg"
                    prepared[label] = (mime, b64)

            # 3) Append a short label *before* each image, then the image block itself
            for label, (mime, b64) in prepared.items():
                content.append({"type": "text", "text": f"This is image {label}:"})

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime,
                        "data": b64,
                    },
                })


            resp = client.messages.create(
                model=model,
                max_tokens=20000,
                messages=[{"role": "user", "content": content}]
            )

            msg = resp.content[0].text


            if as_json:
                   results = load_json_claude(msg)
            else:
                results=msg

            return results

#############################################################################################################
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional, Union


def place_image_with_label(
        img_np: np.ndarray,
        label: str,
        H: int,
        W: int,
        Htxt: int,
        *,
        pad_color: Union[int, Tuple[int, int, int]] = 255,
        text_color: Union[int, Tuple[int, int, int]] = 0,
        font_path: Optional[str] = None,
        max_font_size: int = 48,
) -> np.ndarray:
    """
    Resize an image to height H, then pad or center-crop its width to W.
    Add a white text strip of height Htxt underneath with the given label.

    Args:
        img_np: Input image as a NumPy array (H x W [x C], uint8). Supports
                grayscale, RGB, or RGBA.
        label: Text to render below the image.
        H: Target image height (not counting the text strip).
        W: Target total width (image will be padded or cropped to this).
        Htxt: Height of the text strip to add *below* the image.
        pad_color: Background color for padding and text strip.
                   If grayscale image, use an int (0–255). If RGB, use (R,G,B).
        text_color: Text color (int or (R,G,B)).
        font_path: Optional path to a TrueType font. If None, uses PIL default.
        max_font_size: Upper bound used for auto-fitting the label text.

    Returns:
        NumPy uint8 array of shape (H + Htxt, W, 3) in RGB.
    """
    # ---- Normalize input to a PIL RGB image ----
    if img_np.dtype != np.uint8:
        raise ValueError("img_np must be dtype uint8")
    if img_np.ndim == 2:
        pil = Image.fromarray(img_np, mode="L").convert("RGB")
    elif img_np.ndim == 3 and img_np.shape[2] == 3:
        pil = Image.fromarray(img_np, mode="RGB")
    elif img_np.ndim == 3 and img_np.shape[2] == 4:
        # Composite RGBA over white, then to RGB
        pil_rgba = Image.fromarray(img_np, mode="RGBA")
        bg = Image.new("RGBA", pil_rgba.size, (255, 255, 255, 255))
        pil = Image.alpha_composite(bg, pil_rgba).convert("RGB")
    else:
        raise ValueError("img_np must be HxW, HxWx3, or HxWx4 uint8")

    # ---- Resize to target height H, preserving aspect ----
    w0, h0 = pil.size
    if h0 == 0:
        raise ValueError("Invalid image height 0")
    new_w = max(1, int(round(w0 * (H / h0))))
    pil_resized = pil.resize((new_w, H), resample=Image.BICUBIC)

    # ---- Pad or center-crop width to W ----
    # Normalize pad color to RGB tuple
    def _to_rgb(color):
        if isinstance(color, int):
            return (color, color, color)
        return tuple(color)

    pad_rgb = _to_rgb(pad_color)

    if new_w == W:
        pil_fitted = pil_resized
    elif new_w < W:
        # Center pad
        canvas = Image.new("RGB", (W, H), pad_rgb)
        x = (W - new_w) // 2
        canvas.paste(pil_resized, (x, 0))
        pil_fitted = canvas
    else:
        # Center crop
        left = (new_w - W) // 2
        pil_fitted = pil_resized.crop((left, 0, left + W, H))

    # ---- Create text strip and render label (auto-fit) ----
    strip = Image.new("RGB", (W, Htxt), pad_rgb)
    draw = ImageDraw.Draw(strip)

    # Load font
    if font_path is not None:
        try:
            font = ImageFont.truetype(font_path, size=max_font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        # Try a common default TrueType, else fallback
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max_font_size)
        except Exception:
            font = ImageFont.load_default()

    # Reduce font size until the text fits within W - margin
    margin = max(8, W // 50)
    max_width = W - 2 * margin

    # PIL <=10 compatibility: use textbbox for precise box; fallback to textlength/height
    def text_bbox(fnt):
        try:
            return draw.textbbox((0, 0), label, font=fnt)
        except Exception:
            w = draw.textlength(label, font=fnt)
            h = fnt.size if hasattr(fnt, "size") else 12
            return (0, 0, int(w), int(h))

    if hasattr(font, "path") or isinstance(font, ImageFont.FreeTypeFont):
        size = max_font_size
        # If current font is bitmap (load_default), it has fixed size; skip autosize for it.
        while size > 6:
            if font_path or isinstance(font, ImageFont.FreeTypeFont):
                # Recreate font at this size (for TTFs)
                try:
                    # Try to keep same face if known; otherwise default DejaVu
                    path = getattr(font, "path", None) or font_path or "DejaVuSans.ttf"
                    font = ImageFont.truetype(path, size=size)
                except Exception:
                    # Fallback to default bitmap; break autosize
                    font = ImageFont.load_default()
                    break
            bbox = text_bbox(font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if tw <= max_width and th <= Htxt - 2 * margin:
                break
            size -= 1

    # Final placement (centered)
    bbox = text_bbox(font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = max(margin, (W - tw) // 2)
    ty = max(margin, (Htxt - th) // 2)
    draw.text((tx, ty), label, fill=_to_rgb(text_color), font=font)

    # ---- Stack image + text vertically and return as NumPy ----
    out = Image.new("RGB", (W, H + Htxt), (255, 255, 255))
    out.paste(pil_fitted, (0, 0))
    out.paste(strip, (0, H))
    return np.array(out)


##########################################################################################################################################################


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


#################unify all images in the dictionary############################################################################################
def unify_image(data,num_columns=5, labels=[],sz=300,y_gap=60,x_gap=10,out_file="",disp=False):
    all_raws=[]
    raw_sep = np.ones([y_gap,(sz+x_gap)*num_columns,3],np.uint8)*255
    col_sep = np.ones([sz + int(sz/10),x_gap,3],np.uint8)*255
    for lb in labels: # All images belonging to the same label will be at the same line
        raw=[]
        for ky in sorted(data.keys(), key=natural_key):
            if lb in ky:
                img = cv2.imread(data[ky])
                img = place_image_with_label(img,label=ky,H = sz,W = sz,Htxt=int(sz/10))
                if len(raw)>=num_columns*2:
                    all_raws.append(np.hstack(raw))
                    all_raws.append(raw_sep)
                    raw = []
                raw.append(img)
                raw.append(col_sep)

        while len(raw)<num_columns*2 and len(raw)>0:
              raw.append(img*0+255)
              raw.append(col_sep)
        if len(raw)>0:
            all_raws.append(np.hstack(raw))
            all_raws.append(raw_sep)
 #   if len(all_raws)>1:
        full_im = np.vstack(all_raws)
  #  else:
  #      full_im=all_raws[0]
    if disp:
        cv2.imshow("",full_im)
        cv2.waitKey()

    if len(out_file)>0:
        cv2.imwrite(out_file,full_im)
        new_data={"Images":out_file}
        return full_im,new_data
    return full_im




##############################################################################################################################
#########################################################################################################################
def get_response_image_txt_json_human(
    text: str,
    img_path: list[str] | None = None,         # or pass file paths
    model: str = "gpt-5-mini",
    as_json: bool = True
) -> str:

            print(text)

            if img_path:
                for label in img_path:
                    pth = img_path[label]
                    im=cv2.imread(pth)
                    im=cv2.resize(im,[int(0.7*im.shape[1]),int(0.7*im.shape[0])])
                    cv2.imshow("Which simulate image best match the real press 1-9 if its 10 press 0 (this tests can handle more then 10)",im)
            ky=cv2.waitKey()
            ans="SIM" + str(chr(ky))
            if ans=="SIM0": ans="SIM10"
            return {"answer":ans,"explain":"humans dont explain"}










##########################################################################################################################
# if __name__ == "__main__":
#     unite_statitics("/media/deadcrow/6TB/python_project/Endles_PLAYGROUND/im2txt_im2model_code_results_single_im_100q/")
#    # unite_statitics("/media/deadcrow/6TB/python_project/Endles_PLAYGROUND/im2txt_description_results_single_im_100q//")
#     # unite_statitics("/media/deadcrow/6TB/python_project/Endles_PLAYGROUND/im2txt_code_clean_results_single_im")
#     #
#     # unite_statitics("/media/deadcrow/6TB/python_project/Endles_PLAYGROUND/im2im_results_multi_im_3ref_10choiceas")
#     # unite_statitics("/media/deadcrow/6TB/python_project/Endles_PLAYGROUND/im2im_results_single_im_3ref_10choiceas")