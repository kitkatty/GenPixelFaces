import torch, cv2, numpy as np
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw
from pathlib import Path
import random

# ==============================
# CONFIG
# ==============================
OUT_W, OUT_H = 320, 240
GEN_W, GEN_H = 512, 512           # generate larger, crop down
GUIDANCE = 7.5
STEPS = 30
SEED = None                        # set e.g. 1234 for reproducibility
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

POS_PROMPT = (
    "centered studio headshot, full face visible, looking at camera, "
    "symmetrical face, portrait photography, sharp focus, high detail, soft lighting"
)
NEG_PROMPT = (
    "cropped face, out of frame, cut off, multiple faces, lowres, blurry, distorted, "
    "deformed, extra limbs, bad anatomy, watermark, text"
)

# ==============================
# STEP 1: Generate Face (bigger)
# ==============================
def generate_face(prompt=POS_PROMPT, negative_prompt=NEG_PROMPT):
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()

    # seed
    g = None
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        g = torch.Generator(device=DEVICE).manual_seed(SEED)

    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        height=GEN_H, width=GEN_W,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        generator=g
    )
    return result.images[0]

# ==============================
# STEP 2: Face-aware crop to 320x240
# ==============================
def face_center_crop(pil_img, out_w=OUT_W, out_h=OUT_H, margin=0.25):
    """
    Detects the largest face and returns a 320x240 crop centered on it.
    margin: extra border around the detected face (fraction of bbox size).
    Fallback: safe center crop with 4:3 aspect.
    """
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # OpenCV's built-in Haar cascade path
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))

    H, W, _ = img.shape
    target_aspect = out_w / out_h

    if len(faces) > 0:
        # largest face
        x, y, w, h = max(faces, key=lambda b: b[2]*b[3])

        # expand box by margin
        cx, cy = x + w/2, y + h/2
        w *= (1 + margin)
        h *= (1 + margin)

        # now choose crop size with 4:3 aspect, centered on face
        # set crop height ~ 2.2 * face height (nice framing), then enforce aspect
        crop_h = min(H, max(out_h, int(2.2 * h)))
        crop_w = int(crop_h * target_aspect)
        if crop_w > W:
            crop_w = W
            crop_h = int(crop_w / target_aspect)

        # center on face (cx, cy)
        x1 = int(cx - crop_w/2); y1 = int(cy - crop_h/2)
        x1 = max(0, min(W - crop_w, x1))
        y1 = max(0, min(H - crop_h, y1))
        x2, y2 = x1 + crop_w, y1 + crop_h
    else:
        # fallback: center crop to 4:3
        if W / H > target_aspect:
            crop_h = H
            crop_w = int(H * target_aspect)
        else:
            crop_w = W
            crop_h = int(W / target_aspect)
        x1 = (W - crop_w) // 2
        y1 = (H - crop_h) // 2
        x2, y2 = x1 + crop_w, y1 + crop_h

    crop = Image.fromarray(img[y1:y2, x1:x2])
    return crop.resize((out_w, out_h), Image.LANCZOS)

# ==============================
# STEP 3: Mosaic Effect (circles)
# ==============================
def apply_circle_mosaic(img, cell_size=6):
    img = img.convert("RGB")
    np_img = np.array(img)
    h, w, _ = np_img.shape

    mosaic = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(mosaic)

    r = cell_size // 2
    for y in range(0, h, cell_size):
        for x in range(0, w, cell_size):
            cell = np_img[y:y+cell_size, x:x+cell_size]
            avg = tuple(np.mean(cell.reshape(-1, 3), axis=0).astype(int))
            draw.ellipse([x, y, x+cell_size, y+cell_size], fill=avg)
    return mosaic

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    out_path = Path("realistic_face_mosaic.png")

    # 1) Generate larger portrait that encourages full face
    base = generate_face()

    # 2) Auto-center on face and crop to 320x240
    recentered = face_center_crop(base, OUT_W, OUT_H, margin=0.30)

    # 3) Apply mosaic effect
    mosaic_face = apply_circle_mosaic(recentered, cell_size=6)

    mosaic_face.save(out_path)
    print(f"✅ Face centered and saved at {OUT_W}x{OUT_H}: {out_path.resolve()}")
