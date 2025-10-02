import os, subprocess, sys, shutil, json, pathlib
from pathlib import Path
from PIL import Image
import numpy as np

# ---- config ----
OUT_W, OUT_H = 320, 240
PROMPT_TEXT = "Hey William! I'm ready to ship. Which feature should we add next?"
EMOTION = "happy"   # neutral | happy | sad | angry | surprised
IMAGE_PATH = "input.jpg"  # put your portrait here (prefer 512x512+)

# ---- tiny helper: face-aware crop (same idea as before) ----
def face_center_crop(pil_img, out_w=OUT_W, out_h=OUT_H):
    import cv2
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    H,W,_ = img.shape
    asp = out_w/out_h
    if len(faces):
        x,y,w,h = max(faces, key=lambda b:b[2]*b[3])
        cx,cy = x+w/2, y+h/2
        crop_h = min(H, int(h*2.1))
        crop_w = min(W, int(crop_h*asp))
        x1 = int(max(0, min(W-crop_w, cx - crop_w/2)))
        y1 = int(max(0, min(H-crop_h, cy - crop_h/2)))
        crop = img[y1:y1+crop_h, x1:x1+crop_w]
    else:
        # center 4:3
        if W/H > asp:
            ch = H; cw = int(H*asp); x1=(W-cw)//2; y1=0
        else:
            cw = W; ch = int(W/asp); x1=0; y1=(H-ch)//2
        crop = img[y1:y1+ch, x1:x1+cw]
    return Image.fromarray(crop).resize((out_w,out_h), Image.LANCZOS)

# ---- emotion presets for SadTalker ----
PRESETS = {
    "neutral":   {"expression_scale":"1.2", "blink_every":"3", "pose_scale":"1.0", "extra":["--still"]},
    "happy":     {"expression_scale":"1.4", "blink_every":"2", "pose_scale":"1.1", "extra":[]},
    "sad":       {"expression_scale":"0.9", "blink_every":"4", "pose_scale":"0.8", "extra":["--still"]},
    "angry":     {"expression_scale":"1.6", "blink_every":"3", "pose_scale":"1.3", "extra":[]},
    "surprised": {"expression_scale":"1.5", "blink_every":"2", "pose_scale":"1.4", "extra":[]},
}

def tts(text, outwav="speech.wav"):
    from TTS.api import TTS
    import soundfile as sf
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    audio = tts.tts(text)
    sf.write(outwav, audio, 22050, subtype="PCM_16")
    return outwav

def ensure_portrait():
    img = Image.open(IMAGE_PATH).convert("RGB")
    centered = face_center_crop(img, OUT_W, OUT_H)
    centered.save("portrait_320x240.jpg")
    return "portrait_320x240.jpg"

def run_sadtalker(image, audio, outdir):
    sd = Path("SadTalker")
    assert sd.exists(), "SadTalker folder not found next to this script."
    p = PRESETS.get(EMOTION, PRESETS["neutral"])
    cmd = [
        sys.executable, str(sd/"inference.py"),
        "--driven_audio", audio,
        "--source_image", image,
        "--enhancer", "gfpgan",
        "--preprocess", "crop",
        "--expression_scale", p["expression_scale"],
        "--blink_every", p["blink_every"],
        "--pose_scale", p["pose_scale"],
        "--outdir", outdir
    ] + p["extra"]
    print(">>", " ".join(map(str, cmd)))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    outdir = "out_"+EMOTION
    Path(outdir).mkdir(exist_ok=True)
    wav = tts(PROMPT_TEXT, "speech.wav")
    portrait = ensure_portrait()
    run_sadtalker(portrait, wav, outdir)
    print(f"✅ Done: check {outdir}/ for the MP4")
