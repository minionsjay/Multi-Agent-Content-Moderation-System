#!/usr/bin/env python3
"""Generate and save all test images used in the dHash benchmark.

Images are photorealistic SYNTHETIC images — NOT real photos.
They simulate visual characteristics of real photos: gradients,
textures, noise, objects, text overlays.

Output: test_images/
  originals/    — 8 base "photos"
  variants/     — manipulated versions of landscape_A
"""

import sys, os, io, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

OUT = os.path.join(os.path.dirname(__file__), "test_images")
os.makedirs(f"{OUT}/originals", exist_ok=True)
os.makedirs(f"{OUT}/variants", exist_ok=True)


# ---- Same generators as bench_real.py ----

def make_landscape(seed, size=(400, 300)):
    random.seed(seed)
    img = Image.new("RGB", size)
    sky_top = (random.randint(30, 80), random.randint(100, 180), random.randint(180, 255))
    sky_bot = (random.randint(80, 150), random.randint(150, 210), random.randint(200, 255))
    horizon = int(size[1] * random.uniform(0.35, 0.65))
    for y in range(horizon):
        ratio = y / horizon
        r = int(sky_top[0] + (sky_bot[0] - sky_top[0]) * ratio)
        g = int(sky_top[1] + (sky_bot[1] - sky_top[1]) * ratio)
        b = int(sky_top[2] + (sky_bot[2] - sky_top[2]) * ratio)
        for x in range(size[0]):
            img.putpixel((x, y), (r, g, b))
    gnd_top = (random.randint(30, 100), random.randint(80, 150), random.randint(20, 80))
    gnd_bot = (random.randint(10, 40), random.randint(40, 80), random.randint(5, 30))
    for y in range(horizon, size[1]):
        ratio = (y - horizon) / (size[1] - horizon)
        r = int(gnd_top[0] + (gnd_bot[0] - gnd_top[0]) * ratio)
        g = int(gnd_top[1] + (gnd_bot[1] - gnd_top[1]) * ratio)
        b = int(gnd_top[2] + (gnd_bot[2] - gnd_top[2]) * ratio)
        for x in range(size[0]):
            img.putpixel((x, y), (r, g, b))
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(8, 20)):
        x = random.randint(10, size[0] - 30)
        y = random.randint(horizon - 10, size[1] - 30)
        w = random.randint(15, 60)
        h = random.randint(w // 2, w * 2)
        color = (random.randint(20, 150), random.randint(40, 120), random.randint(10, 60))
        shape = random.choice(["rect", "ellipse", "triangle"])
        if shape == "rect":
            draw.rectangle([x, y, x + w, y + h], fill=color)
        elif shape == "ellipse":
            draw.ellipse([x, y, x + w, y + h], fill=color)
        else:
            draw.polygon([x, y + h, x + w // 2, y, x + w, y + h], fill=color)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            r, g, b = pixels[x, y]
            noise = random.randint(-8, 8)
            pixels[x, y] = (max(0, min(255, r + noise)), max(0, min(255, g + noise)), max(0, min(255, b + noise)))
    return img


def make_indoor(seed, size=(400, 300)):
    random.seed(seed)
    img = Image.new("RGB", size)
    wr, wg, wb = random.randint(200, 255), random.randint(180, 240), random.randint(150, 220)
    for y in range(size[1]):
        for x in range(size[0]):
            v = random.randint(-5, 5)
            img.putpixel((x, y), (max(0, min(255, wr + v)), max(0, min(255, wg + v)), max(0, min(255, wb + v))))
    draw = ImageDraw.Draw(img)
    floor_y = int(size[1] * random.uniform(0.6, 0.85))
    draw.rectangle([0, floor_y, size[0], size[1]], fill=(random.randint(60, 150), random.randint(40, 100), random.randint(20, 60)))
    for _ in range(random.randint(3, 7)):
        x = random.randint(20, size[0] - 100)
        y = random.randint(20, floor_y - 50)
        w = random.randint(40, 150)
        h = random.randint(30, 100)
        c = (random.randint(50, 180), random.randint(40, 150), random.randint(30, 120))
        draw.rectangle([x, y, x + w, y + h], fill=c)
        draw.rectangle([x, y, x + w, y + 3], fill=(min(255, c[0] + 40), min(255, c[1] + 40), min(255, c[2] + 40)))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            r, g, b = pixels[x, y]
            n = random.randint(-5, 5)
            pixels[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
    return img


def make_text_document(seed, size=(400, 300)):
    random.seed(seed)
    img = Image.new("RGB", size, color=(random.randint(240, 255), random.randint(240, 255), random.randint(240, 255)))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size[0], random.randint(30, 50)], fill=(random.randint(30, 80), random.randint(80, 150), random.randint(180, 240)))
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    y = random.randint(50, 70)
    for _ in range(random.randint(5, 12)):
        text = "".join(random.choice(chars) for _ in range(random.randint(20, 60)))
        draw.text((random.randint(8, 40), y), text, fill=(random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)))
        y += random.randint(15, 25)
    if random.random() > 0.3:
        px, py = random.randint(20, 80), y + random.randint(5, 20)
        pw, ph = random.randint(200, 300), random.randint(100, 180)
        c = random.randint(100, 200)
        draw.rectangle([px, py, px + pw, py + ph], fill=(c, c, c), outline=(180, 180, 180))
    return img


# ---- Generate and save ----

print("Generating original images...")
originals = {
    "landscape_A": make_landscape(100),
    "landscape_B": make_landscape(200),
    "landscape_sunset": make_landscape(700),
    "indoor_A": make_indoor(300),
    "indoor_B": make_indoor(400),
    "indoor_dark": make_indoor(800),
    "document_A": make_text_document(500),
    "document_B": make_text_document(600),
}

for name, img in originals.items():
    path = f"{OUT}/originals/{name}.png"
    img.save(path)
    print(f"  {path}")

# Variants of landscape_A
print("\nGenerating variants of landscape_A...")
ref = originals["landscape_A"]

def save_variant(img, name):
    path = f"{OUT}/variants/{name}.png"
    if hasattr(img, 'save'):
        img.save(path)
    else:
        # bytes → write directly
        with open(path, "wb") as f:
            f.write(img)
    print(f"  {path}")

save_variant(ref, "00_original")

# JPEG at different qualities
for q in [95, 70, 40]:
    buf = io.BytesIO()
    ref.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    with open(f"{OUT}/variants/01_jpeg_q{q}.jpg", "wb") as f:
        f.write(buf.read())
    print(f"  {OUT}/variants/01_jpeg_q{q}.jpg")

# Resize
for pct in [75, 50]:
    w, h = ref.size
    img = ref.resize((int(w * pct / 100), int(h * pct / 100)), Image.LANCZOS).resize(ref.size, Image.LANCZOS)
    save_variant(img, f"02_resize_{pct}pct")

# Watermark
wm = ref.copy()
draw = ImageDraw.Draw(wm)
draw.text((10, 10), "SAMPLE TEXT", fill=(255, 255, 255, 200))
draw.text((150, 150), "© WATERMARK", fill=(255, 255, 255, 120))
draw.text((10, 280), "APPROVED-001", fill=(255, 255, 255, 180))
save_variant(wm, "03_watermarked")

# Brightness
for label, factor in [("bright", 1.2), ("dark", 0.8)]:
    save_variant(ImageEnhance.Brightness(ref).enhance(factor), f"04_{label}")

# Contrast
save_variant(ImageEnhance.Contrast(ref).enhance(1.3), "05_contrast_high")

# Blur
save_variant(ref.filter(ImageFilter.GaussianBlur(radius=1)), "06_blur_1px")

# Crop (center crop then resize back)
cw, ch = int(ref.width * 0.8), int(ref.height * 0.8)
cx, cy = (ref.width - cw) // 2, (ref.height - ch) // 2
cropped = ref.crop((cx, cy, cx + cw, cy + ch)).resize(ref.size, Image.LANCZOS)
save_variant(cropped, "07_center_crop")

print(f"\nDone! Images saved to: {OUT}/")
print(f"  originals/ — {len(originals)} base images")
print(f"  variants/ — 13 manipulated versions")
