#!/usr/bin/env python3
"""Benchmark: dHash on Photorealistic Synthetic Images

Unlike bench.py (geometric patterns), this generates images that simulate
real-world photo characteristics:
  - Natural gradients (sky, skin tones)
  - Gaussian noise (camera sensor simulation)
  - Multiple visual elements (simulating objects in a scene)
  - JPEG compression with real quantization artifacts
  - Text/watermark overlays

Why not real photos:
  - No actual photos in the project (ethical + licensing)
  - These synthetic images exercise the same visual features dHash depends on:
    edges, gradients, texture distribution, compression artifacts

Tests:
  1. Same "photo", different JPEG quality
  2. Same "photo", resized
  3. Same "photo", watermarked
  4. Same "photo", brightness/contrast adjusted
  5. Different "photos" with similar color palettes
  6. Different "photos" with different content

Usage:
  python bench_real.py
"""

import sys, os, time, io, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from src.skills.image_phash import image_phash


# ---- Realistic Image Generators ----

def make_landscape(seed: int, size=(400, 300)) -> Image.Image:
    """Generate a landscape-like image with sky, ground, and objects.

    Simulates: outdoor photo with horizon, gradient sky, textured ground,
    and scattered elements (trees, rocks, etc).
    """
    random.seed(seed)
    img = Image.new("RGB", size)

    # Sky gradient (top half)
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

    # Ground gradient (bottom half)
    gnd_top = (random.randint(30, 100), random.randint(80, 150), random.randint(20, 80))
    gnd_bot = (random.randint(10, 40), random.randint(40, 80), random.randint(5, 30))
    for y in range(horizon, size[1]):
        ratio = (y - horizon) / (size[1] - horizon)
        r = int(gnd_top[0] + (gnd_bot[0] - gnd_top[0]) * ratio)
        g = int(gnd_top[1] + (gnd_bot[1] - gnd_top[1]) * ratio)
        b = int(gnd_top[2] + (gnd_bot[2] - gnd_top[2]) * ratio)
        for x in range(size[0]):
            img.putpixel((x, y), (r, g, b))

    # Scattered elements (simulating trees, rocks, buildings)
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(8, 20)):
        x = random.randint(10, size[0] - 30)
        y = random.randint(horizon - 10, size[1] - 30)
        w = random.randint(15, 60)
        h = random.randint(w // 2, w * 2)
        color = (
            random.randint(20, 150),
            random.randint(40, 120),
            random.randint(10, 60),
        )
        shape = random.choice(["rect", "ellipse", "triangle"])
        if shape == "rect":
            draw.rectangle([x, y, x + w, y + h], fill=color)
        elif shape == "ellipse":
            draw.ellipse([x, y, x + w, y + h], fill=color)
        else:
            draw.polygon([x, y + h, x + w // 2, y, x + w, y + h], fill=color)

    # Gaussian noise (camera sensor simulation)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            r, g, b = pixels[x, y]
            noise = random.randint(-8, 8)
            pixels[x, y] = (
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise)),
            )

    return img


def make_indoor(seed: int, size=(400, 300)) -> Image.Image:
    """Generate an indoor-like scene with walls, furniture-like shapes.

    Simulates: room photo with walls, floor, furniture.
    """
    random.seed(seed)
    img = Image.new("RGB", size)

    # Wall color (warm indoor tones)
    wall_r = random.randint(200, 255)
    wall_g = random.randint(180, 240)
    wall_b = random.randint(150, 220)
    for y in range(size[1]):
        for x in range(size[0]):
            variance = random.randint(-5, 5)
            img.putpixel((x, y), (
                max(0, min(255, wall_r + variance)),
                max(0, min(255, wall_g + variance)),
                max(0, min(255, wall_b + variance)),
            ))

    draw = ImageDraw.Draw(img)

    # Floor line
    floor_y = int(size[1] * random.uniform(0.6, 0.85))
    floor_color = (
        random.randint(60, 150),
        random.randint(40, 100),
        random.randint(20, 60),
    )
    draw.rectangle([0, floor_y, size[0], size[1]], fill=floor_color)

    # Furniture-like rectangles
    for _ in range(random.randint(3, 7)):
        x = random.randint(20, size[0] - 100)
        y = random.randint(20, floor_y - 50)
        w = random.randint(40, 150)
        h = random.randint(30, 100)
        color = (
            random.randint(50, 180),
            random.randint(40, 150),
            random.randint(30, 120),
        )
        draw.rectangle([x, y, x + w, y + h], fill=color)
        # Highlight edge
        draw.rectangle([x, y, x + w, y + 3], fill=(
            min(255, color[0] + 40),
            min(255, color[1] + 40),
            min(255, color[2] + 40),
        ))

    # Noise
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            r, g, b = pixels[x, y]
            n = random.randint(-5, 5)
            pixels[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))

    return img


def make_text_document(seed: int, size=(400, 300)) -> Image.Image:
    """Generate a document/screenshot-like image with text blocks.

    Simulates: screenshot of a social media post, chat, or document.
    """
    random.seed(seed)
    img = Image.new("RGB", size, color=(
        random.randint(240, 255),
        random.randint(240, 255),
        random.randint(240, 255),
    ))
    draw = ImageDraw.Draw(img)

    # Header bar
    header_color = (
        random.randint(30, 80),
        random.randint(80, 150),
        random.randint(180, 240),
    )
    draw.rectangle([0, 0, size[0], random.randint(30, 50)], fill=header_color)

    # Text blocks (simulated lines)
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    y = random.randint(50, 70)
    for _ in range(random.randint(5, 12)):
        line_len = random.randint(20, 60)
        text = "".join(random.choice(chars) for _ in range(line_len))
        text_color = (
            random.randint(0, 50),
            random.randint(0, 50),
            random.randint(0, 50),
        )
        draw.text((random.randint(8, 40), y), text, fill=text_color)
        y += random.randint(15, 25)

    # Image placeholder (simulating embedded image in post)
    if random.random() > 0.3:
        px = random.randint(20, 80)
        py = y + random.randint(5, 20)
        pw = random.randint(200, 300)
        ph = random.randint(100, 180)
        img_color = (
            random.randint(100, 200),
            random.randint(100, 200),
            random.randint(100, 200),
        )
        draw.rectangle([px, py, px + pw, py + ph], fill=img_color, outline=(180, 180, 180))
        # Simulate image content with gradient blocks
        for gy in range(py, py + ph, 20):
            shade = random.randint(80, 180)
            draw.rectangle([px, gy, px + pw, min(gy + 18, py + ph)], fill=(shade, shade, shade))

    return img


# ---- Manipulation Functions ----

def to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def jpeg_compress(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return buf.read()


def add_watermark(img: Image.Image, text: str) -> Image.Image:
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    w, h = img2.size
    for i, char in enumerate(text):
        x = 10 + i * 20
        y = 10 + i * 8
        draw.text((x, y), char, fill=(255, 255, 255, 180))
    draw.text((w // 2 - 50, h // 2), text, fill=(255, 255, 255, 100))
    return img2


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


# ---- Main Benchmark ----

def bench_realistic():
    print("=" * 70)
    print("dHash on Photorealistic Synthetic Images")
    print("=" * 70)

    # Generate 8 different "photos"
    generators = [
        ("landscape_A", lambda: make_landscape(100)),
        ("landscape_B", lambda: make_landscape(200)),
        ("indoor_A", lambda: make_indoor(300)),
        ("indoor_B", lambda: make_indoor(400)),
        ("document_A", lambda: make_text_document(500)),
        ("document_B", lambda: make_text_document(600)),
        ("landscape_sunset", lambda: make_landscape(700)),
        ("indoor_dark", lambda: make_indoor(800)),
    ]

    images = {}
    for name, gen in generators:
        img = gen()
        images[name] = img

    print(f"\nGenerated {len(images)} 'photos':")
    for name, img in images.items():
        h = image_phash.dhash(to_bytes(img))
        print(f"  {name:20s}: {h}")

    # ---- Test 1: Same photo, different manipulations ----
    print(f"\n{'='*70}")
    print("Test 1: Same Photo vs Manipulations (should MATCH)")
    print(f"{'='*70}")

    ref = images["landscape_A"]
    ref_hash = image_phash.dhash(to_bytes(ref))
    print(f"Reference: landscape_A = {ref_hash}")

    manipulations = [
        ("JPEG q=95", jpeg_compress(ref, 95)),
        ("JPEG q=70", jpeg_compress(ref, 70)),
        ("JPEG q=40", jpeg_compress(ref, 40)),
        ("resize 75%", to_bytes(ref.resize((300, 225), Image.LANCZOS).resize(ref.size, Image.LANCZOS))),
        ("resize 50%", to_bytes(ref.resize((200, 150), Image.LANCZOS).resize(ref.size, Image.LANCZOS))),
        ("watermark", to_bytes(add_watermark(ref, "SAMPLE"))),
        ("bright +20%", to_bytes(adjust_brightness(ref, 1.2))),
        ("bright -20%", to_bytes(adjust_brightness(ref, 0.8))),
        ("contrast +30%", to_bytes(adjust_contrast(ref, 1.3))),
        ("blur 1px", to_bytes(ref.filter(ImageFilter.GaussianBlur(radius=1)))),
    ]

    all_ok = True
    for label, img_bytes in manipulations:
        h = image_phash.dhash(img_bytes)
        d = image_phash.hamming(ref_hash, h)
        status = "✓" if d <= 10 else "✗ FAIL"
        if d > 10:
            all_ok = False
        bar = "█" * min(d, 40) + "░" * max(0, 40 - min(d, 40))
        print(f"  [{status}] {label:18s}: d={d:>3}/64  {bar}")

    # ---- Test 2: Different photos ----
    print(f"\n{'='*70}")
    print("Test 2: Different Photos vs Each Other (should NOT match)")
    print(f"{'='*70}")

    photo_hashes = {name: image_phash.dhash(to_bytes(img)) for name, img in images.items()}

    different_ok = True
    for name1 in sorted(images.keys()):
        for name2 in sorted(images.keys()):
            if name1 >= name2:
                continue
            d = image_phash.hamming(photo_hashes[name1], photo_hashes[name2])
            status = "✓" if d > 10 else "✗ FALSE MATCH!"
            if d <= 10:
                different_ok = False
            print(f"  [{status}] {name1:20s} ↔ {name2:20s}: d={d:>3}/64")

    # ---- Test 3: Similar but different scenes ----
    print(f"\n{'='*70}")
    print("Test 3: Similar Scenes (two landscapes, two indoors)")
    print(f"{'='*70}")

    similar_pairs = [
        ("landscape_A", "landscape_B", "two different landscapes"),
        ("landscape_A", "landscape_sunset", "landscape vs sunset variant"),
        ("indoor_A", "indoor_B", "two different indoor scenes"),
        ("document_A", "document_B", "two different documents"),
    ]

    for a, b, desc in similar_pairs:
        d = image_phash.hamming(photo_hashes[a], photo_hashes[b])
        status = "✓" if d > 10 else "~ similar"
        print(f"  [{status}] {desc:30s}: d={d:>3}/64")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Same-photo manipulations: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print(f"  Different-photo distinction: {'ALL PASSED' if different_ok else 'FALSE MATCHES FOUND'}")
    print(f"  Recommended threshold: Hamming distance ≤ 10")


if __name__ == "__main__":
    bench_realistic()
    print("\n✅ Realistic image dHash benchmark complete")
