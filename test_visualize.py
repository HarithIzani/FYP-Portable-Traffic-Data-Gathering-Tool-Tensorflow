"""
Quick visual test — sends an image to the local API and saves the result
with bounding boxes drawn on it.

Usage:
    python test_visualize.py <path-to-image>
    python test_visualize.py                  # prompts for a path
"""

import sys
import requests
from PIL import Image, ImageDraw, ImageFont

API_URL = "http://localhost:8000/detect"

# Colour per class (R, G, B)
CLASS_COLORS = {
    "car":           (0,   200, 80),
    "bus":           (0,   120, 255),
    "truck":         (255, 160, 0),
    "motorcycle":    (200, 0,   200),
    "bicycle":       (0,   220, 220),
    "pedestrian":    (255, 60,  60),
    "rider":         (255, 140, 180),
    "train":         (80,  80,  255),
    "trailer":       (180, 120, 0),
    "other_vehicle": (140, 200, 100),
    "other_person":  (200, 200, 0),
}
DEFAULT_COLOR = (255, 255, 255)

def draw_results(image_path: str, result: dict) -> str:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype("arial.ttf", max(14, h // 40))
    except OSError:
        font = ImageFont.load_default()

    for det in result["detections"]:
        cls   = det["class"]
        score = det["score"]
        box   = det["box"]

        x1 = int(box["xmin"] * w)
        y1 = int(box["ymin"] * h)
        x2 = int(box["xmax"] * w)
        y2 = int(box["ymax"] * h)

        color = CLASS_COLORS.get(cls, DEFAULT_COLOR)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = f"{cls} {int(score * 100)}%"
        bbox  = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=color)
        draw.text((x1, y1), label, fill=(0, 0, 0), font=font)

    # Overlay: model used + luminance + total detections
    info = (
        f"Model: {result['model_used']}  |  "
        f"Lumen: {result['luminance']}  |  "
        f"Detections: {len(result['detections'])}"
    )
    draw.rectangle([0, 0, w, 30], fill=(0, 0, 0, 180))
    draw.text((8, 6), info, fill=(0, 255, 80), font=font)

    out_path = image_path.rsplit(".", 1)[0] + "_detected.jpg"
    img.save(out_path, quality=95)
    return out_path


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ").strip().strip('"')

    print(f"Sending {image_path} to {API_URL} ...")
    with open(image_path, "rb") as f:
        resp = requests.post(API_URL, files={"file": (image_path, f, "image/jpeg")})

    if resp.status_code != 200:
        print(f"API error {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()
    print(f"\nModel used : {result['model_used']}")
    print(f"Luminance  : {result['luminance']}")
    print(f"Detections : {len(result['detections'])}")
    print("\nCounts:")
    for cls, count in result["counts"].items():
        if count > 0:
            print(f"  {cls:<16} {count}")

    out = draw_results(image_path, result)
    print(f"\nSaved: {out}")
    Image.open(out).show()


if __name__ == "__main__":
    main()
