from PIL import Image
from pathlib import Path

def resize_images(source_dir, target_dir, target_width):
    source = Path(source_dir)
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    for image_path in source.glob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue  # skip non-images

        with Image.open(image_path) as img:
            # Bereken nieuwe hoogte met behoud van aspect ratio
            width_percent = target_width / float(img.width)
            new_height = int(float(img.height) * width_percent)

            resized = img.resize((target_width, new_height), Image.LANCZOS)

            output_path = target / image_path.name
            resized.save(output_path)
            print(f"✅ {image_path.name} -> {output_path}")
resize_images("assets/images/uncompressed", "assets/images/1200", 1200)
resize_images("assets/images/uncompressed", "assets/images/400", 400)