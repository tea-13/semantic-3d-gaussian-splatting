from PIL import Image
import os

image_folder = './output_seg'
output_gif = 'output.gif'
duration = 300  # длительность каждого кадра в мс (500 мс = 2 FPS)

# Сбор изображений
images = [
    Image.open(os.path.join(image_folder, f))
    for f in sorted(os.listdir(image_folder))
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Сохранение в gif
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=duration,
    loop=0
)

print(f"GIF сохранён как {output_gif}")
