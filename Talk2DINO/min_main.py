import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import numpy as np
from torchvision.io import read_image
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

sys.path.insert(0, "src/open_vocabulary_segmentation")
from models.dinotext import DINOText
from models import build_model

device = "cuda"

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

palette = [
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    [0, 0, 255],
    [128, 128, 128]
]

def np_save_data(path, name, feat, segmap):
    # Преобразуем тензоры, убирая первую размерность
    feat = feat.squeeze(0).cpu().numpy()
    segmap = segmap.squeeze(0).cpu().numpy()

    save_path_s = os.path.join(path, name + '_s.npy')
    save_path_f = os.path.join(path, name + '_f.npy')
    print(save_path_s, save_path_f)

    # Сохраняем в .npy
    np.save(save_path_f, feat)
    np.save(save_path_s, segmap)


def plot_qualitative(image, sim, output_path, palette):
    qualitative_plot = np.zeros((sim.shape[0], sim.shape[1], 3)).astype(np.uint8)

    for j in list(np.unique(sim)):
        qualitative_plot[sim == j] = np.array(palette[j])
    plt.axis('off')
    plt.imshow(image)
    plt.imshow(qualitative_plot, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def get_mask(model, args, img):
    with_background = args.with_background
    text = args.textual_categories.replace("_", " ").split(",")

    if len(text) > len(palette):
        for _ in range(len(text) - len(palette)):
            palette.append([np.random.randint(0, 255) for _ in range(3)])

    if with_background:
        palette.insert(0, [0, 0, 0])
        model.with_bg_clean = True


    with torch.no_grad():
        text_emb = model.build_dataset_class_tokens("sub_imagenet_template", text)
        text_emb = model.build_text_embedding(text_emb)
        
        mask, _, save_tensor_feature, save_tensor_segmap = model.generate_masks(img, img_metas=None, text_emb=text_emb, classnames=text, apply_pamr=True)

        if with_background:
            background = torch.ones_like(mask[:, :1]) * 0.55
            mask = torch.cat([background, mask], dim=1)
        
        mask = mask.argmax(dim=1) 

    return mask, save_tensor_feature, save_tensor_segmap
        

class GlobalArgs:
    config = "src/open_vocabulary_segmentation/configs/cityscapes/dinotext_cityscapes_vitb_mlp_infonce.yml"
    with_background = False
    textual_categories = "bear,knife,bottle,robot,keys,headphones"
    input_folder = "./images"
    output_folder = "./output"
    output_folder_seg = "./output_seg"
    cfg = OmegaConf.load(config)

args = GlobalArgs()

model = build_model(args.cfg.model)
model.to(device).eval()


for filename in os.listdir(args.input_folder):
    if filename.lower().endswith(image_extensions):
        full_path = os.path.join(args.input_folder, filename)
        dirname = os.path.dirname(full_path)
        name, ext = os.path.splitext(os.path.basename(full_path))

        full_path_seg = os.path.join(args.output_folder_seg, filename)

        img = read_image(full_path).to(device).float().unsqueeze(0)
        mask, save_tensor_feature, save_tensor_segmap = get_mask(model, args, img)


        plot_qualitative(img.cpu()[0].permute(1,2,0).int().numpy(), mask.cpu()[0].numpy(), full_path_seg, palette)

        np_save_data(args.output_folder, name, save_tensor_feature, save_tensor_segmap)

        print(f"Папка: {dirname}")
        print(f"Имя файла: {name}")
        print("-" * 40)
