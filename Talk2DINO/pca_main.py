import sys
import os

# Configure CUDA allocation behavior
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
import numpy as np
from torchvision.io import read_image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from omegaconf import OmegaConf

# Добавляем путь к библиотеке open_vocabulary_segmentation
sys.path.insert(0, "src/open_vocabulary_segmentation")
from models.dinotext import DINOText
from models import build_model

device = "cuda"

# Поддерживаемые расширения изображений
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

# Базовая палитра для отображения сегментаций
palette = [
    [255, 0, 0],
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    [0, 0, 255],
    [128, 128, 128]
]

def np_save_data(path, name, features_pca, segmap):
    """
    Сохраняет PCA-признаки и карту сегментации в формате .npy
    :param path: Папка для сохранения
    :param name: Базовое имя файлов
    :param features_pca: NumPy-массив PCA-признаков размером (N, 3)
    :param segmap: Тензор сегментации размером (1, H, W)
    """
    feat_out = features_pca
    seg_out = segmap.squeeze(0).cpu().numpy()

    save_feat = os.path.join(path, f"{name}_f.npy")
    save_seg  = os.path.join(path, f"{name}_s.npy")

    print(f"Saving PCA features to: {save_feat}")
    print(f"Saving segmap to: {save_seg}")

    np.save(save_feat, feat_out)
    np.save(save_seg, seg_out)


def plot_qualitative(image, mask, output_path, palette):
    """
    Накладывает сегментационную карту на изображение и сохраняет результат.
    """
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in np.unique(mask):
        color = palette[lbl] if lbl < len(palette) else np.random.randint(0, 256, 3, dtype=np.uint8)
        overlay[mask == lbl] = color

    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.imshow(overlay, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def get_mask(model, args, img):
    """
    Возвращает тензор маски, тензор признаков и тензор сегментации.
    """
    labels = args.textual_categories.replace('_', ' ').split(',')
    pal = palette.copy()
    # расширяем палитру, если нужно
    for _ in range(len(labels) - len(pal)):
        pal.append([np.random.randint(0, 255) for _ in range(3)])
    if args.with_background:
        pal.insert(0, [0,0,0])
        model.with_bg_clean = True

    with torch.no_grad():
        text_emb = model.build_dataset_class_tokens("sub_imagenet_template", labels)
        text_emb = model.build_text_embedding(text_emb)
        mask_logits, _, feats, segmap = model.generate_masks(
            img, img_metas=None, text_emb=text_emb,
            classnames=labels, apply_pamr=True)
        if args.with_background:
            bg = torch.ones_like(mask_logits[:, :1]) * 0.55
            mask_logits = torch.cat([bg, mask_logits], dim=1)
        mask = mask_logits.argmax(dim=1)
    return mask, feats, segmap, pal


class GlobalArgs:
    config = "src/open_vocabulary_segmentation/configs/cityscapes/dinotext_cityscapes_vitb_mlp_infonce.yml"
    with_background = False
    textual_categories = "bear,knife,bottle,robot,keys,headphones"
    input_folder = "./images"
    output_folder = "./output_pca"
    output_folder_seg = "./output_seg_pca"
    cfg = OmegaConf.load(config)


def main():
    args = GlobalArgs()
    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(args.output_folder_seg, exist_ok=True)

    model = build_model(args.cfg.model)
    model.to(device).eval()
    pca = PCA(n_components=512)

    print("Starting processing loop...")
    for f in os.listdir(args.input_folder):
        if not f.lower().endswith(image_extensions):
            continue
        full = os.path.join(args.input_folder, f)
        name, ext = os.path.splitext(f)
        out_vis = os.path.join(args.output_folder_seg, f)
        try:
            print(f"Processing: {f}")
            img_t = read_image(full).to(device).float().unsqueeze(0)
            img_vis = img_t.cpu().squeeze(0).permute(1,2,0).byte().numpy()

            mask, feats, segmap, pal = get_mask(model, args, img_t)
            feats_np = feats.squeeze(0).cpu().numpy()  # (N, D)
            feats_pca = pca.fit_transform(feats_np)   # (N, 3)

            plot_qualitative(img_vis, mask.cpu().squeeze(0).numpy(), out_vis, pal)
            print(f"Saved segmentation overlay to: {out_vis}")

            np_save_data(args.output_folder, name, feats_pca, segmap)
            print(f"Finished: {f}\n{'-'*40}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
            print('-'*40)
    print("All images processed.")

if __name__ == '__main__':
    main()
