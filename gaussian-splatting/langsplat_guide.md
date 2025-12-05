# Полный цикл обучения сцены RAMEN из LERF OVS RAMEN

## 1. Подготовка репозиториев и данных

1. **Склонируйте репозитории**  
   Склонируйте репозитории `gaussian-splatting` и `LangSplat` в одну папку, чтобы они находились рядом. Например:
   ```bash
   git clone https://github.com/graphdeco-inria/gaussian-splatting.git
   git clone https://github.com/minghanqin/LangSplat.git
   ```
   В итоге структура должна быть такой:
   ```
   <ваша_папка>/
   ├── gaussian-splatting/
   └── LangSplat/
   ```

2. **Положите датасет**  
   Поместите ваш датасет в папку `LangSplat/lerf_ovs/ramen/` (пример папки).  
   Пример структуры:
   ```
   LangSplat/
   └── lerf_ovs/
       └── ramen/
           ├── input/
           │   ├── image0.png
           │   ├── image1.png
           │   └── ...
           └── ...
   ```

---

## 2. Подготовка среды и запуск Gaussian Splatting

1. **Активируйте окружение**  
   Перейдите в папку `gaussian-splatting` и активируйте (установите) соответствующее conda-окружение:
   ```bash
   conda activate gaussian_splatting
   ```

2. **Конвертация датасета**  
   Выполните конвертацию датасета (предобротка с помощью Colmap):
   ```bash
   python convert.py -s ../LangSplat/lerf_ovs/ramen/
   ```

3. **Обучение Gaussian Splatting**  
   Запустите обучение:
   ```bash
   python train.py -s ../LangSplat/lerf_ovs/ramen/
   ```

4. **Просмотр результатов (опционально)**  
   Для просмотра результатов используйте SIBR_remoteGaussian_app (из 3DGS):
   ```bash
   SIRB_HOME/SIBR_remoteGaussian_app --path /media/titrom/storage/mipt/LangSplatDocker/LangSplat/lerf_ovs/ramen/
   ```

5. **Рендеринг результатов (опционально)**  
   Для рендеринга:
   ```bash
   python render.py -m output/lerf_ovs_ramen -s ../LangSplat/lerf_ovs/ramen/
   ```

---

## 3. Подготовка и запуск LangSplat

1. **Активируйте окружение**  
   Перейдите в папку `LangSplat` и активируйте (установите) окружение:
   ```bash
   conda activate langsplat
   ```

   Если на машине стоит cuda12, можно установить другое окружение (будет приложено в файле "environment_cuda12.yaml")
   ```bash
   conda activate langsplat_cuda12
   ```

2. **Извлечение языковых признаков сцены**  
   Выполните препроцессинг:
   ```bash
   python preprocess.py --dataset_path lerf_ovs/ramen
   ```

3. **Обучение автокодировщика**  
   Перейдите в папку `autoencoder`:
   ```bash
   cd autoencoder
   ```
   Запустите обучение автокодировщика:
   ```bash
   python train.py --dataset_path ../lerf_ovs/ramen --encoder_dims 256 128 64 32 3 --decoder_dims 16 32 64 128 256 256 512 --lr 0.0007 --dataset_name lerf_ovs_ramen
   ```

4. **Получение 3-мерных языковых признаков**  
   После обучения автокодировщика выполните:
   ```bash
   python test.py --dataset_path ../lerf_ovs/ramen --dataset_name lerf_ovs_ramen
   ```
   Вернитесь в корень проекта:
   ```bash
   cd ..
   ```

5. **Обучение LangSplat с разными уровнями признаков**  
   Запустите обучение для каждого уровня признаков:
   ```bash
   python train.py -m output/lerf_ovs_ramen --start_checkpoint ../gaussian-splatting/output/lerf_ovs_ramen/chkpnt30000.pth --feature_level 1 -s lerf_ovs/ramen
   python train.py -m output/lerf_ovs_ramen --start_checkpoint ../gaussian-splatting/output/lerf_ovs_ramen/chkpnt30000.pth --feature_level 2 -s lerf_ovs/ramen
   python train.py -m output/lerf_ovs_ramen --start_checkpoint ../gaussian-splatting/output/lerf_ovs_ramen/chkpnt30000.pth --feature_level 3 -s lerf_ovs/ramen
   ```

---

## 4. Визуализация результатов

1. **Запуск визуализатора**  
   Для онлайн-визуализации во время тренировки:
   ```bash
   SIRB_HOME/install/bin/SIBR_remoteGaussian_app --port 55555
   ```

2. **Рендеринг с языковыми признаками**  
   Для рендеринга с языковыми признаками:
   ```bash
   python render.py -m output/lerf_ovs_ramen_1 --include_feature --feature_level 1
   ```

3. **Онлайн-рендеринг с языковыми признаками**  
   Для онлайн-рендеринга:
   ```bash
   python run_renderer.py --start_checkpoint output/lerf_ovs_ramen_1/chkpnt30000.pth --feature_level 1 -s lerf_ovs/ramen --vis_language
   SIRB_HOME/install/bin/SIBR_remoteGaussian_app --port 55555
   ```
