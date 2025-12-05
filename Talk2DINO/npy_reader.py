import numpy as np
import os

def read_npy_file(file_path):
    if not os.path.exists(file_path):
        print(f"Ошибка: Файл не найден по пути {file_path}")
        return None

    try:
        data = np.load(file_path)
        print(f"Успешно прочитан файл: {file_path}")
        print(f"Размерность данных (shape): {data.shape}")
        print(f"Тип данных (dtype): {data.dtype}")
        # print("Первые 5 элементов данных (пример):")
        # print(data.flatten()[:5]) # Распечатать первые несколько элементов

        return data

    except Exception as e:
        print(f"Произошла ошибка при чтении файла {file_path}: {e}")
        return None

# --- Пример использования ---
if __name__ == "__main__":
    # Укажите путь к вашему .npy файлу
    # Например, если у вас есть файл с именем 'my_data.npy' в той же директории:
    npy_file_name = "output_pca/00_f_pca.npy" # <--- Замените на имя вашего файла
    # Или укажите полный путь:
    # npy_file_path = "/путь/к/вашему/файлу/my_data.npy" # <--- Замените на полный путь

    # Создадим тестовый .npy файл для примера, если его нет
    if not os.path.exists(npy_file_name):
        test_data = np.random.rand(10, 5)
        np.save(npy_file_name, test_data)
        print(f"Создан тестовый файл {npy_file_name}")

    # Читаем файл
    loaded_data = read_npy_file(npy_file_name)

    # Теперь вы можете работать с loaded_data (это NumPy массив)
    if loaded_data is not None:
        print("\nДанные успешно загружены в переменную 'loaded_data'.")
        # Пример операции с загруженными данными:
        print("Файл:", loaded_data.shape)