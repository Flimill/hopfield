from PIL import Image
import numpy as np

def image_to_vector(image_path, threshold=128):
    image = Image.open(image_path).convert('L')  # Открываем изображение и конвертируем его в черно-белое
    image_array = np.array(image)
    binary_vector = []
    for pixel_value in image_array.flatten():
        # Преобразуем каждое значение пикселя в 5 элементов 0 или 1, представляющих оттенки серого
        binary_vector.extend([1 if pixel_value >= i * 51 else 0 for i in range(1, 6)])
    return binary_vector

def vector_to_image(binary_vector, width, height):
    if len(binary_vector) % 5 != 0:
        raise ValueError("Invalid binary vector length. It should be divisible by 5.")
    
    pixel_values = []
    for i in range(0, len(binary_vector), 5):
        # Обратное преобразование: каждые 5 элементов вектора переводим в значение пикселя
        shade_value = sum(binary_vector[i:i+5]) * 51
        pixel_values.append(shade_value)
    
    # Изменяем размерность массива значений пикселей и создаём изображение
    image_array = np.array(pixel_values, dtype=np.uint8).reshape((height, width))
    return Image.fromarray(image_array)

# Example usage:
image_path = 'patterns/box/box-0.png'
binary_vector = image_to_vector(image_path)
reconstructed_image = vector_to_image(binary_vector, width=128, height=128)
reconstructed_image.show()