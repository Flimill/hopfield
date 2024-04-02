import math
import os
import random
from shutil import rmtree
from PIL import Image, ImageDraw
import cv2
import numpy as np


# Функция для генерации случайного цвета
def random_color():
    #return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return (255,255,255)


    # Функция для создания изображения и сохранения его в указанную директорию
def draw_triangles_random_rotated(w, h, fc, mfs, num_shapes):
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for _ in range(num_shapes):
        side_length = random.randint(mfs, min(w, h) // 2 - 1)
        angle = random.uniform(0, 360)
        angle_rad = math.radians(angle)

        x1 = w // 2
        y1 = h // 2 - side_length // 2
        x2 = w // 2 - side_length // 2
        y2 = h // 2 + side_length // 2
        x3 = w // 2 + side_length // 2
        y3 = h // 2 + side_length // 2

        x1_rotated = int((x1 - w // 2) * math.cos(angle_rad) - (y1 - h // 2) * math.sin(angle_rad) + w // 2)
        y1_rotated = int((x1 - w // 2) * math.sin(angle_rad) + (y1 - h // 2) * math.cos(angle_rad) + h // 2)
        x2_rotated = int((x2 - w // 2) * math.cos(angle_rad) - (y2 - h // 2) * math.sin(angle_rad) + w // 2)
        y2_rotated = int((x2 - w // 2) * math.sin(angle_rad) + (y2 - h // 2) * math.cos(angle_rad) + h // 2)
        x3_rotated = int((x3 - w // 2) * math.cos(angle_rad) - (y3 - h // 2) * math.sin(angle_rad) + w // 2)
        y3_rotated = int((x3 - w // 2) * math.sin(angle_rad) + (y3 - h // 2) * math.cos(angle_rad) + h // 2)

        draw.polygon([(x1_rotated, y1_rotated), (x2_rotated, y2_rotated), (x3_rotated, y3_rotated)], fill=fc, outline=(0, 0, 0))
    return img

def draw_boxes_random_rotated(w, h, fc, mfs, num_shapes):
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for _ in range(num_shapes):
        side_length = random.randint(mfs, min(w, h) // 2 - 1)
        angle = random.uniform(0, 360)
        angle_rad = math.radians(angle)

        x_center = random.randint(side_length // 2, w - side_length // 2)
        y_center = random.randint(side_length // 2, h - side_length // 2)

        half_length = side_length // 2
        half_width = side_length // 2

        x1_rotated = int(x_center - half_length * math.cos(angle_rad) + half_width * math.sin(angle_rad))
        y1_rotated = int(y_center - half_length * math.sin(angle_rad) - half_width * math.cos(angle_rad))
        x2_rotated = int(x_center + half_length * math.cos(angle_rad) + half_width * math.sin(angle_rad))
        y2_rotated = int(y_center + half_length * math.sin(angle_rad) - half_width * math.cos(angle_rad))
        x3_rotated = int(x_center + half_length * math.cos(angle_rad) - half_width * math.sin(angle_rad))
        y3_rotated = int(y_center + half_length * math.sin(angle_rad) + half_width * math.cos(angle_rad))
        x4_rotated = int(x_center - half_length * math.cos(angle_rad) - half_width * math.sin(angle_rad))
        y4_rotated = int(y_center - half_length * math.sin(angle_rad) + half_width * math.cos(angle_rad))

        draw.polygon([(x1_rotated, y1_rotated), (x2_rotated, y2_rotated), (x3_rotated, y3_rotated), (x4_rotated, y4_rotated)], fill=fc, outline=(0, 0, 0))
    return img

def draw_circles_random_rotated(w, h, fc, mfs, num_shapes):
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for _ in range(num_shapes):
        r = random.randint(mfs // 2, min(w, h) // 2 - 1)
        angle = random.uniform(0, 360)
        angle_rad = math.radians(angle)

        x = random.randint(r, w - r)
        y = random.randint(r, h - r)

        x_rotated = int((x - w // 2) * math.cos(angle_rad) - (y - h // 2) * math.sin(angle_rad) + w // 2)
        y_rotated = int((x - w // 2) * math.sin(angle_rad) + (y - h // 2) * math.cos(angle_rad) + h // 2)

        draw.ellipse((x_rotated - r, y_rotated - r, x_rotated + r, y_rotated + r), fill=fc, outline=(0, 0, 0))
    return img

def genererate_pictures(_path, num_samples,w,h,min_fig_size,num_shapes):


    

    if os.path.exists(_path): 
        rmtree(_path)
    os.makedirs(_path)
    _folder = os.path.join(_path, "box")
    if not os.path.exists(_folder): os.makedirs(_folder)
    _folder = os.path.join(_path, "circle")
    if not os.path.exists(_folder): os.makedirs(_folder)
    _folder = os.path.join(_path, "triangle")
    if not os.path.exists(_folder): os.makedirs(_folder)



    for i in range(num_samples):
        # Генерация прямоугольника с одним из 7 цветов
        img_box = draw_boxes_random_rotated(w=w, h=h, fc=random_color(), mfs=min_fig_size, num_shapes=num_shapes)
        img_box.save(os.path.join(os.path.join(_path, "box"), f"box-{i}.png"))

        # Генерация круга с одним из 7 цветов
        img_circle = draw_circles_random_rotated(w=w, h=h, fc=random_color(), mfs=min_fig_size, num_shapes=num_shapes)
        img_circle.save(os.path.join(os.path.join(_path, "circle"), f"circle-{i}.png"))

        # Генерация круга с одним из 7 цветов
        img_treangle = draw_triangles_random_rotated(w=w, h=h, fc=random_color(), mfs=min_fig_size,num_shapes=num_shapes)
        img_treangle.save(os.path.join(os.path.join(_path, "triangle"), f"triangle-{i}.png"))

def resize_image(image_path, new_width, new_height):
    image = Image.open(image_path).convert('L')  # Открываем изображение и преобразуем в оттенки серого
    resized_image = image.resize((new_width, new_height))  # Изменяем размер изображения
    return resized_image

def image_to_vector(image,new_width,new_height):
    image = image.resize((new_width, new_height)).convert('L')  # Открываем изображение и конвертируем его в черно-белое
    image_array = np.array(image)
    binary_vector = []
    for pixel_value in image_array.flatten():
        # Преобразуем каждое значение пикселя в 5 элементов 0 или 1, представляющих оттенки серого
        binary_vector.extend([1 if pixel_value >= i * 51 else 0 for i in range(1, 6)])
    binary_vector = np.where(binary_vector, 1, -1)
    return binary_vector

def image_path_to_vector(image_path,new_width,new_height):
    image = resize_image(image_path, new_width, new_height).convert('L')  # Открываем изображение и конвертируем его в черно-белое
    image_array = np.array(image)
    binary_vector = []
    for pixel_value in image_array.flatten():
        # Преобразуем каждое значение пикселя в 5 элементов 0 или 1, представляющих оттенки серого
        binary_vector.extend([1 if pixel_value >= i * 51 else 0 for i in range(1, 6)])
    binary_vector = np.where(binary_vector, 1, -1)
    return binary_vector

def vector_to_image(binary_vector, width, height):
    if len(binary_vector) % 5 != 0:
        raise ValueError("Invalid binary vector length. It should be divisible by 5.")
    binary_vector = np.where(binary_vector == -1, 0, binary_vector)
    pixel_values = []
    for i in range(0, len(binary_vector), 5):
        # Обратное преобразование: каждые 5 элементов вектора переводим в значение пикселя
        shade_value = sum(binary_vector[i:i+5]) * 51
        pixel_values.append(shade_value)
    
    # Изменяем размерность массива значений пикселей и создаём изображение
    image_array = np.array(pixel_values, dtype=np.uint8).reshape((height, width))
    return Image.fromarray(image_array)

def add_noise(image, noise_level):

        img_array = np.array(image.convert('L'))
        # Генерация случайных координат для добавления шума
        salt_and_pepper = np.random.rand(*img_array.shape)
        
        # Добавление salt-and-pepper шума к изображению
        img_array[salt_and_pepper < noise_level/2] = 0
        img_array[salt_and_pepper > 1 - noise_level/2] = 255
        
        # Преобразование массива обратно в изображение PIL
        noisy_image = Image.fromarray(img_array)
        return noisy_image


def get_noisy_picture(from_, w, h, noise_level):
    image = Image.open(from_)
    image = image.resize((w, h))  # Изменяем размер изображения
    noisy_image = add_noise(image, noise_level)
    return noisy_image