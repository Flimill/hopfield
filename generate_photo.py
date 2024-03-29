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

def image_path_to_binary_vector(image_path, threshold=128):
    image = Image.open(image_path).convert('L')  # Открываем изображение и преобразуем в оттенки серого
    return image_to_binary_vector(image, threshold)
 
def image_to_binary_vector(image, threshold=128):
    binary_image = image.convert('L').point(lambda x: 0 if x < threshold else 255, mode='1')  # Применяем пороговую бинаризацию
    binary_vector = np.array(binary_image).flatten()  # Изменяем размерность до одномерного массива (вектора)
    binary_vector = np.where(binary_vector, 1, -1)
    return binary_vector

def binary_vector_to_image(binary_vector, width, height):
    # Преобразование значений вектора обратно в значения пикселей
    binary_vector = np.where(binary_vector == 1, 255, 0)
    
    # Изменение размерности массива до двумерного массива (изображения)
    image_array = binary_vector.reshape((height, width)).astype(np.uint8)
    
    # Создание изображения из массива пикселей
    image = Image.fromarray(image_array)
    return image


def add_noise(image, noise_type='gaussian'):
    if noise_type == 'gaussian':
        row, col, ch = np.array(image).shape
        mean = 0
        var = 1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = np.array(image) + gauss
        noisy_image = Image.fromarray(noisy.astype(np.uint8))
        return image
    elif noise_type == 'salt_and_pepper':
        # Ваш код для добавления шума salt-and-pepper
        pass
    else:
        raise ValueError("Unsupported noise type. Choose from 'gaussian' or 'salt_and_pepper'.")


def get_noisy_picture(from_, w, h, noise_type='gaussian'):
    image = Image.open(from_)
    image = image.resize((w, h))  # Изменяем размер изображения
    noisy_image = add_noise(image, noise_type)
    noisy_image.show()  # Показываем изображение
    return noisy_image
