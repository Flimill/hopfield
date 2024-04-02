import numpy as np
import generate_photo
import time
from PIL import Image, ImageDraw

import numpy as np
np.set_printoptions(threshold=np.inf)


class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))

    def train(self, patterns):
        num_patterns = len(patterns)
        train_count=0
        for pattern in patterns:
            train_count+=1
            print(f"train_count={train_count}")
            pattern = np.array(pattern)
            outer_product = np.outer(pattern, pattern)
            self.weights += outer_product / num_patterns  # Вычисление весов по формуле
        np.fill_diagonal(self.weights, 0)  # Диагональные элементы весов устанавливаем в 0
        
    def predict(self, pattern, max_iters=100):
        pattern = np.array(pattern)
        predict_count=0
        for _ in range(max_iters):
            predict_count+=1
            print(f"predict_count={predict_count}")
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):
                return new_pattern
            pattern = new_pattern
        print("max_iters")



# Пример использования
if __name__ == "__main__":

    weight = 64
    hight = 64
    w,h = weight,hight
    min_fig_size =10
    mfs=10
    num_shapes=1
    num_samples = 3
    train_path = "patterns"
    #shape_types=['box','circle','triangle']
    shape_types=['box']
    noise_level = 0.1
    
    #generate_photo.genererate_pictures(train_path, num_samples,w,h,min_fig_size,num_shapes)
    
    
    
    # Предположим, что у нас есть три различных образца: квадрат, круг и треугольник.
    # Каждый образец представлен в виде бинарного вектора.
    patterns=[]
    
    for type in shape_types:
        for i in range(num_samples):
            path=f"patterns/{i}.png"
            #image_binary_vector = generate_photo.image_path_to_binary_vector(path, threshold=128, new_width=w, new_height=h)
            image_binary_vector = generate_photo.image_path_to_vector(path,w,h)
            patterns.append(image_binary_vector)

    network = HopfieldNetwork(pattern_size=len(patterns[0]))
    network.train(patterns)
    for i in range(num_samples):
        for type in shape_types:
            path=f"patterns/{i}.png"
            generate_photo.vector_to_image(generate_photo.image_path_to_vector(path,w,h),w,h).show()
            
            noisy_image= generate_photo.get_noisy_picture(path,w,h, noise_level)
            generate_photo.vector_to_image(generate_photo.image_to_vector(noisy_image,w,h),w,h).show()
            noisy_image_binary_vector = generate_photo.image_to_vector(noisy_image,w,h)
            predicted_pattern=network.predict(noisy_image_binary_vector)
            result_image = generate_photo.vector_to_image(predicted_pattern, w, h)
            result_image.show()
            time.sleep(2)

