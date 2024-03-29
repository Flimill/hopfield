import numpy as np
import generate_photo
import time

import numpy as np
np.set_printoptions(threshold=np.inf)


class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))

    def train(self, patterns):
        num_patterns = len(patterns)
        
        for pattern in patterns:
            pattern = np.array(pattern)
            outer_product = np.outer(pattern, pattern)
            self.weights += outer_product / num_patterns  # Вычисление весов по формуле
        np.fill_diagonal(self.weights, 0)  # Диагональные элементы весов устанавливаем в 0

        # Проверка соблюдения условия Хопфилда для каждого образца после обучения
        #for pattern in patterns:
        #    pattern = np.array(pattern)
        #    reconstructed_pattern = np.sign(np.dot(self.weights, pattern))
        #    if not np.array_equal(pattern, reconstructed_pattern):
        #        print("Original pattern:", pattern)
        #        print("Reconstructed pattern:", reconstructed_pattern)
        #        raise ValueError("Hopfield network training failed. Condition Xi = W * Xi is not satisfied.")

    def predict(self, pattern, max_iters=100):
        pattern = np.array(pattern)
        for _ in range(max_iters):
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):
                return new_pattern
            pattern = new_pattern



# Пример использования
if __name__ == "__main__":

    weight = 320
    hight = 320
    w,h = weight,hight
    min_fig_size =10
    mfs=10
    num_shapes=1
    num_samples = 100
    train_path = "patterns"
    shape_types=['box','circle','triangle']
    #shape_types=['box']
    generate_photo.genererate_pictures(train_path, num_samples,w,h,min_fig_size,num_shapes)
    # Предположим, что у нас есть три различных образца: квадрат, круг и треугольник.
    # Каждый образец представлен в виде бинарного вектора.
    patterns=[]
    
    for type in shape_types:
        for i in range(num_samples):
            path=f"patterns/{type}/{type}-{i}.png"
            image_binary_vector = generate_photo.image_path_to_binary_vector(path)
            patterns.append(image_binary_vector)

    network = HopfieldNetwork(pattern_size=len(patterns[0]))
    network.train(patterns)

    for type in shape_types:
        path=f"patterns/{type}/{type}-1.png"
        noisy_image= generate_photo.get_noisy_picture(path,w,h)
        noisy_image_binary_vector = generate_photo.image_to_binary_vector(noisy_image)
        predicted_pattern=network.predict(noisy_image_binary_vector)
        result_image = generate_photo.binary_vector_to_image(predicted_pattern, w, h)
        result_image.show()
        time.sleep(5)












'''''
    # Теперь, если у нас есть искажённое изображение (например, с шумом), мы можем использовать сеть Хопфилда для восстановления оригинального образа.
    test_path = "test"
    
    num_samples = 5
    generate_photo.genererate_pictures(test_path, num_samples,w,h,min_fig_size,num_shapes)
    for i in range(num_samples):    
        test_pattern = generate_photo.image_to_binary_vector(f'test/boxes/box-{i}.png')
        predicted_pattern = network.predict(test_pattern)
        if np.array_equal(predicted_pattern, box):
            print("Predicted pattern represents a box.")
        elif np.array_equal(predicted_pattern, circle):
            print("Predicted pattern represents a circle.")
        elif np.array_equal(predicted_pattern, triangle):
            print("Predicted pattern represents a triangle.")
        else:
            print("Predicted pattern does not match any known pattern.")
'''