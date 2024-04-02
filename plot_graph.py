import time
import matplotlib.pyplot as plt

def save_graph(array, accuracy_array, array_name):
    filename = f'{array_name}.png'
    time.sleep(1)
# Создаем новое окно для графика
    plt.figure()

    positions = list(range(len(array)))


    # Строим график, используя позиции в качестве абсцисс
    plt.plot(positions, accuracy_array, marker='o')

    # Устанавливаем метки на оси абсцисс с использованием значений из array
    plt.xticks(positions, array)
    
    plt.title(f'Зависимость accuracy от {array_name}')
    plt.xlabel(array_name)
    plt.ylabel('Точность (accuracy)')
    plt.grid(True)
    plt.savefig(filename) # Сохраняем график в файл
    plt.close() # Закрываем окно графика