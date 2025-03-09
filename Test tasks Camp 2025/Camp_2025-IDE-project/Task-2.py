import numpy as np
import matplotlib.pyplot as plt 

SIZE = 20  # Розмір сітки (20x20 клітин)
STEPS = 20  # Кількість кроків симуляції

# Створення початкової сітки з випадковим розподілом: 70% мертвих і 30% живих клітин
start_grid = np.random.choice([0, 1], size=(SIZE, SIZE), p=[0.7, 0.3])
grid = start_grid.copy()  # Створення копії для подальшого використання

def update(grid):
    """
    Функція для оновлення стану сітки згідно з правилами гри життя:
    1. Жива клітина з менше ніж 2 сусідами помирає (самотність)
    2. Жива клітина з 2-3 сусідами виживає
    3. Жива клітина з більше ніж 3 сусідами помирає (перенаселення)
    4. Мертва клітина з рівно 3 сусідами стає живою (розмноження)
    """
    # Створення розширеної сітки з нулями по краях для спрощення обчислення кількості сусідів
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
    neighbors = np.zeros(grid.shape, dtype=int)  # Масив для підрахунку кількості сусідів

    # Обчислення кількості сусідів для кожної клітини
    for dx in range(3):
        for dy in range(3):
            if dx == 1 and dy == 1:  # Пропускаємо центральну клітину (вона не є сусідом сама собі)
                continue
            neighbors += padded[dx:dx+SIZE, dy:dy+SIZE]  # Додаємо значення сусідів до загальної кількості

    # Створення нової сітки на основі поточної
    new_grid = grid.copy()
    
    # Застосування правил гри життя:
    # 1. Мертва клітина (0) з рівно 3 сусідами оживає (1)
    new_grid[(grid == 0) & (neighbors == 3)] = 1
    
    # 2. Жива клітина (1) з менше ніж 2 або більше ніж 3 сусідами помирає (0)
    new_grid[(grid == 1) & ((neighbors < 2) | (neighbors > 3))] = 0
    
    return new_grid

grids = [start_grid.copy()]  # Список для зберігання всіх станів сітки
alive_cells = [np.sum(start_grid)]  # Список для зберігання кількості живих клітин на кожному кроці

for _ in range(STEPS):
    grid = update(grid)  # Оновлення стану сітки
    grids.append(grid.copy())  # Збереження поточного стану
    alive_cells.append(np.sum(grid))  # Підрахунок кількості живих клітин

def show_grid(grid, title, save_as=None):
    """
    Функція для візуалізації стану сітки
    
    Параметри:
    grid (numpy.ndarray) - сітка для відображення
    title (str) - заголовок графіка
    save_as (str, optional) - шлях для збереження зображення
    """
    plt.figure(figsize=(8, 8))  # Створення фігури заданого розміру
    plt.imshow(grid, cmap='Greens', interpolation='nearest')  # Відображення сітки у відтінках зеленого
    plt.title(title)  # Встановлення заголовка
    plt.colorbar(ticks=[0, 1], label='Стан')  # Додавання шкали кольорів
    plt.xticks([])  # Вимкнення відображення позначок осі X
    plt.yticks([])  # Вимкнення відображення позначок осі Y
    plt.tight_layout()  # Оптимізація розміщення елементів
    if save_as:  # Якщо вказано шлях для збереження
        plt.savefig(save_as)  # Зберігаємо зображення
    plt.show()  # Відображаємо графік

# Візуалізація початкового та кінцевого станів сітки
show_grid(start_grid, 'Початкова сітка', 'start_grid.png')
show_grid(grid, f'Сітка після {STEPS} ітерацій', 'final_grid.png')

# Побудова графіка зміни кількості живих клітин з часом
plt.figure(figsize=(10, 6))
plt.plot(range(STEPS + 1), alive_cells, marker='o', linestyle='-')
plt.title('Кількість живих клітин')
plt.xlabel('Ітерація')
plt.ylabel('Живі клітини')
plt.grid(True)
plt.savefig('alive_cells.png')
plt.show()

# Виведення статистики
print(f"Початкові живі клітини: {alive_cells[0]}")
print(f"Живі клітини після {STEPS} ітерацій: {alive_cells[-1]}")
print(f"Зміна кількості живих клітин: {alive_cells[-1] - alive_cells[0]}")