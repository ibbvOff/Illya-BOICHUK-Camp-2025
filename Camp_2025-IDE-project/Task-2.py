import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

SIZE = 20
STEPS = 20

start_grid = np.random.choice([0, 1], size=(SIZE, SIZE), p=[0.7, 0.3])
grid = start_grid.copy()

def update(grid):
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=0)
    neighbors = np.zeros(grid.shape, dtype=int)

    for dx in range(3):
        for dy in range(3):
            if dx == 1 and dy == 1:  
                continue
            neighbors += padded[dx:dx+SIZE, dy:dy+SIZE]

    new_grid = grid.copy()
    new_grid[(grid == 0) & (neighbors == 3)] = 1
    new_grid[(grid == 1) & ((neighbors < 2) | (neighbors > 3))] = 0
    
    return new_grid

grids = [start_grid.copy()]
alive_cells = [np.sum(start_grid)]

for _ in range(STEPS):
    grid = update(grid)
    grids.append(grid.copy())
    alive_cells.append(np.sum(grid))

def show_grid(grid, title, save_as=None):
    """Візуалізація гри"""
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap='Greens', interpolation='nearest')
    plt.title(title)
    plt.colorbar(ticks=[0, 1], label='Стан')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()

show_grid(start_grid, 'Початкова сітка', 'start_grid.png')
show_grid(grid, f'Сітка після {STEPS} ітерацій', 'final_grid.png')

plt.figure(figsize=(10, 6))
plt.plot(range(STEPS + 1), alive_cells, marker='o', linestyle='-')
plt.title('Кількість живих клітин')
plt.xlabel('Ітерація')
plt.ylabel('Живі клітини')
plt.grid(True)
plt.savefig('alive_cells.png')
plt.show()

print(f"Початкові живі клітини: {alive_cells[0]}")
print(f"Живі клітини після {STEPS} ітерацій: {alive_cells[-1]}")
print(f"Зміна кількості живих клітин: {alive_cells[-1] - alive_cells[0]}")