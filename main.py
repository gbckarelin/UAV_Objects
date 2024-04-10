import numpy as np
import matplotlib.pyplot as plt

# Задаем координаты и приоритеты объектов
objects = np.array([
    [10,10,0.7],
    [20,20,0.2],
    [30,10,0.8],
    [5,20,0.3],
    [5,5,0.9],
    [40,10,0.1]
])

# Задаем координаты БПЛА
drones = np.array([
    [5,10],
    [15,10],
    [25,10],
    [50,10],
    [5,30],
    [10,20]
])

# Вычисляем матрицу удаленности между объектами и БПЛА
def compute_distance_matrix(objects, drones):
    num_objects = objects.shape[0]
    num_drones = drones.shape[0]
    distance_matrix = np.zeros((num_objects, num_drones))

    for i in range(num_objects):
        for j in range(num_drones):
            distance_matrix[i,j] = np.linalg.norm(objects[i,:2] - drones[j,:2])
    return distance_matrix

# Вычисляем матрицу приоритетов объекта
priority_matrix = objects[:,2].reshape(-1,1)

# Вычисляем матрицу стоимостей выполнения задачи БПЛА
cost_matrix = compute_distance_matrix(objects, drones)

# Вычисляем матрицу минимаксных значений для каждого объекта и соритруем
def minimax_allocation(priority_matrix, cost_matrix):
    num_objects = priority_matrix.shape[0]
    num_drones = cost_matrix.shape[1]

    # Вычисляем минимаксную оценку для каждого объекта 
    minimax_scores = []
    for obj_idx in range(num_objects):
        priority_scores = priority_matrix[obj_idx]
        max_cost = np.max(cost_matrix[obj_idx])
        minimax_score = np.max(priority_scores) * max_cost
        minimax_scores.append(minimax_score)

        # сортируем
    sorted_objects = np.argsort(minimax_scores)[::-1]
    print("Распределение БПЛА на объекты:")
    print(len(sorted_objects))
    for i in [1,2,3,4,5,6]:
        print(f'БПЛА {i} -> {sorted_objects[i-1]}')

        # Распределяем БПЛА на объекты
        allocation = np.zeros((num_objects,num_drones))
        assigned_drones = set()
        for obj_idx in sorted_objects:
            avaliable_drones = np.where(allocation[obj_idx] == 0)[0]
            avaliable_drones = [d for d in avaliable_drones if d not in assigned_drones]
            if len(avaliable_drones) > 0:
                # Выбираем БПЛА с минимальным расстоянием до данного объекта
                min_cost_drone = np.argmin(cost_matrix[obj_idx][avaliable_drones])
                drone_idx = avaliable_drones[min_cost_drone]
                allocation[obj_idx][drone_idx] = 1
                assigned_drones.add(drone_idx)

        return allocation
    
# Функция для визуализации распределения БПЛА на объекты
def visualize_allocation_with_lines(objects, drones, allocation):
    plt.figure(figsize=(8,8))
    fig, ax = plt.subplots()

    # Рисуем
    ax.scatter(objects[:, 0], objects[:, 1], color = 'red', label = 'Объекты')
    for i, (x,y, _) in enumerate(objects):
        ax.annotate(f'Obj{i+1}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')


    # Рисуем
    ax.scatter(drones[:, 0], drones[:, 1], color = 'blue', label = 'БПЛА')
    for i, (x,y) in enumerate(drones):
        ax.annotate(f'Drone{i+1}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')
        
    # Визуализация соединений
    for i in range(allocation.shape[0]):
        for j in range(allocation.shape[1]):
            if allocation[i, j] == 1:
                plt.plot([objects[i, 0], drones[j, 0]], [objects[i, 1], drones[j, 1]], color = 'green')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Minimax Allocation')
    plt.legend()
    plt.grid(True)
    plt.show()

allocation = minimax_allocation(priority_matrix, cost_matrix)
# visualize_allocation(objects, drones, allocation)
visualize_allocation_with_lines(objects, drones, allocation)