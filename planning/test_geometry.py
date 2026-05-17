# test_geometry_viz.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from geometry import Contour


def create_test_shapes():
    """Создаёт набор тестовых фигур."""
    shapes = {}


    shapes['square'] = np.array([
        [0, 0], [10, 0], [10, 10], [0, 10], [0, 0]
    ])


    t = np.linspace(0, 2 * np.pi, 50)
    shapes['circle'] = np.column_stack([
        10 * np.cos(t),
        10 * np.sin(t)
    ])

    # 3. Звезда
    shapes['star'] = np.array([
        [0, 10], [2, 3], [10, 3], [3, -2],
        [6, -10], [0, -5], [-6, -10], [-3, -2],
        [-10, 3], [-2, 3], [0, 10]
    ])

    # 4. Случайный многоугольник (шумный)
    np.random.seed(42)
    angles = np.linspace(0, 2 * np.pi, 20)
    r = 10 + np.random.normal(0, 2, 20)
    shapes['noisy'] = np.column_stack([
        r * np.cos(angles),
        r * np.sin(angles)
    ])

    return shapes


def test_discretization_modes():
    """Тестирует все режимы дискретизации на разных фигурах."""

    shapes = create_test_shapes()
    tolerances = [0.5, 1.0, 2.0]
    modes = [Contour.MODE_AS_IS, Contour.MODE_LESS, Contour.MODE_MORE]

    fig, axes = plt.subplots(
        len(shapes), len(modes) * len(tolerances) + 1,
        figsize=(20, 15)
    )

    for row, (shape_name, points) in enumerate(shapes.items()):
        col = 0
        # Исходная фигура
        ax = axes[row, col]
        ax.plot(points[:, 0], points[:, 1], 'b-', alpha=0.5)
        ax.scatter(points[:, 0], points[:, 1], c='red', s=10)
        ax.set_title(f'{shape_name}\nисходный\n{len(points)} точек')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

        # Тестируем разные tolerance и режимы
        col = 1
        for tol in tolerances:
            contour = Contour(points, approximation_tolerance=tol)

            for mode in modes:
                if col >= axes.shape[1]:
                    continue

                ax = axes[row, col]
                processed = contour.discretize(mode=mode)

                ax.plot(processed.points[:, 0], processed.points[:, 1],
                        'g-', linewidth=2)
                ax.scatter(processed.points[:, 0], processed.points[:, 1],
                           c='red', s=5, alpha=0.5)

                ax.set_title(
                    f'tol={tol}\n{mode}\n{len(processed.points)} точек')
                ax.axis('equal')
                ax.grid(True, alpha=0.3)

                col += 1

    plt.tight_layout()
    plt.savefig('discretization_test.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_perimeter_calculation():
    """Тестирует вычисление периметра."""
    shapes = create_test_shapes()

    print("\n🔵 ТЕСТ ПЕРИМЕТРА")
    print("-" * 40)

    for name, points in shapes.items():
        contour = Contour(points, approximation_tolerance=0.5)
        perim = contour.perimeter()

        # Для квадрата знаем точное значение
        if name == 'square':
            expected = 40.0
            error = abs(perim - expected) / expected * 100
            print(
                f"Квадрат: {perim:.2f} мм (ожидалось 40.0, ошибка {error:.2f}%)")
        else:
            print(f"{name}: {perim:.2f} мм")


def test_intersections():
    """Тестирует поиск пересечений."""

    # Квадрат 10x10
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
    contour = Contour(square, approximation_tolerance=0.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Тестируем разные уровни Y
    y_levels = [2, 5, 8]

    for idx, y in enumerate(y_levels):
        ax = axes[0, idx]

        # Рисуем квадрат
        ax.plot(contour.points[:, 0], contour.points[:, 1], 'b-', linewidth=2)

        # Рисуем линию
        ax.axhline(y=y, color='r', linestyle='--', alpha=0.5)

        # Находим и отмечаем пересечения
        intersections = contour.get_intersections(y)
        if intersections:
            ax.scatter(intersections, [y] * len(intersections),
                       color='green', s=100, zorder=5)
            ax.set_title(f'y={y}: пересечения {intersections}')
        else:
            ax.set_title(f'y={y}: нет пересечений')

        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    # Сложная фигура с несколькими пересечениями
    complex_shape = np.array([
        [0, 0], [5, 5], [10, 0], [15, 5], [20, 0],
        [20, 10], [15, 15], [10, 10], [5, 15], [0, 10], [0, 0]
    ])
    contour2 = Contour(complex_shape, approximation_tolerance=0.5)

    ax = axes[1, 0]
    ax.plot(contour2.points[:, 0], contour2.points[:, 1], 'b-', linewidth=2)

    for y in [3, 7, 12]:
        ax.axhline(y=y, color='gray', linestyle='--', alpha=0.3)
        intersections = contour2.get_intersections(y)
        if intersections:
            ax.scatter(intersections, [y] * len(intersections),
                       color='green', s=50)

    ax.set_title('Сложная фигура\nс множественными пересечениями')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    # График зависимости количества пересечений от Y
    ax = axes[1, 1]
    y_values = np.linspace(-2, 17, 100)
    intersection_counts = []

    for y in y_values:
        intersections = contour2.get_intersections(y)
        intersection_counts.append(len(intersections))

    ax.plot(y_values, intersection_counts, 'b-', linewidth=2)
    ax.set_xlabel('Y')
    ax.set_ylabel('Количество пересечений')
    ax.set_title('Зависимость числа пересечений от Y')
    ax.grid(True, alpha=0.3)

    # Статистика
    ax = axes[1, 2]
    ax.axis('off')

    stats_text = """
    📊 СТАТИСТИКА ПЕРЕСЕЧЕНИЙ

    Количество точек: {}
    Периметр: {:.1f} мм
    Границы X: [{:.1f}, {:.1f}]
    Границы Y: [{:.1f}, {:.1f}]
    """.format(
        len(contour2.points),
        contour2.perimeter(),
        *contour2.bounds
    )

    ax.text(0.1, 0.5, stats_text, fontsize=12,
            verticalalignment='center', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig('intersections_test.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_rotation():
    """Тестирует поворот контура."""

    # Создаём треугольник
    triangle = np.array([[0, 0], [10, 0], [5, 10], [0, 0]])
    contour = Contour(triangle, approximation_tolerance=0.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    angles = [0, 30, 45, 60, 90, 120]

    for idx, angle in enumerate(angles):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        rotated = contour.rotate(angle)

        ax.plot(contour.points[:, 0], contour.points[:, 1],
                'b--', alpha=0.5, label='Исходный')
        ax.plot(rotated.points[:, 0], rotated.points[:, 1],
                'r-', linewidth=2, label=f'Повёрнут на {angle}°')

        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(f'Поворот на {angle}°')

    plt.tight_layout()
    plt.savefig('rotation_test.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_bounds():
    """Тестирует вычисление границ."""

    shapes = create_test_shapes()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, points) in zip(axes.flat, shapes.items()):
        contour = Contour(points, approximation_tolerance=0.5)
        xmin, xmax, ymin, ymax = contour.bounds

        ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
        ax.scatter(points[:, 0], points[:, 1], c='red', s=20)

        # Рисуем границы
        ax.axvline(xmin, color='g', linestyle='--', alpha=0.7,
                   label=f'Xmin={xmin:.1f}')
        ax.axvline(xmax, color='g', linestyle='--', alpha=0.7,
                   label=f'Xmax={xmax:.1f}')
        ax.axhline(ymin, color='orange', linestyle='--', alpha=0.7,
                   label=f'Ymin={ymin:.1f}')
        ax.axhline(ymax, color='orange', linestyle='--', alpha=0.7,
                   label=f'Ymax={ymax:.1f}')

        ax.set_title(f'{name}\nграницы')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bounds_test.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("🔵 Тест 1: Границы контура")
    test_bounds()

    print("\n🔵 Тест 2: Периметр")
    test_perimeter_calculation()

    print("\n🔵 Тест 3: Дискретизация")
    test_discretization_modes()

    print("\n🔵 Тест 4: Пересечения")
    test_intersections()

    print("\n🔵 Тест 5: Поворот")
    test_rotation()

    print("\n✅ Все тесты завершены! Проверьте сохранённые PNG файлы.")