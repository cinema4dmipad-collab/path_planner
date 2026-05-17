import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from logger import logger

from geometry import Contour
from planner import PathPlanner



class Visualizer:
    """
    Визуализатор контуров и траекторий.
    Позволяет отлаживать алгоритм и видеть результаты.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
        self.colors = {
            'original_contour': 'blue',
            'discretized_contour': 'orange',
            'rotated_contour': 'purple',
            'path': 'red',
            'intersections': 'green',
            'grid': 'gray',
            'start': 'green',
            'end': 'red'
        }

    def plot_contour(self,
                     contour: Contour,
                     title: str = "Контур",
                     show_points: bool = True,
                     color: Optional[str] = None,
                     ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Отображает контур.

        Args:
            contour: контур для отображения
            title: заголовок графика
            show_points: показывать ли вершины
            color: цвет линии (если None, используется цвет по умолчанию)
            ax: существующая ось (если None, создаётся новая)
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        x = contour.points[:, 0]
        y = contour.points[:, 1]

        color = color or self.colors['original_contour']

        ax.plot(x, y, color=color, linewidth=2,
                label=f'Контур ({len(contour.points)} точек)')

        if show_points:
            ax.scatter(x, y, color=color, s=20, zorder=5, alpha=0.7)

        xmin, xmax, ymin, ymax = contour.bounds
        padding = max(xmax - xmin, ymax - ymin) * 0.1
        ax.set_xlim(xmin - padding, xmax + padding)
        ax.set_ylim(ymin - padding, ymax + padding)

        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X, мм')
        ax.set_ylabel('Y, мм')
        ax.legend()

        return ax

    def plot_intersections(self,
                           contour: Contour,
                           y_levels: Optional[List[float]] = None,
                           num_levels: int = 10,
                           ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Отображает пересечения контура с горизонтальными линиями.

        Args:
            contour: контур
            y_levels: список уровней Y (если None, генерируются автоматически)
            num_levels: количество уровней для генерации
            ax: существующая ось
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        x = contour.points[:, 0]
        y = contour.points[:, 1]
        ax.plot(x, y, color=self.colors['original_contour'],
                linewidth=2, alpha=0.5, label='Контур')

        if y_levels is None:
            _, _, ymin, ymax = contour.bounds
            y_levels = np.linspace(ymin, ymax, num_levels)

        for y_level in y_levels:
            ax.axhline(y=y_level, color=self.colors['grid'],
                       linestyle='--', alpha=0.2, linewidth=0.5)

            intersections = contour.get_intersections(y_level)
            if intersections:
                ax.scatter(intersections, [y_level] * len(intersections),
                           color=self.colors['intersections'],
                           s=30, zorder=5, alpha=0.7)

        ax.set_aspect('equal')
        ax.set_title('Пересечения контура с горизонтальными линиями')
        ax.set_xlabel('X, мм')
        ax.set_ylabel('Y, мм')
        ax.legend()

        return ax

    def plot_path(self,
                  contour: Contour,
                  path: np.ndarray,
                  title: str = "Траектория",
                  show_points: bool = False,
                  ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Отображает траекторию поверх контура.

        Args:
            contour: исходный контур
            path: массив точек траектории (n, 2)
            title: заголовок графика
            show_points: показывать ли точки траектории
            ax: существующая ось
        """
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        x_cont = contour.points[:, 0]
        y_cont = contour.points[:, 1]
        ax.plot(x_cont, y_cont, color=self.colors['original_contour'],
                linewidth=2, alpha=0.5, label='Контур')

        if len(path) > 0:
            x_path = path[:, 0]
            y_path = path[:, 1]

            ax.plot(x_path, y_path, color=self.colors['path'],
                    linewidth=1.5, alpha=0.8, label='Траектория')

            if show_points:
                ax.scatter(x_path, y_path, color=self.colors['path'],
                           s=10, alpha=0.5)

            ax.scatter(path[0, 0], path[0, 1],
                       color=self.colors['start'], s=100,
                       marker='o', label='Старт', zorder=5)
            ax.scatter(path[-1, 0], path[-1, 1],
                       color=self.colors['end'], s=100,
                       marker='s', label='Финиш', zorder=5)

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('X, мм')
        ax.set_ylabel('Y, мм')
        ax.legend()

        return ax

    def plot_comparison(self,
                        original: Contour,
                        discretized: Contour,
                        rotated: Contour,
                        save_path: Optional[Path] = None) -> None:
        """
        Сравнивает исходный, дискретизированный и повёрнутый контуры.

        Args:
            original: исходный контур
            discretized: дискретизированный контур
            rotated: повёрнутый контур
            save_path: путь для сохранения
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))


        self.plot_contour(original,
                          title=f'Исходный ({len(original.points)} точек)',
                          color=self.colors['original_contour'],
                          ax=axes[0])


        self.plot_contour(discretized,
                          title=f'Дискретизированный\ntolerance={discretized.tolerance}, {len(discretized.points)} точек',
                          color=self.colors['discretized_contour'],
                          ax=axes[1])


        self.plot_contour(rotated,
                          title=f'Повёрнутый\nугол={rotated.angle if hasattr(rotated, "angle") else "?"}°',
                          color=self.colors['rotated_contour'],
                          ax=axes[2])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Сравнение сохранено в {save_path}")

        plt.show()

    def plot_all(self,
                 contour: Contour,
                 path: np.ndarray,
                 planner: Optional[PathPlanner] = None,
                 save_path: Optional[Path] = None) -> None:
        """
        Комплексная визуализация с 4 графиками.

        Args:
            contour: исходный контур
            path: траектория
            planner: объект планировщика (для статистики)
            save_path: путь для сохранения
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))


        self.plot_contour(contour, title='1. Исходный контур', ax=axes[0, 0])


        self.plot_intersections(contour, num_levels=8, ax=axes[0, 1])


        self.plot_path(contour, path, title='3. Готовая траектория',
                       ax=axes[1, 0])


        ax = axes[1, 1]
        ax.axis('off')

        stats_text = "📊 СТАТИСТИКА\n\n"
        stats_text += f"Точек в контуре: {len(contour.points)}\n"
        stats_text += f"Границы X: [{contour.bounds[0]:.1f}, {contour.bounds[1]:.1f}]\n"
        stats_text += f"Границы Y: [{contour.bounds[2]:.1f}, {contour.bounds[3]:.1f}]\n"

        if planner:
            stats = planner.get_statistics()
            stats_text += f"\nТочек траектории: {stats['total_points']}\n"
            stats_text += f"Длина пути: {stats['total_length']:.1f} мм\n"
            stats_text += f"Линий: {stats['num_lines']}\n"
            stats_text += f"Шаг: {stats['line_distance']} мм\n"
            stats_text += f"Угол: {stats['fill_angle']}°"

        ax.text(0.1, 0.9, stats_text, fontsize=12,
                verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Комплексная визуализация сохранена в {save_path}")

        plt.show()

    def save_all_plots(self,
                       contour: Contour,
                       path: np.ndarray,
                       planner: Optional[PathPlanner] = None,
                       output_dir: Path = Path("output")) -> None:
        """
        Сохраняет все виды визуализации в файлы.

        Args:
            contour: исходный контур
            path: траектория
            planner: объект планировщика
            output_dir: директория для сохранения
        """
        output_dir.mkdir(exist_ok=True)


        self.plot_contour(contour)
        plt.savefig(output_dir / "contour.png", dpi=150, bbox_inches='tight')


        self.plot_intersections(contour)
        plt.savefig(output_dir / "intersections.png", dpi=150,
                    bbox_inches='tight')


        self.plot_path(contour, path)
        plt.savefig(output_dir / "path.png", dpi=150, bbox_inches='tight')


        self.plot_all(contour, path, planner,
                      save_path=output_dir / "report.png")

        logger.info(f"Все визуализации сохранены в {output_dir}")
