import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, List
from logger import logger

from geometry import Contour
from planner import PathPlanner

if TYPE_CHECKING:
    from batch_planner import ContourPlanResult



class Visualizer:
    """
    Визуализатор контуров и траекторий.
    Позволяет отлаживать алгоритм и видеть результаты.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
        self.colors = {
            'original_contour': 'blue',
            'raw_contour': '#616161',
            'hole_contour': '#FF6F00',
            'discretized_contour': 'orange',
            'rotated_contour': 'purple',
            'path': '#D50000',
            'path_fill': '#D50000',
            'path_travel': '#1565C0',
            'intersections': 'green',
            'grid': 'gray',
            'start': 'green',
            'current': '#7B1FA2',
            'end': 'darkred',
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
                  path_segments: Tuple[Tuple[int, int, str], ...] = (),
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
            self._plot_segmented_path(
                ax,
                path,
                path_segments,
                show_fill_label=True,
                show_travel_label=True,
            )

            if show_points:
                ax.scatter(
                    path[:, 0],
                    path[:, 1],
                    color=self.colors['path_fill'],
                    s=10,
                    alpha=0.5,
                )

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

    def _plot_hole_points(
        self,
        ax: plt.Axes,
        hole_points: np.ndarray,
        *,
        label: Optional[str] = None,
    ) -> None:
        ax.plot(
            hole_points[:, 0],
            hole_points[:, 1],
            color=self.colors['hole_contour'],
            linewidth=1.8,
            linestyle="--",
            alpha=0.95,
            label=label,
            zorder=6,
        )
        ax.scatter(
            self._sample_points(hole_points)[:, 0],
            self._sample_points(hole_points)[:, 1],
            color=self.colors['hole_contour'],
            s=12,
            alpha=0.75,
            zorder=7,
        )

    @staticmethod
    def _sample_points(points: np.ndarray, max_points: int = 1500) -> np.ndarray:
        """Ограничивает число маркеров, чтобы большие DXF быстро рисовались."""
        if len(points) <= max_points:
            return points
        indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
        return points[indices]

    def _plot_segmented_path(
        self,
        ax: plt.Axes,
        path: np.ndarray,
        segments: Tuple[Tuple[int, int, str], ...],
        *,
        show_fill_label: bool = False,
        show_travel_label: bool = False,
    ) -> None:
        """Рисует заливку и переходы сканера разными цветами."""
        if len(path) == 0:
            return

        if not segments:
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=self.colors['path_fill'],
                linewidth=2.5,
                alpha=1.0,
                solid_capstyle="round",
                label="Заливка" if show_fill_label else None,
                zorder=10,
            )
            return

        fill_label_used = not show_fill_label
        travel_label_used = not show_travel_label

        for start_idx, end_idx, kind in segments:
            if end_idx <= start_idx:
                continue
            segment_path = path[start_idx : end_idx + 1]
            if len(segment_path) < 2:
                continue

            if kind == "fill":
                label = None if fill_label_used else "Заливка"
                fill_label_used = True
                ax.plot(
                    segment_path[:, 0],
                    segment_path[:, 1],
                    color=self.colors['path_fill'],
                    linewidth=2.5,
                    alpha=1.0,
                    solid_capstyle="round",
                    label=label,
                    zorder=10,
                )
            elif kind == "teleport_travel":
                label = None if travel_label_used else "Переход сканера"
                travel_label_used = True
                ax.plot(
                    segment_path[:, 0],
                    segment_path[:, 1],
                    color=self.colors['path_travel'],
                    linewidth=1.8,
                    alpha=0.75,
                    linestyle=(0, (4, 2)),
                    dash_capstyle="round",
                    label=label,
                    zorder=9,
                )
            else:
                label = None if travel_label_used else "Переход сканера"
                travel_label_used = True
                ax.plot(
                    segment_path[:, 0],
                    segment_path[:, 1],
                    color=self.colors['path_travel'],
                    linewidth=1.8,
                    alpha=0.95,
                    linestyle="--",
                    dash_capstyle="round",
                    label=label,
                    zorder=9,
                )

    @staticmethod
    def _clip_path_data(
        path: np.ndarray,
        segments: Tuple[Tuple[int, int, str], ...],
        point_limit: Optional[int],
    ) -> Tuple[np.ndarray, Tuple[Tuple[int, int, str], ...]]:
        """Возвращает первые N точек траектории и соответствующие сегменты."""
        if point_limit is None or point_limit >= len(path):
            return path, segments
        if point_limit <= 0:
            return np.array([]), ()

        clipped_path = path[:point_limit]
        clipped_segments = []
        last_idx = point_limit - 1
        for start_idx, end_idx, kind in segments:
            if start_idx >= point_limit:
                continue
            clipped_end = min(end_idx, last_idx)
            if clipped_end > start_idx:
                clipped_segments.append((start_idx, clipped_end, kind))
        return clipped_path, tuple(clipped_segments)

    def plot_multiple_results(
        self,
        results: List["ContourPlanResult"],
        *,
        ax: Optional[plt.Axes] = None,
        show_paths: bool = True,
        show_holes: bool = True,
        path_point_limit: Optional[int] = None,
        title: str = "Все контуры и траектории",
    ) -> plt.Axes:
        """Отображает внешние контуры, отверстия и траектории заливки."""
        if ax is None:
            _, ax = plt.subplots(figsize=self.figsize)

        outer_count = 0
        hole_labels = 0
        fill_label_shown = False
        travel_label_shown = False
        path_points_label_shown = False
        remaining_path_points = path_point_limit

        for result in results:
            if result.contour is None:
                if (
                    result.skipped
                    and result.original_points is not None
                    and len(result.original_points) > 1
                    and "отверстие" not in result.skip_reason
                ):
                    ax.plot(
                        result.original_points[:, 0],
                        result.original_points[:, 1],
                        color="#9E9E9E",
                        linewidth=1.5,
                        linestyle=":",
                        alpha=0.9,
                        label="Контур (ошибка)" if outer_count == 0 else None,
                        zorder=4,
                    )
                continue

            if result.skipped:
                continue

            outer_count += 1
            contour = result.contour

            if result.original_points is not None and len(result.original_points) > 1:
                raw_label = "Исходный контур" if outer_count == 1 else None
                ax.plot(
                    result.original_points[:, 0],
                    result.original_points[:, 1],
                    color=self.colors['raw_contour'],
                    linewidth=1.4,
                    linestyle=":",
                    alpha=0.75,
                    label=raw_label,
                    zorder=3,
                )
                ax.scatter(
                    self._sample_points(result.original_points)[:, 0],
                    self._sample_points(result.original_points)[:, 1],
                    color=self.colors['raw_contour'],
                    s=14,
                    alpha=0.6,
                    zorder=4,
                )

            ax.plot(
                contour.points[:, 0],
                contour.points[:, 1],
                color=self.colors['discretized_contour'],
                linewidth=2.0,
                alpha=0.95,
                label="Дискретизированный контур" if outer_count == 1 else None,
                zorder=5,
            )
            ax.scatter(
                self._sample_points(contour.points)[:, 0],
                self._sample_points(contour.points)[:, 1],
                color=self.colors['discretized_contour'],
                s=18,
                alpha=0.85,
                zorder=6,
            )

            if show_holes:
                for hole_idx, hole_pts in enumerate(result.hole_points):
                    label = "Отверстие" if hole_labels == 0 else None
                    self._plot_hole_points(ax, hole_pts, label=label)
                    hole_labels += 1

            if show_paths and len(result.path) > 0:
                if remaining_path_points is not None and remaining_path_points <= 0:
                    continue
                current_limit = (
                    None
                    if remaining_path_points is None
                    else min(remaining_path_points, len(result.path))
                )
                visible_path, visible_segments = self._clip_path_data(
                    result.path,
                    result.path_segments,
                    current_limit,
                )
                if len(visible_path) == 0:
                    continue

                self._plot_segmented_path(
                    ax,
                    visible_path,
                    visible_segments,
                    show_fill_label=not fill_label_shown,
                    show_travel_label=not travel_label_shown,
                )
                if visible_segments:
                    fill_label_shown = True
                    travel_label_shown = True
                else:
                    fill_label_shown = True
                ax.scatter(
                    self._sample_points(visible_path)[:, 0],
                    self._sample_points(visible_path)[:, 1],
                    color="#212121",
                    s=10,
                    alpha=0.65,
                    label="Точки траектории" if not path_points_label_shown else None,
                    zorder=12,
                )
                path_points_label_shown = True
                ax.scatter(
                    visible_path[0, 0],
                    visible_path[0, 1],
                    color=self.colors['start'],
                    s=65,
                    marker='o',
                    label="Старт" if outer_count == 1 else None,
                    zorder=13,
                )
                current_is_finish = len(visible_path) == len(result.path)
                ax.scatter(
                    visible_path[-1, 0],
                    visible_path[-1, 1],
                    color=self.colors['end'] if current_is_finish else self.colors['current'],
                    s=75,
                    marker='s' if current_is_finish else 'D',
                    label=(
                        "Финиш" if current_is_finish and outer_count == 1
                        else "Текущая точка" if not current_is_finish and outer_count == 1
                        else None
                    ),
                    zorder=13,
                )
                if remaining_path_points is not None:
                    remaining_path_points -= len(result.path)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("X, мм")
        ax.set_ylabel("Y, мм")
        if outer_count or hole_labels:
            ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

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
