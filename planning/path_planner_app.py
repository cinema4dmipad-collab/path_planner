import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np


    try:
        from geometry import Contour
        from planner import PathPlanner
        from visualizer import Visualizer
        from load_txt import load_txt
        from load_dxf import load_dxf
        print("✅ Все модули planning загружены!")
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
    return Contour, PathPlanner, Visualizer, load_dxf, load_txt, mo, np


@app.cell
def _(mo):
    mo.md("""
    # 🛠️ **Планировщик траектории заливки**

    Интерактивный инструмент для построения траектории движения датчика по принципу "змейка".
    Загрузите файл с контуром (TXT или DXF) и настройте параметры.

    ---
    """)
    return


@app.cell
def _(mo):
    file_input = mo.ui.file(
        label="📁 Загрузить файл контура (TXT или DXF)",
        multiple=False,
        filetypes=[".txt", ".dxf"]
    )

    file_input
    return (file_input,)


@app.cell
def _(file_input, mo):
    import tempfile
    from pathlib import Path

    if not file_input.value:
        mo.md("⏳ Загрузите файл контура для начала работы")
        file_loaded = False
        filename = None
        tmp_path = None
    else:
        # Новый API marimo для работы с файлами
        uploaded_file = file_input.value[0]
        filename = uploaded_file.name

        # Получаем содержимое файла
        file_data = uploaded_file.contents  # ← вместо .data!

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        mo.md(f"✅ Загружен файл: **{filename}**")
        file_loaded = True
    return file_loaded, filename, tmp_path


@app.cell
def _(mo):
    tol_slider = mo.ui.slider(
        0.01, 2.0, step=0.01, value=0.1,
        label="🎯 **Tolerance** (мм)"
    ).form()

    step_slider = mo.ui.slider(
        0.5, 5.0, step=0.5, value=2.0,
        label="📏 **Шаг змейки** (мм)"
    ).form()

    angle_slider = mo.ui.slider(
        0, 90, step=1, value=0,
        label="↗️ **Угол заливки** (°)"
    ).form()

    mode_select = mo.ui.dropdown(
        options=["as_is", "less", "more"],
        value="as_is",
        label="⚙️ **Режим дискретизации**"
    ).form()

    mo.hstack([tol_slider, step_slider, angle_slider, mode_select], widths="equal", gap=1)
    return angle_slider, mode_select, step_slider, tol_slider


@app.cell
def _(angle_slider, mo, mode_select, step_slider, tol_slider):
    tol = tol_slider.value if tol_slider.value is not None else 0.1
    step = step_slider.value if step_slider.value is not None else 2.0
    angle = angle_slider.value if angle_slider.value is not None else 0
    mode = mode_select.value if mode_select.value is not None else "as_is"

    mo.md(
        f"""
        **Текущие параметры:**
        - Tolerance: `{tol:.3f}` мм
        - Шаг змейки: `{step:.1f}` мм
        - Угол: `{angle}°`
        - Режим дискретизации: `{mode}`
        """
    )
    return angle, mode, step, tol


@app.cell
def _(
    Contour,
    file_loaded,
    filename,
    load_dxf,
    load_txt,
    mo,
    np,
    tmp_path,
    tol,
):
    if not file_loaded:
        mo.md("⚠️ **Файл не загружен** — загрузите файл в ячейке 3")
        points = None
    else:
        try:
            if filename.lower().endswith('.txt'):
                points = load_txt(tmp_path)
                mo.md("✅ Загружен TXT-файл")
            
                # 👇 ДИАГНОСТИКА ПРЯМО ЗДЕСЬ
                print("🔍 ДИАГНОСТИКА ЗАГРУЖЕННОГО ФАЙЛА:")
                print(f"   Всего точек: {len(points)}")
                print(f"   Первая точка: [{points[0][0]:.2f}, {points[0][1]:.2f}]")
                print(f"   Последняя точка: [{points[-1][0]:.2f}, {points[-1][1]:.2f}]")
            
                # Проверка замкнутости
                dist = np.linalg.norm(points[-1] - points[0])
                print(f"   Расстояние между первой и последней: {dist:.6f}")
                print(f"   Контур замкнут? {dist < 1e-6}")
            
                # Диапазон по Y
                ymin, ymax = points[:, 1].min(), points[:, 1].max()
                print(f"   Диапазон Y: [{ymin:.2f}, {ymax:.2f}]")
            
                # Проверка пересечений на нескольких уровнях
                test_ys = np.linspace(ymin + 0.1, ymax - 0.1, 5)
                for y in test_ys:
                    temp_contour = Contour(points, approximation_tolerance=tol)
                    intersections = temp_contour.get_intersections(y)
                    print(f"   y={y:.2f}: {len(intersections)} пересечений")
            
            elif filename.lower().endswith('.dxf'):
                points = load_dxf(tmp_path, tolerance=tol)
                mo.md(f"✅ Загружен DXF-файл с tolerance={tol}")
            
            else:
                mo.md("❌ Неподдерживаемый формат. Используйте TXT или DXF")
                points = None
            
            if points is not None:
                mo.md(f"📊 Загружено **{len(points)}** точек")
        
        except Exception as e:
            mo.md(f"❌ Ошибка загрузки: {e}")
            points = None
    return (points,)


@app.cell
def _(Contour, PathPlanner, angle, mo, mode, np, points, step, tol):
    if points is None:
        mo.md("⏳ **Нет данных** — загрузите файл для построения траектории")
        contour = None
        path = np.array([])
        stats = {'total_points': 0, 'total_length': 0.0, 'num_lines': 0}
    else:
        print("🔍 Диагностика planner")
        print(f"points shape: {points.shape}")
        print(f"tol: {tol}")
        print(f"step: {step}")
        print(f"angle: {angle}")
        print(f"mode: {mode}")

        # Создаём контур
        contour = Contour(points, approximation_tolerance=tol)
        print(f"contour created: {len(contour.points)} points")

        # Применяем режим дискретизации
        contour = contour.discretize(mode=mode)
        print(f"after discretize: {len(contour.points)} points")

        # Создаём планировщик
        planner = PathPlanner(
            contour=contour,
            line_distance=step,
            fill_angle=angle,
            tolerance=tol
        )
        print("planner created")

        # Строим траекторию
        path = planner.generate_path()
        print(f"path generated, length: {len(path)}")

        stats = planner.get_statistics()

    mo.md(
        f"""
        ### 📊 Статистика траектории
        - Точек в контуре: **{len(contour.points) if contour else 0}**
        - Точек в траектории: **{stats['total_points']}**
        - Длина пути: **{stats['total_length']:.2f}** мм
        - Количество линий: **{stats['num_lines']}**
        """
    )
    return contour, path, stats


@app.cell
def _(Visualizer, contour, mode, path, points, stats, step, tol):
    import matplotlib.pyplot as plt

    # Создаём фигуру с тремя подграфиками
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    if contour is None or len(path) == 0:
        # Пустые графики
        ax1.set_title('Контур (нет данных)')
        ax2.set_title('Траектория (нет данных)')
        ax3.set_title('Дискретизация (нет данных)')
        for ax in [ax1, ax2, ax3]:
            ax.grid(True)
            ax.axis('equal')
    else:
        # Создаём визуализатор ТОЛЬКО если есть данные
        viz = Visualizer()

        # 1. Левый график: контур с пересечениями
        viz.plot_intersections(contour, num_levels=8, ax=ax1)
        info_text = f"Режим: {mode}\nTolerance: {tol:.3f} мм\nТочек: {len(contour.points)}"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 2. Центральный график: траектория
        viz.plot_path(contour, path, ax=ax2)
        stats_text = f"Шаг: {step:.1f} мм\nТочек пути: {len(path)}\nДлина пути: {stats['total_length']:.1f} мм"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 3. Правый график: сравнение дискретизации
        original_points = points
        ax3.plot(original_points[:, 0], original_points[:, 1], 
                 'gray', linewidth=1, alpha=0.5, label='Исходный')
        ax3.plot(contour.points[:, 0], contour.points[:, 1], 
                 'blue', linewidth=2, alpha=0.8, label='Дискретизир.')
        ax3.scatter(contour.points[:, 0], contour.points[:, 1], 
                    c='red', s=10, alpha=0.7, label=f'{len(contour.points)} точек')
        ax3.set_title('Сравнение дискретизации')
        ax3.legend(fontsize=8)

        disc_text = f"Исходных точек: {len(original_points)}\n"
        disc_text += f"После дискретизации: {len(contour.points)}\n"
        disc_text += f"Сжатие: {len(original_points)/len(contour.points):.1f}x"
        ax3.text(0.02, 0.98, disc_text, transform=ax3.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(contour, file_loaded, mo, np, path, points):
    mo.md("## 🔍 Диагностика состояния")

    diagnostics = []

    if 'file_loaded' in globals() and file_loaded:
        diagnostics.append("✅ Файл загружен")
    else:
        diagnostics.append("❌ Файл не загружен")

    if 'points' in globals() and points is not None:
        diagnostics.append(f"✅ Точки загружены: {len(points)} шт.")
    else:
        diagnostics.append("❌ Точки отсутствуют")

    if 'contour' in globals() and contour is not None:
        diagnostics.append(f"✅ Контур создан: {len(contour.points)} точек")
    else:
        diagnostics.append("❌ Контур не создан")

    if 'path' in globals() and isinstance(path, np.ndarray) and len(path) > 0:
        diagnostics.append(f"✅ Траектория построена: {len(path)} точек")
    else:
        diagnostics.append("❌ Траектория не построена")

    mo.md("\n".join(diagnostics))
    return


if __name__ == "__main__":
    app.run()
