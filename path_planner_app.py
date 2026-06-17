import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np

    _cwd = Path.cwd()
    _planning_dir = (
        _cwd / "planning"
        if (_cwd / "planning" / "batch_planner.py").exists()
        else _cwd
    )
    if str(_planning_dir) not in sys.path:
        sys.path.insert(0, str(_planning_dir))

    from batch_planner import plan_file_contours, summarize_results
    from exceptions import (
        BridgePlanningError,
        ContourError,
        DXFLoadError,
        EmptyFileError,
        InsufficientPointsError,
        InvalidFormatError,
    )
    from load_txt import load_txt
    from visualizer import Visualizer

    return (
        BridgePlanningError,
        ContourError,
        DXFLoadError,
        EmptyFileError,
        InsufficientPointsError,
        InvalidFormatError,
        Visualizer,
        load_txt,
        mo,
        np,
        plan_file_contours,
        summarize_results,
    )


@app.cell
def _(mo):
    mo.md("""
    # 🛠️ **Планировщик траектории заливки**

    Интерактивный инструмент для построения траектории движения датчика по принципу "змейка".
    Загрузите файл с контуром (TXT или DXF) и настройте параметры.

    **DXF:** обрабатываются **все** замкнутые контуры файла.

    ---
    """)
    return


@app.cell
def _(mo):
    file_input = mo.ui.file(
        label="📁 Загрузить файл контура (TXT или DXF)",
        multiple=False,
        filetypes=[".txt", ".dxf"],
    )

    file_input
    return (file_input,)


@app.cell
def _(file_input, mo):
    import tempfile

    if not file_input.value:
        mo.md("⏳ Загрузите файл контура для начала работы")
        file_loaded = False
        filename = None
        tmp_path = None
    else:
        uploaded_file = file_input.value[0]
        filename = uploaded_file.name
        file_data = uploaded_file.contents

        suffix = filename[filename.rfind(".") :] if "." in filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        mo.md(f"✅ Загружен файл: **{filename}**")
        file_loaded = True
    return file_loaded, filename, tmp_path


@app.cell
def _(mo):
    tol_slider = mo.ui.slider(
        0.01, 2.0, step=0.01, value=0.1,
        label="🎯 **Tolerance** (мм)",
    ).form()

    step_slider = mo.ui.slider(
        0.5, 5.0, step=0.5, value=2.0,
        label="📏 **Шаг змейки** (мм)",
    ).form()

    angle_slider = mo.ui.slider(
        0, 90, step=1, value=0,
        label="↗️ **Угол заливки** (°)",
    ).form()

    mode_select = mo.ui.dropdown(
        options=["as_is", "less", "more"],
        value="as_is",
        label="⚙️ **Режим дискретизации**",
    ).form()

    parameter_controls = mo.vstack([
        mo.hstack(
            [tol_slider, step_slider, angle_slider],
            widths="equal",
            gap=1,
        ),
        mo.hstack(
            [mode_select],
            widths="equal",
            gap=1,
        ),
    ])
    parameter_controls
    return (
        angle_slider,
        mode_select,
        step_slider,
        tol_slider,
    )


@app.cell
def _(
    angle_slider,
    mo,
    mode_select,
    step_slider,
    tol_slider,
):
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
    BridgePlanningError,
    ContourError,
    DXFLoadError,
    EmptyFileError,
    InsufficientPointsError,
    InvalidFormatError,
    angle,
    file_loaded,
    filename,
    load_txt,
    mo,
    mode,
    np,
    plan_file_contours,
    step,
    summarize_results,
    tmp_path,
    tol,
):
    plan_results = []
    summary = summarize_results([])
    status_msg = mo.md("⚠️ **Файл не загружен** — загрузите файл выше")

    if file_loaded:
        status_msg = None
        try:
            if filename.lower().endswith(".txt"):
                txt_points = load_txt(tmp_path)
                txt_closed_dist = np.linalg.norm(txt_points[-1] - txt_points[0])
                status_msg = mo.md(
                    f"✅ TXT: **{len(txt_points)}** точек, "
                    f"замкнут: **{'да' if txt_closed_dist < 1e-3 else 'нет'}**"
                )

            plan_results = plan_file_contours(
                tmp_path,
                tolerance=tol,
                line_distance=step,
                fill_angle=angle,
                mode=mode,
                closed_only=True,
            )
            summary = summarize_results(plan_results)

            if filename.lower().endswith(".dxf"):
                dxf_rows = []
                for plan_item in plan_results:
                    item_status = "✅" if not plan_item.skipped else "⏭️"
                    item_reason = (
                        f" ({plan_item.skip_reason})" if plan_item.skipped else ""
                    )
                    item_holes_info = (
                        f", отверстий: {plan_item.hole_count}"
                        if plan_item.hole_count
                        else ""
                    )
                    dxf_rows.append(
                        f"| {item_status} {plan_item.label} | "
                        f"{'да' if plan_item.is_closed else 'нет'} | "
                        f"{plan_item.area:.1f} | "
                        f"{plan_item.stats['total_length']:.1f} | "
                        f"{plan_item.stats['num_lines']} | "
                        f"{plan_item.nesting_depth}{item_holes_info}{item_reason} |"
                    )

                dxf_table = (
                    "| Контур | Замкнут | Площадь мм² | Длина пути мм | Линий | Примечание |\n"
                    "|--------|---------|-------------|---------------|-------|------------|\n"
                    + "\n".join(dxf_rows)
                )
                status_msg = mo.md(
                    f"✅ DXF: найдено **{summary['total_contours']}** контуров, "
                    f"обработано **{summary['processed']}**, "
                    f"пропущено **{summary['skipped']}**\n\n"
                    + dxf_table
                )
            elif plan_results:
                txt_result = plan_results[0]
                status_msg = mo.md(
                    f"### 📊 Статистика\n"
                    f"- Точек в траектории: **{txt_result.stats['total_points']}**\n"
                    f"- Длина пути: **{txt_result.stats['total_length']:.2f}** мм\n"
                    f"- Количество линий: **{txt_result.stats['num_lines']}**"
                )

        except (
            BridgePlanningError,
            ContourError,
            DXFLoadError,
            EmptyFileError,
            InsufficientPointsError,
            InvalidFormatError,
            ValueError,
        ) as exc:
            plan_results = []
            summary = summarize_results([])
            status_msg = mo.md(f"❌ Ошибка обработки: {exc}")
        except Exception as exc:
            plan_results = []
            summary = summarize_results([])
            status_msg = mo.md(f"❌ Неожиданная ошибка: {exc}")

    status_msg
    mo.md(
        f"""
        ### 📊 Итого
        - Контуров: **{summary['total_contours']}**
        - Обработано: **{summary['processed']}**
        - Пропущено: **{summary['skipped']}**
        - Суммарная длина путей: **{summary['total_path_length']:.2f}** мм
        """
    )
    return plan_results, summary


@app.cell
def _(mo, summary):
    trajectory_total_points = int(summary["total_path_points"])
    trajectory_slider_max = max(1, trajectory_total_points)
    if trajectory_total_points > 0:
        trajectory_point_slider = mo.ui.slider(
            1,
            trajectory_slider_max,
            step=1,
            value=trajectory_slider_max,
            label="▶️ Показать траекторию до точки",
        )
    else:
        trajectory_point_slider = None
    return (
        trajectory_point_slider,
        trajectory_slider_max,
        trajectory_total_points,
    )


@app.cell
def _(
    mo,
    trajectory_point_slider,
    trajectory_slider_max,
    trajectory_total_points,
):
    if trajectory_total_points > 0 and trajectory_point_slider is not None:
        trajectory_point_limit = int(
            trajectory_point_slider.value or trajectory_slider_max
        )
        trajectory_slider_view = mo.vstack([
            mo.md(
                f"### ▶️ Просмотр порядка движения\n"
                f"Показаны точки траектории от старта до "
                f"**{trajectory_point_limit} / {trajectory_total_points}**"
            ),
            trajectory_point_slider,
        ])
    else:
        trajectory_point_limit = 0
        trajectory_slider_view = mo.md(
            "### ▶️ Просмотр порядка движения\nТраектория ещё не построена"
        )

    trajectory_slider_view
    return (trajectory_point_limit,)


@app.cell
def _(
    Visualizer,
    mode,
    plan_results,
    step,
    summary,
    tol,
    trajectory_point_limit,
):
    import matplotlib.pyplot as plt

    plot_fig, (plot_ax1, plot_ax2, plot_ax3) = plt.subplots(1, 3, figsize=(18, 6))
    plot_viz = Visualizer()

    if not plan_results:
        plot_ax1.set_title("Контуры (нет данных)")
        plot_ax2.set_title("Траектории (нет данных)")
        plot_ax3.set_title("Сводка (нет данных)")
        for plot_ax in (plot_ax1, plot_ax2, plot_ax3):
            plot_ax.grid(True)
            plot_ax.axis("equal")
    else:
        plot_viz.plot_multiple_results(
            plan_results,
            ax=plot_ax1,
            show_paths=False,
            show_holes=True,
            title="Контур: серый исходный, оранжевый дискретизация",
        )

        plot_viz.plot_multiple_results(
            plan_results,
            ax=plot_ax2,
            show_paths=True,
            show_holes=True,
            path_point_limit=trajectory_point_limit,
            title="Траектория: красная заливка, синяя переходы, точки пути",
        )

        plot_ax3.axis("off")
        summary_lines = [
            "Сводка обработки",
            "",
            f"Всего контуров: {summary['total_contours']}",
            f"Обработано: {summary['processed']}",
            f"Пропущено: {summary['skipped']}",
            f"Точек в путях: {summary['total_path_points']}",
            f"Суммарная длина: {summary['total_path_length']:.1f} мм",
            f"Линий заливки: {summary['total_fill_lines']}",
            f"Показано точек: {trajectory_point_limit}/{summary['total_path_points']}",
            "",
            f"Tolerance: {tol:.3f} мм",
            f"Шаг: {step:.1f} мм",
            f"Режим: {mode}",
        ]
        for skipped_item in plan_results:
            if skipped_item.skipped:
                summary_lines.append(
                    f"⏭ {skipped_item.label}: {skipped_item.skip_reason}"
                )

        plot_ax3.text(
            0.05,
            0.95,
            "\n".join(summary_lines),
            transform=plot_ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

    plot_fig.tight_layout()
    plot_fig
    return


@app.cell
def _(file_loaded, mo, plan_results, summary):
    diagnostics = ["## 🔍 Диагностика состояния", ""]

    if file_loaded:
        diagnostics.append("✅ Файл загружен")
    else:
        diagnostics.append("❌ Файл не загружен")

    if plan_results:
        diagnostics.append(f"✅ Контуров в обработке: {len(plan_results)}")
        diagnostics.append(f"✅ Успешно: {summary['processed']}")
        if summary["skipped"]:
            diagnostics.append(f"⏭ Пропущено: {summary['skipped']}")
    else:
        diagnostics.append("❌ Контуры не обработаны")

    successful_results = [item for item in plan_results if not item.skipped]
    if successful_results:
        diagnostics.append(
            f"✅ Траектории построены: {len(successful_results)} "
            f"({summary['total_path_points']} точек)"
        )
    else:
        diagnostics.append("❌ Траектории не построены")

    mo.md("\n".join(diagnostics))
    return


if __name__ == "__main__":
    app.run()
