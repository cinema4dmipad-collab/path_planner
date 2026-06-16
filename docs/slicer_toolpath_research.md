# Как open source слайсеры строят траектории

Краткий обзор по CuraEngine, PrusaSlicer/libslic3r и OrcaSlicer с выводами для нашего `path_planner`.

## Главная идея

Открытые FDM-слайсеры обычно не строят всю траекторию слоя как одну глобальную оптимизационную задачу. Вместо этого они:

1. режут модель на 2D-слои;
2. превращают слой в полигоны/острова с отверстиями;
3. делят область на типы траекторий: внешние стенки, внутренние стенки, gap fill, skin, infill, support;
4. генерируют геометрию каждого типа отдельно;
5. локально оптимизируют порядок линий/контуров;
6. отдельно планируют travel-переходы между печатными участками.

Это важно: заполнение и переходы между участками чаще считаются разными задачами. Для нашей задачи с внутренними контурами это означает, что "змейка" должна строить печатные сегменты внутри разрешённой области, а соединяющие переходы должны иметь свой алгоритм обхода отверстий.

## CuraEngine

Источники:

- [Generating Paths, CuraEngine Wiki](https://github.com/Ultimaker/CuraEngine/wiki/Generating-Paths)
- [Internals, CuraEngine Wiki](https://github.com/Ultimaker/CuraEngine/wiki/Internals)
- [`PathOrderOptimizer.h`](https://github.com/Ultimaker/CuraEngine/blob/45a71e41/include/PathOrderOptimizer.h)

Ключевые решения:

- Слой превращается в `LayerParts`: отдельные области слоя. При построении `LayerParts` используется Clipper; результатом могут быть полигоны с holes.
- Insets/perimeters строятся через offset-операции. В документации прямо сказано, что Clipper делает основную геометрическую работу.
- Infill генерируется позже, уже внутри рассчитанных областей. Для line infill область пересекается набором scan lines; между парами пересечений создаются отрезки заполнения.
- Порядок печати жёстко иерархичен: mesh group -> layer -> extruder plan -> mesh -> part -> feature type. Внутри part порядок feature type обычно: infill, inner walls, outer wall, gaps, skin и т.д., но часть порядка настраивается.
- Для группы линий используется приближённая оптимизация порядка, а не точное TSP. Cura описывает nearest-neighbour: после печати линии выбирается ближайший конец ещё не напечатанной линии.
- Travel-переходы планируются отдельной системой `Comb`. Combing старается двигаться внутри модели и не пересекать наружные стенки.
- Collision avoidance для travel: препятствия offset-ятся на безопасное расстояние, прямой travel проверяется на пересечение, затем выбирается обход вдоль стороны с меньшим количеством вершин. Это эвристика, а не точный shortest path.

Что полезно для нас:

- Отверстия надо хранить как часть области заполнения, а не как отдельные самостоятельные контуры для заливки.
- Прямой переход между концами infill-сегментов нельзя считать безопасным только потому, что обе точки валидны. Нужно проверять весь сегмент на пересечение с holes.
- Если прямой переход пересекает hole, переход должен строиться как travel/bridge around obstacle: через offset-обход, по верхней/нижней стороне отверстия или по графу видимости.

## PrusaSlicer / libslic3r

Источники:

- [`PerimeterGenerator.cpp`](https://github.com/prusa3d/PrusaSlicer/blob/69c8e569/src/libslic3r/PerimeterGenerator.cpp)
- [`Fill.cpp`](https://github.com/prusa3d/PrusaSlicer/blob/69c8e569/src/libslic3r/Fill/Fill.cpp)
- [`FillPlanePath.cpp`](https://github.com/prusa3d/PrusaSlicer/blob/69c8e569/src/libslic3r/Fill/FillPlanePath.cpp)
- [`ShortestPath.cpp`](https://github.com/prusa3d/PrusaSlicer/blob/master/src/libslic3r/ShortestPath.cpp)

Ключевые решения:

- Геометрия представлена через `ExPolygon`: внешний контур плюс holes.
- Perimeters строятся в `PerimeterGenerator`; поддерживаются classic и Arachne-подходы.
- Infill строится внутри `Fill`-подсистемы. Паттерны получают область, угол, spacing, flow, density и возвращают `Polylines` / extrusion entities.
- Для плоских line-like паттернов область поворачивается на угол заполнения, строится bounding box, генерируются линии, затем результат возвращается в исходную систему координат.
- Для порядка путей используется `ShortestPath`. В коде прямо описаны:
  - naive TSP через closest neighbour;
  - greedy chaining для периметров и open/closed fills;
  - возможность разворачивать open segments, если это уменьшает переходы.
- В `ShortestPath.cpp` задача формулируется как TSP-like, но решается greedy/multi-fragment эвристиками. Это практичный компромисс между качеством пути и скоростью слайсинга.

Что полезно для нас:

- Для open infill segments стоит разрешать разворот сегмента при соединении. Это может уменьшить длину bridge/travel и снизить число пересечений с отверстиями.
- Для набора сегментов на одной строке "змейка" хороша как локальное правило, но при holes лучше думать о сегментах как о графе: у каждого сегмента есть два конца, между концами есть стоимость безопасного перехода.
- Стоимость перехода должна учитывать не только евклидово расстояние, но и валидность: пересечение отверстия, движение вдоль запрещённой границы, лишний выход из рабочей области.

## OrcaSlicer

Источники:

- [`PerimeterGenerator.cpp`, OrcaSlicer](https://github.com/SoftFever/OrcaSlicer/blob/7ec32fc4/src/libslic3r/PerimeterGenerator.cpp)
- [Infill Generation, DeepWiki по OrcaSlicer](https://deepwiki.com/SoftFever/OrcaSlicer/4.4-infill-generation)
- [`FillBase.hpp`, OrcaSlicer](https://github.com/SoftFever/OrcaSlicer/blob/7ec32fc4/src/libslic3r/Fill/FillBase.hpp)

Ключевые решения:

- OrcaSlicer унаследовал большую часть архитектуры от PrusaSlicer/libslic3r.
- Infill pipeline похож: `Layer::make_fills()` группирует поверхности по совместимым параметрам, создаёт `Fill`-класс нужного паттерна и генерирует extrusion paths.
- После генерации infill выполняется оптимизация:
  - reorder/chaining для уменьшения travel distance;
  - упрощение полилиний;
  - greedy nearest-neighbour для выбора следующего пути.
- В обсуждениях по OrcaSlicer видно, что качество sparse infill path optimisation остаётся практической проблемой: разные варианты `connect_infill()` и `ShortestPath::chain_polylines()` дают разные компромиссы, а для rectilinear infill используют свойства паттерна.

Что полезно для нас:

- Для rectilinear/scanline filling можно использовать специализированную логику, а не общий TSP.
- Но при отверстиях одной только сортировки по Y недостаточно: соединение соседних отрезков должно знать топологию препятствий.

## Общие стратегии оптимизации

Источник:

- [Fundamental Path Optimization Strategies for Extrusion-based Additive Manufacturing, ORNL/SFF 2024](https://www.osti.gov/servlets/purl/2498425)

Ключевые идеи из обзора:

- В промышленных и desktop-слайсерах редко используются тяжёлые методы вроде полного TSP, Chinese Postman Problem, ant colony и т.п.
- На практике чаще применяются простые автоматические стратегии:
  - outside-in;
  - inside-out;
  - next closest;
  - next farthest;
  - random;
  - custom point / seam point.
- Оптимизация делится на уровни:
  - порядок islands;
  - порядок paths внутри region;
  - выбор стартовой точки/направления внутри path;
  - travel insertion между paths.

## Практические выводы для `path_planner`

### 1. Разделить печатные сегменты и переходы

Заполнение должно отвечать только за генерацию валидных отрезков внутри области:

- outer contour задаёт разрешённую область;
- inner contours становятся holes;
- на каждом scanline пересечения с holes вычитаются из интервала заполнения.

Соединение сегментов должно быть отдельной задачей. Переход должен проверяться на:

- пересечение interior любого hole;
- движение вдоль запрещённой границы hole;
- выход за outer contour;
- лишние резкие detour, если есть более короткий безопасный переход.

### 2. Для holes использовать obstacle-aware bridge

Минимальная практичная схема:

1. Если прямой переход безопасен, использовать его.
2. Если переход пересекает hole, построить кандидаты обхода:
   - выше hole: `ymin - offset`;
   - ниже hole: `ymax + offset`;
   - через ближайшие допустимые углы/вершины offset-контура.
3. Проверить каждый кандидат на пересечение holes.
4. Выбрать минимальный по стоимости.

Стоимость:

```text
cost = length
     + penalty_for_large_y_detour
     + penalty_for_crossing_many_scanlines
     + penalty_for_getting_close_to_hole_edge
```

Важно: допуск для "на границе отверстия" должен быть маленьким геометрическим epsilon, а не пользовательским `tolerance` загрузки/дискретизации. Иначе безопасный обход около отверстия может ошибочно считаться движением по границе.

### 3. Для сложных holes перейти от эвристики Y-level к графу видимости

Если появятся произвольные отверстия, не только прямоугольные/почти прямоугольные, лучше строить граф:

- узлы: start, end, вершины offset-контуров отверстий, возможно точки касания scanline;
- рёбра: прямые отрезки, которые не пересекают holes и лежат внутри outer contour;
- вес: длина + технологические штрафы;
- поиск: Dijkstra или A*.

Это близко к тому, что Cura делает для collision avoidance, только в более явной форме.

### 4. Порядок сегментов можно улучшить без полного TSP

Текущая змейка полезна и предсказуема, но можно добавить локальную оптимизацию:

- внутри одной строки сохранять snake-order;
- при наличии нескольких сегментов из-за holes выбирать следующий сегмент по минимальному безопасному bridge-cost;
- разрешить разворот open segment, если это уменьшает стоимость перехода;
- между строками выбирать сторону перехода с учётом ближайшего безопасного конца следующей строки.

Это повторяет подход PrusaSlicer/libslic3r: не решать полный TSP, но учитывать endpoints, reversing и greedy chaining.

### 5. Логи должны показывать не только точки, но и причину выбора

Для отладки внутренних контуров полезно логировать:

- `scanline y`;
- интервалы fill до/после вычитания holes;
- start/end каждого сегмента;
- direct bridge accepted/rejected;
- если rejected: какой hole пересечён;
- список кандидатов обхода и их cost;
- выбранный route.

Пример:

```text
y=5.00 intervals before holes: [(0.00, 10.00)]
y=5.00 hole #1 interval: (3.00, 7.00)
y=5.00 fill segments: [(0.00, 3.00), (7.00, 10.00)]
bridge (3.00, 5.00) -> (7.00, 5.00): direct rejected, crosses hole #1
candidate route_y=2.90 rejected/accepted cost=...
candidate route_y=7.10 accepted cost=...
selected route_y=7.10
```

## Рекомендуемая архитектура для нашей задачи

```text
Contour + holes
    -> normalize / rotate
    -> scanline intersection
    -> fill segments per y
    -> segment graph / snake rows
    -> obstacle-aware bridge planner
    -> connected trajectory
    -> inverse rotate
```

Где:

- `fill_segments_at_y()` должен быть чистой геометрической операцией вычитания holes;
- `_connect_lines()` должен только выбирать порядок сегментов;
- `_bridge_points()` должен быть единственным местом, где строится безопасный переход между концами;
- проверка `segment_crosses_hole` должна использовать малый epsilon, не общий `tolerance`.

## Короткий вывод

Open source слайсеры решают похожую задачу не через "одну идеальную змейку", а через pipeline:

- geometry first: полигоны, holes, offsets, scanlines;
- local path generation: infill/perimeters по типам;
- local ordering: nearest-neighbour, greedy chaining, endpoint reversing;
- safe travel: combing/collision avoidance вокруг препятствий.

Для нашего случая с внутренними контурами самый важный урок: дырка должна быть препятствием и на этапе генерации fill-сегментов, и на этапе соединения сегментов. Даже если сами отрезки заполнения корректны, траектория ломается, если bridge/travel между ними не obstacle-aware.
