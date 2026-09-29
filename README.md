# NVIDIA GEM-X → Epic Fight (Blender)

Транслятор видео-мокапа **NVIDIA GEM-X** на риг **Epic Fight** в Blender.

GEM-X ([NVlabs/GEM-X](https://github.com/NVlabs/GEM-X)) восстанавливает по обычному
видео движение тела SOMA (77 суставов). Этот репозиторий — аддон **Epic Fight / VIX
0.25.0** (`ef_blender/`), в который добавлен импорт результата GEM-X: одна команда —
и снятое на телефон движение играет на бипеде Epic Fight (Steve/Alex) с контроллерами,
IK и экспортом в `.json` для мода.

```
видео.mp4 ──GEM-X──▶ hpe_results.pt ──Blender: File ▸ Import ▸ NVIDIA GEM-X Motion──▶ риг Epic Fight
                                     └─tools/gemx_to_ef.py (без Blender)──────────────▶ анимация .json для игры
```

## Установка

1. Скачайте `ef_blender-0.25.0.zip` (или соберите: `blender --command extension build
   --source-dir ef_blender --output-dir dist`).
2. Blender 4.2+ → *Edit ▸ Preferences ▸ Get Extensions ▸ ⌄ ▸ Install from Disk…* → zip.

Аддону не нужны torch, numpy или GPU: файл `.pt` читается встроенным безопасным
ридером (см. ниже).

## Шаг 1 — снять движение в GEM-X

```bash
python scripts/demo/demo_soma.py --video myvideo.mp4        # камера двигается
python scripts/demo/demo_soma.py --video myvideo.mp4 -s     # камера на штативе
```

Результат: `outputs/demo_soma/myvideo/hpe_results.pt`. Подходит и ускоренный
`demo_soma_onnx.py`. Советы по съёмке: человек целиком в кадре, без сильных
перекрытий, один человек в кадре.

## Шаг 2 — импорт в Blender

*File ▸ Import ▸ NVIDIA GEM-X Motion to Epic Fight (.pt, .npz, .bvh)* — или
N-панель *Epic Fight ▸ Build ▸ Import GEM-X Motion*. Выберите `hpe_results.pt`.

> **Video rate** — укажите частоту кадров **исходного** видео (30, 60…).
> `hpe_results.pt` её не хранит, а копия `.mp4`, которую демо кладёт рядом,
> всегда перекодирована в 30 fps. Ошибка здесь = ускоренное/замедленное движение.

Что происходит:

* если в сцене нет рига — создаётся тот же риг, что делает *Generate* (контроллеры,
  тело, скин); если риг есть — анимация ложится на него, а прежний клип
  паркуется на заглушённый NLA-трек;
* ключи ложатся на FK-контроллеры, затем IK-ручки запекаются по FK-результату —
  любую конечность можно переключить на IK в любом кадре, поза не сдвинется
  (проверено: ≤ 0.1 мм);
* если захват выходит за упоры рига (пресет *Epic Fight* — это ровно диапазон
  штатных клипов мода, а человек, например, сгибает локоть больше 140°), импорт
  выключает у рига *Joint Limits* и пишет об этом предупреждение — иначе экспорт
  молча обрезал бы движение. Опция *Keep joint limits* — обрезать намеренно;
* action получает имя видео; параметры импорта сохраняются в нём (`efb_gemx`).

### Параметры импорта

| Группа | Параметр | По умолчанию | Смысл |
|---|---|---|---|
| Source | Video rate | 30 | fps исходного видео (для BVH берётся из файла) |
| | Clip rate | 0 | fps клипа; 0 = как у видео (до 60) |
| | First / Last frame | 0 / −1 | диапазон кадров видео |
| | Smoothing | 1.0 | гауссово сглаживание, σ в кадрах; 0 — выкл |
| Placement | Root motion | Full path | *Full path* — персонаж идёт по траектории; *In place* — только высота (для зацикленной локомоции) |
| | Face forward | вкл | развернуть так, чтобы в первом кадре смотрел вперёд |
| | Start at origin | вкл | начать в начале координат |
| | Feet on the floor | вкл | опустить так, чтобы ступни бипеда стояли на полу |
| | Person's height | 0 | рост человека в м; 0 — оценить по клипу |
| Body | Hinge elbows and knees | вкл | локти/колени — чистые шарниры, как у рига (см. ниже) |
| | Wrist to Tool | вкл | поворот кисти → сокет `Tool_R/L` (предмет в руке следует за кистью) |
| | Collarbones | 1.0 | доля движения ключиц |
| | Keep joint limits | выкл | не выключать упоры рига |
| Rig | Build / Body Mesh / Bake IK | Wide / вкл / вкл | как у обычного импорта Epic Fight |

## Шаг 3 — в игру

*File ▸ Export ▸ Epic Fight Animation (.json)* — штатный экспорт аддона.

Или без Blender вовсе (обычный python 3.8+, без зависимостей):

```bash
python tools/gemx_to_ef.py outputs/demo_soma/myvideo/hpe_results.pt --fps-in 30
# -> outputs/demo_soma/myvideo/myvideo_epicfight.json
```

Полезные ключи: `--in-place`, `--slim` (Alex), `--height 1.80`, `--start/--end`,
`--fps 60`, `--smooth 0`, `--no-hinge`, `--no-tools`, `--describe` (показать, что
лежит в файле). CLI пишет все 20 суставов, включая маркеры `Elbow_*/Knee_*` со
сдвигом шва, как в штатных клипах. Результат совпадает с экспортом из Blender
(проверено: расхождение ≤ 0.0001°).

## Другие источники SOMA

* `.bvh` со скелетом SOMA — BONES-SEED, NVIDIA Kimodo (`--bvh`), экспорт
  soma-retargeter;
* `.npz` / `.pt` с ключами `global_orient`, `body_pose`, `transl` (или `poses`).

## Как это работает

* **Чтение `.pt` без torch** (`efb/tensorio.py`). `torch.save` — это zip: pickle
  с описанием + сырые массивы. Pickle не исполняется «как есть»: разрешён только
  фиксированный список имён (контейнеры, сборка тензоров torch, массивы numpy),
  всё остальное превращается в инертную заглушку. Открытие файла не может
  выполнить код.
* **SOMA** (`efb/soma.py`, `efb/data/soma77.json`). Позы GEM-X — в стиле SMPL,
  относительно T-позы: SOMA собирает сустав как `orient[p]⁻¹·pose[j]·orient[j]`,
  ориентиры сокращаются, и мировой поворот сустава от T-позы —
  `G[j] = G[parent]·R(pose[j])`. Больше ничего (модель тела, меш, torch) не нужно.
  Таблица T-позы сгенерирована `tools/make_soma_table.py` из ассетов NVIDIA.
* **Ретаргет** (`efb/gemx.py`). Оба скелета переводятся в пространство арматуры
  (Z вверх, бипед смотрит на +Y). Кость Epic Fight копирует поворот «своего»
  сустава SOMA: `pose = G · align · rest`, где `align` для рук/ног разворачивает
  опущенные руки бипеда в T-позу SOMA. Колени и локти решаются как шарниры:
  верхняя кость поворачивается вокруг своей оси, пока шарнир не встанет по
  нормали к плоскости конечности, нижняя сгибается только по оси X (как требует
  пин рига). Пронация предплечья при этом не теряется — она уходит в `Tool_*`.
  Траектория таза масштабируется отношением длины ног (0.761 м у бипеда против
  ~0.93 у человека), клип опускается на пол по ступням самого бипеда.
* Результат — обычный документ анимации Epic Fight (формат `attributes`), который
  идёт через штатный импорт аддона.

## Ограничения

* У бипеда Epic Fight нет пальцев, лица и стоп — эти суставы SOMA не переносятся.
* Форма тела (identity/scale) не используется — только движение скелета.
* `hpe_results.pt` не хранит fps — его задаёте вы.
* ONNX-демо GEM-X не «приземляет» клип; тогда рост не оценивается (берётся
  средний 1.77 м, задайте *Person's height*), а пол находится по ступням бипеда.
  Если там подставлены координаты камеры (Y вниз), клип переворачивается
  автоматически.

## Проверки

```bash
python3 -m unittest discover -s tests -v          # 21 тест, без Blender и torch
blender -b --factory-startup --python-exit-code 1 \
        --python tests/blender_gemx.py -- tests/data/squat_hpe_results.pt
```

Blender-гейт (4.5 LTS): импорт на настоящий риг; поза deform-костей = ретаргет
(≤ 6·10⁻⁶ м) на каждом кадре; ступни на полу; IK-переключение не двигает позу;
штатный экспорт = CLI-json. Проверено на реальных клипах SOMA (ходьба, прыжок,
танец, удар ногой, присед).

## Лицензии

* Аддон Epic Fight / VIX — MIT (epic-port), см. `ef_blender/blender_manifest.toml`.
* `efb/data/soma77.json` получен из NVIDIA SOMA-X (`SOMA_neutral.npz`) и
  soma-retargeter (`soma_zero_frame0.bvh`), Apache-2.0.
* `tests/data/squat_hpe_results.pt` — кадры 600–659 `assets/example_animation.npy`
  из SOMA-X (Apache-2.0), см. `tools/make_test_fixture.py`.
* Сама модель GEM-X сюда не входит; её использование регулирует
  NVIDIA Open Model License.
