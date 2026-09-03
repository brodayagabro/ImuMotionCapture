# IMU Motion Capture

Система захвата движений верхней части тела на пяти MPU6050, подключённых к
ESP32 через мультиплексор TCA9548A. Контроллер вычисляет кватернионы с помощью
DMP и передаёт согласованные кадры по UDP. На компьютере доступны основной
PyQt-визуализатор, диагностические кубики и демонстрационная Blender-сцена.

Основной поддерживаемый тракт проекта:

```text
MPU6050 ×5 → TCA9548A → ESP32 → UDP → PyQt / Cube Viewer / Blender
```

## Структура проекта

```text
ImuMotionCapture/
├── pyproject.toml                 # зависимости, установка пакетов и pytest
├── sensors/
│   └── esp32/
│       ├── DMP_with_TCA9548A_udp/ # основная UDP-прошивка
│       └── scan_bus_web/          # диагностика I2C и TCA9548A
├── host/
│   ├── viz/
│   │   ├── pyqt_mocap/            # основное приложение захвата движений
│   │   ├── cube_viewer/           # диагностическая визуализация кубиками
│   │   └── blender/               # демонстрационная сцена и её сборщик
│   ├── storage/                   # экспериментальный модуль PostgreSQL
│   └── tools/                     # вспомогательные утилиты
├── tools/
│   └── benchmarks/udp_frequency/  # измерение и анализ частоты UDP
└── docs/                           # дополнительная документация
```

Материалы в `sensors/arduino/` и `sensors/WT901WIFI/` относятся к прежним
прототипам. Текущая рабочая прошивка находится в `sensors/esp32/`.

## Установка Python-приложений

Требуется Python 3.11 или новее. Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Editable-установка добавляет пакеты из `host/viz` в окружение. После неё снова
работает ожидаемая команда:

```bash
python -m pyqt_mocap
```

Также устанавливаются короткие консольные команды:

```bash
pyqt-mocap
cube-viewer
```

На Ubuntu/Debian для GUI могут понадобиться системные пакеты:

```bash
sudo apt install libxcb-cursor0 python3-tk
```

Все Python-настройки сведены в корневой `pyproject.toml`. Дополнительные группы
устанавливаются по необходимости:

```bash
python -m pip install -e ".[dev]"       # pytest и плагины
python -m pip install -e ".[analysis]"  # pandas для анализа частоты
python -m pip install -e ".[db]"        # asyncpg для PostgreSQL
python -m pip install -e ".[tools]"     # pyserial для служебных утилит
```

Для полного окружения разработчика:

```bash
python -m pip install -e ".[dev,analysis,db,tools]"
```

## ESP32 и физический маппинг

Основная прошивка:
[`sensors/esp32/DMP_with_TCA9548A_udp`](sensors/esp32/DMP_with_TCA9548A_udp/README.md).
Перед сборкой создайте локальный файл с параметрами Wi-Fi:

```bash
cd sensors/esp32/DMP_with_TCA9548A_udp
cp wifi_secrets.example.h wifi_secrets.h
```

`wifi_secrets.h` игнорируется Git. В репозитории остаётся только безопасный
пример без имени и пароля локальной сети.

Утверждённый маппинг датчиков:

| Канал TCA / UDP ID | Сегмент | Имя в приложениях |
|---:|---|---|
| 7 | левое плечо / верхняя часть руки | `shoulder.L` |
| 6 | левое предплечье | `forearm.L` |
| 2 | корпус / спина | `spine` |
| 1 | правое предплечье | `forearm.R` |
| 0 | правое плечо / верхняя часть руки | `shoulder.R` |

Для проверки собранной I2C-системы используется отдельная прошивка
[`scan_bus_web`](sensors/esp32/README.md). Она поднимает временную точку доступа
ESP32 и показывает обнаруженные устройства на каждом канале TCA9548A.

## Визуализаторы

### PyQt MoCap

Основное приложение отображает скелетную модель, принимает UDP-кадры,
управляет потоком ESP32 и поддерживает комплексную N → T → «руки вперёд»
калибровку:

```bash
python -m pyqt_mocap
```

Подробности: [`host/viz/pyqt_mocap/README.md`](host/viz/pyqt_mocap/README.md).

### Cube Viewer

Диагностическое Tk/Matplotlib-приложение показывает каждый датчик отдельным
кубом. Оно удобно для проверки UDP, нумерации каналов и направлений вращения:

```bash
python -m cube_viewer
```

Подробности: [`host/viz/cube_viewer/README.md`](host/viz/cube_viewer/README.md).

### Blender

Готовая демонстрационная сцена находится в
`host/viz/blender/udp_receiver/Human_spine_UDP.blend`. Встроенный скрипт
принимает те же пять кватернионов и управляет ригом `Human_Rig`.

Инструкции по запуску, калибровке и воспроизводимой пересборке:
[`host/viz/blender/udp_receiver/README.md`](host/viz/blender/udp_receiver/README.md).

ESP32 передаёт поток последнему зарегистрированному UDP-клиенту. Поэтому PyQt,
Cube Viewer и Blender следует подключать к контроллеру по очереди.

## Модуль БД

`host/storage/db_client.py` содержит асинхронный клиент PostgreSQL/asyncpg и
создание таблицы `imu_stream`. Сейчас это экспериментальная заготовка:
подключение и схема таблицы реализованы, но `insert_batch()` ещё не записывает
данные. Для продолжения разработки установите группу `db`:

```bash
python -m pip install -e ".[db]"
```

## Тесты

Основной набор не требует контроллера или дисплея:

```bash
python -m pip install -e ".[dev]"
pytest
```

GUI smoke-тест PyQt запускается отдельно:

```bash
QT_QPA_PLATFORM=offscreen pytest -m gui \
  host/viz/pyqt_mocap/tests/test_gui_udp_smoke.py
```

Headless-проверка Blender пересобирает проект во временный файл и тестирует
риг, встроенный скрипт, UDP-парсер и команды контроллеру:

```bash
./host/viz/blender/udp_receiver/test.sh
```

## Измерение частоты UDP

Аппаратный бенчмарк и анализ результатов находятся в
`tools/benchmarks/udp_frequency/`:

```bash
python -m pip install -e ".[analysis]"
python tools/benchmarks/udp_frequency/test_frequency.py IP_АДРЕС_ESP32
python tools/benchmarks/udp_frequency/analyze_frequency.py
```

Скрипт измерения последовательно запрашивает частоты, сохраняет времена прихода
кадров в CSV, а анализатор рассчитывает фактическую частоту и строит график.

## Документация

- [прошивки ESP32](sensors/esp32/README.md);
- [протокол основной прошивки](sensors/esp32/DMP_with_TCA9548A_udp/README.md);
- [визуализаторы](host/viz/README.md);
- [PyQt-приложение и калибровка](host/viz/pyqt_mocap/README.md);
- [Cube Viewer](host/viz/cube_viewer/README.md);
- [Blender-демонстрация](host/viz/blender/udp_receiver/README.md).
