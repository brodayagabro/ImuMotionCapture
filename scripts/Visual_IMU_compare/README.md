# Оптический захват движений: развёртывание по роли компьютера

| Компьютер | Каталог | Назначение |
|---|---|---|
| Компьютер/одноплатник с камерой | [camera/](camera/README.md) | V4L2 → H.264 → RTSP |
| Хост-сервер | [server/](server/README.md) | Приём RTSP, RTMPose, 3D, WebGL, BVH |

```text
Камера 1 → camera_streamer.py ─┐
Камера 2 → camera_streamer.py ─┼→ MediaMTX → visual_mocap_server.py → WebGL / BVH
Камера 3 → camera_streamer.py ─┘   rtsp_relay.py     RTMPose + FreeMoCap DLT
```

На каждом компьютере нужен **только этот репозиторий**. Установщики сами
получают пакеты, исходники, модели и бинарники. Пути к ранее склонированному
`MotionCapture/equipment/VisualMoCap` не нужны. Первое развёртывание требует
интернета. Серверу нужен полный клон ImuMotionCapture: он использует также
`host/storage/bvh.py` из этого же репозитория.

## Клонирование

```bash
git clone https://github.com/brodayagabro/ImuMotionCapture.git
cd ImuMotionCapture/scripts/Visual_IMU_compare
```

Далее выбрать роль. `--system-deps` устанавливает системные зависимости
через apt/sudo. Если они уже есть, этот флаг можно опустить.
Не запускайте весь setup через sudo: повышение прав используется только для apt.

## Компьютер с камерой

```bash
cd camera
./setup.sh --system-deps
source .venv/bin/activate
python camera_streamer.py --list-devices
python camera_streamer.py \
  --server 192.168.1.10 --name cam1 \
  --resolution 1280x720 --fps 30
```

Заменить IP адресом хост-сервера. На остальных компьютерах использовать
`cam2`, `cam3`. Если камер несколько, добавить `--device /dev/videoN`;
единственная камера выбирается автоматически.

Создаётся `camera/.venv`, проверяются FFmpeg/V4L2. Нейросети и CUDA
на компьютере с камерой не устанавливаются.
[Подробная инструкция камеры](camera/README.md).

## Хост-сервер

Для NVIDIA GPU с подходящим драйвером:

```bash
cd server
./setup.sh --system-deps --device cuda
source .venv-cuda/bin/activate
python rtsp_relay.py
```

Оставить ретранслятор работающим. Во втором терминале, также из `server/`:

```bash
source .venv-cuda/bin/activate
python visual_mocap_server.py \
  --device cuda:0 --host 0.0.0.0 --port 8080 \
  --camera 'cam1=rtsp://127.0.0.1:8554/cam1' \
  --camera 'cam2=rtsp://127.0.0.1:8554/cam2' \
  --bvh-target 192.168.1.20:7001
```

Открыть **http://IP_СЕРВЕРА:8080**. Указывать только запущенные камеры;
для третьей добавить `--camera cam3=rtsp://127.0.0.1:8554/cam3`.
`--bvh-target` задаёт адрес принимающего хоста; параметр можно опустить.

Для CPU вместо CUDA, из `server/`:

```bash
./setup.sh --device cpu
source .venv-cpu/bin/activate
python visual_mocap_server.py --demo --device cpu
```

`--demo` показывает синтетические камеры. Для реального захвата заменить
его параметрами `--camera`. Метрический 3D требует калибровки минимум двух
камер через веб-интерфейс.

Серверный setup создаёт:

- `.venv-cpu/` или `.venv-cuda/` с закреплёнными Python-пакетами;
- `repositories/rtmlib` и `repositories/freemocap` на закреплённых коммитах;
- `.tools/bin/mediamtx` с проверкой SHA-256;
- `.cache/rtmlib/` с готовыми моделями YOLOX-m и RTMPose-m.

Если системный Python слишком старый, setup автоматически получает приватный
Python 3.12 через закреплённый uv. Он остаётся внутри папки выбранной роли.
Драйвер NVIDIA установщик не меняет. Закреплённый ONNX Runtime GPU 1.31.1
использует CUDA 13.x: драйвер должен быть совместим. Jetson требует
согласования с JetPack. [Подробная инструкция сервера](server/README.md).

## Повторная установка и Git

Setup можно запускать повторно: проверяются окружения, пакеты и исходники.
Пользовательские изменения в локальных клонах не сбрасываются; при
несовпадении setup останавливается с диагностикой. Камеры и постоянные
сервисы во время установки не запускаются.
`server/setup.sh --skip-models` откладывает загрузку моделей до первого запуска.
`--with-tests` добавляет зависимости для тестов.

В Git хранятся **исходники, setup.sh, requirements-файлы, тесты и документация**.
Окружения `.venv*`/`.env*`, `.tools`, `.cache`, `repositories` и `runtime`
исключены через `.gitignore`. Фактические версии установленных пакетов
сохраняются локально в `runtime/requirements-resolved*.txt`.

Прежнее `.venv-visual-cpu` в корне эксперимента сохранено как локальная
резервная установка. Новые команды его не используют. Окружения не копируются
на другие компьютеры: на каждом клоне запускается setup.

## Материалы исследования

- [Сравнение решений](COMPARISON.md).
- [Публикации и JCR](SOURCES.md).
- [Исторический аудит conda freemocap-env](FREEMOCAP_ENV.md).

RTSP-синхронизация сейчас основана на времени приёма после декодирования.
Перед сравнением точности с IMU нужно проверить синхронизацию экспозиций
и задержки физических камер.
