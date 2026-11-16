# Камера: локальная запись и HTTP-управление

Основной сценарий: [README эксперимента](../README.md).
`camera_streamer.py` теперь записывает локально по команде сервера. Веб-страницы
на камере нет; API отдаёт маленькие JPEG-превью и завершённые файлы.

## Установка

```bash
./setup.sh --system-deps --clean-legacy
source .venv/bin/activate
python camera_streamer.py --list-devices
python camera_streamer.py --device /dev/video0 --list-formats
python camera_streamer.py --list-audio
python camera_streamer.py --name cam1 --device /dev/video0 \
  --audio plughw:CARD=Camera,DEV=0 --resolution max --port 8081
```

Требования: Linux, Python ≥3.10, FFmpeg **5.1+** с входами V4L2 и ALSA,
ffprobe, v4l-utils, alsa-utils. Системный FFmpeg Ubuntu 22.04 (4.4) слишком стар
для `-isync`: используйте более новый FFmpeg, доступный в `PATH` (например,
Ubuntu 24.04 / Debian 12 и новее). Установщик проверяет наличие нужной опции.
На современных Orange Pi системный Python 3.14 подходит. `sudo` нужен только
для `--system-deps`. Камера не требует Python-пакетов, pip/uv/SkellyCam/OpenCV.
В `.venv` создаётся окружение стандартной библиотеки; можно запускать и просто
`python3 camera_streamer.py ...` при наличии системных инструментов.

Вместо номера видео используйте `/dev/v4l/by-id/...-video-index0`, вместо номера
ALSA — `plughw:CARD=<имя>,DEV=0`: они устойчивее к переподключению.
Если нет прав на устройства, добавьте пользователя в группы `video,audio`
и войдите в систему повторно. Не запускайте узел от root.

## Параметры записи

| Ключ | Поведение |
|---|---|
| `--resolution max` | Наибольшее разрешение MJPEG/H.264, затем максимальный FPS |
| `--resolution 1920x1080 --fps 30` | Точный поддерживаемый режим; ошибка, если его нет |
| `--resolution max --fps 30` | Наибольшее разрешение среди режимов 30 FPS |
| `--input-format mjpeg` / `h264` | Выбор формата камеры; по умолчанию `auto` |
| `--audio plughw:CARD=Camera,DEV=0` | Микрофон ALSA; по умолчанию `default` |
| `--records /data/camera-recordings` | Каталог локальных записей |
| `--preview-width 320 --preview-fps 1` | Размер и частота превью; можно уменьшить |
| `--min-free-mb 256` | Резерв свободного места, при достижении запись останавливается |
| `--ffmpeg /path/ffmpeg --ffprobe /path/ffprobe` | Явные исполняемые файлы |
| `--token-file /path/shared-token` | Необязательный общий секрет для HTTP API |

Таблица режимов конкретной камеры: `--device /dev/video0 --list-formats`.
Универсальной таблицы разрешений/FPS нет: она зависит от устройства и формата.
Система выбирает максимум **сжатых** режимов, чтобы не кодировать полноразмерное
видео на CPU одноплатника. Камеры только с raw/YUYV в этом варианте не поддержаны.

Во время записи исходные MJPEG/H.264-пакеты копируются в MKV; PCM-звук пишется
в тот же файл. FFmpeg связывает начало ALSA и V4L2 по общему времени (`-isync 0`,
V4L2 `-timestamps abs`) и дополняет начальный зазор аудио тишиной.
Для MJPEG декодируется только редкий кадр по запросу превью, для H.264 — опорные
кадры. Видео не передаётся непрерывно в сеть. Без зрителя камера в ожидании
освобождается примерно через 5 секунд; при начале записи переключается в полный режим.

При потере камеры/места запись помечается ошибкой; частичный файл сохраняется.
После перезапуска незавершённую запись можно попробовать завершить командой
выгрузки с сервера. Сохранность видео при обрыве питания не гарантируется.

## Автозапуск systemd

Остановите прежнюю службу (для cam2 имя обычно `visual-mocap-camera-cam2.service`).
Проверьте его через `systemctl list-unit-files '*camera*'`.
Замените её `ExecStart` новым вызовом либо перезапишите unit:

```bash
sudo systemctl stop visual-mocap-camera-cam1.service
sudo nano /etc/systemd/system/visual-mocap-camera-cam1.service
```

Пример для пользователя `camera`; замените пути, имя камеры и ALSA-устройство:

```ini
[Unit]
Description=Local motion-capture camera recorder
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=camera
SupplementaryGroups=video audio
WorkingDirectory=/home/camera/MoCap/ImuMotionCapture/scripts/Visual_IMU_compare/camera
ExecStart=/usr/bin/python3 /home/camera/MoCap/ImuMotionCapture/scripts/Visual_IMU_compare/camera/camera_streamer.py --name cam1 --device /dev/v4l/by-id/REPLACE-video-index0 --audio plughw:CARD=REPLACE,DEV=0 --resolution max --port 8081
Restart=on-failure
RestartSec=5
TimeoutStopSec=90
KillSignal=SIGTERM
KillMode=mixed

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now visual-mocap-camera-cam1.service
journalctl -u visual-mocap-camera-cam1.service -f
```

Установщик не создаёт и не перезапускает службы автоматически.
В UFW камеры разрешите управление только серверу:

```bash
sudo ufw allow from 192.168.1.90 to any port 8081 proto tcp
```

## HTTP API

Все POST имеют `Content-Type: application/json`. При `--token-file` каждый
запрос требует `Authorization: Bearer <token>` (тот же файл на сервере).

| Запрос | Назначение |
|---|---|
| `GET /api/status` | Режим, состояние, свободное место |
| `GET /preview.jpg` | Редкий JPEG, 503 пока нет кадра |
| `POST /api/start` | `{"id":"32 hex","name":"take_01","kind":"record"}` |
| `POST /api/stop` | `{"id":"32 hex"}`, асинхронное завершение |
| `GET /api/recordings/<id>` | Манифест, состояние, размер и SHA-256 |
| `GET /files/<id>` | Только завершённый MKV; `Range: bytes=N-` для докачки |

Имена типов: `record` и `calibration`. Повторный start с тем же ID и данными
не создаёт второй файл. Камера обслуживает одну запись за раз.
