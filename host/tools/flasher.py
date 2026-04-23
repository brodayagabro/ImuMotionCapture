#!/usr/bin/env python3
"""
🔧 WT901WIFI UART Flasher (Stub)

CLI-утилита для прошивки датчика через UART.
Пока содержит заглушки — команды и протокол уточняются.

Запуск:
    python -m host.tools.flasher --port COM3 --firmware fw.bin --config ...
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional, List

#  Авто-добавление корня проекта в path (как в main.py)
CURRENT_FILE = Path(__file__).resolve()
for parent in [CURRENT_FILE] + list(CURRENT_FILE.parents):
    if (parent / 'host').is_dir() and (parent / 'firmware').is_dir():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import serial
except ImportError:
    print("❌ Установите pyserial: pip install pyserial")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """Загрузка конфига прошивки."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calc_checksum(data: List[int], method: str = 'xor') -> int:
    """Заглушка под расчёт контрольной суммы."""
    # TODO: реализовать CRC8/CRC16 по спецификации
    if method == 'xor':
        cs = 0
        for b in data:
            cs ^= b
        return cs & 0xFF
    return 0


def send_command(ser: serial.Serial, cmd: List[int], timeout: float = 1.0) -> Optional[bytes]:
    """Отправка команды и чтение ответа (заглушка)."""
    # TODO: добавить framing, ACK/NAK, retry-логику
    cs = calc_checksum(cmd)
    packet = cmd + [cs]
    ser.write(bytes(packet))
    ser.flush()
    time.sleep(0.05)
    if ser.in_waiting > 0:
        return ser.read(ser.in_waiting)
    return None


def enter_bootloader(ser: serial.Serial, cfg: dict) -> bool:
    """Попытка входа в режим прошивки (заглушка)."""
    print("🔑 Вход в bootloader...")
    # TODO: реализовать последовательность входа (пины / команда / перезагрузка)
    sync = cfg.get('bootloader', {}).get('sync_command', [])
    if sync:
        resp = send_command(ser, sync, timeout=2.0)
        if resp:
            print(f"✅ Sync response: {resp.hex()}")
            return True
    print("⚠️ Bootloader entry stub — всегда возвращает True для теста")
    return True


def flash_firmware(ser: serial.Serial, fw_path: str, cfg: dict) -> bool:
    """Прошивка бинарника (заглушка)."""
    fw_data = Path(fw_path).read_bytes()
    print(f"📦 Загрузка прошивки: {len(fw_data)} байт")
    
    # ✅ Валидация размера
    fw_cfg = cfg.get('firmware', {})
    min_sz, max_sz = fw_cfg.get('expected_size_min', 0), fw_cfg.get('expected_size_max', 1024*1024)
    if not (min_sz <= len(fw_data) <= max_sz):
        print(f"❌ Размер прошивки вне диапазона [{min_sz}, {max_sz}]")
        return False
    
    # 🔁 Эмуляция поблочной записи
    packet_size = cfg.get('flash', {}).get('packet_size', 64)
    delay = cfg.get('flash', {}).get('inter_packet_delay_ms', 10) / 1000
    
    for offset in range(0, len(fw_data), packet_size):
        chunk = fw_data[offset:offset + packet_size]
        # TODO: сформировать пакет write_block с адресом и данными
        # send_command(ser, [CMD_WRITE, addr_hi, addr_lo, ...data...])
        time.sleep(delay)  # эмуляция задержки
        if offset % 1024 == 0:
            print(f"⏩ Прогресс: {offset}/{len(fw_data)} байт")
    
    print("✅ Прошивка завершена (заглушка)")
    return True


def main():
    parser = argparse.ArgumentParser(description="🔧 WT901WIFI UART Flasher")
    parser.add_argument('--port', required=True, help='COM-порт, например COM3')
    parser.add_argument('--firmware', required=True, help='Путь к .bin файлу')
    parser.add_argument('--config', default='firmware/sensors/wt901wifi/flash_config.json',
                        help='Путь к JSON-конфигу')
    parser.add_argument('--baud', type=int, default=9600, help='Baud rate UART')
    parser.add_argument('--verify', action='store_true', help='Проверить после записи')
    
    args = parser.parse_args()
    
    # Загрузка конфига
    cfg = load_config(args.config)
    
    print(f"🚀 Flasher started | Port: {args.port} | FW: {args.firmware}")
    
    try:
        with serial.Serial(args.port, args.baud, timeout=2) as ser:
            print(f"🔌 UART открыт: {ser.name} @ {args.baud}")
            time.sleep(1)  # ожидание инициализации
            
            # Вход в bootloader
            if not enter_bootloader(ser, cfg):
                print("❌ Не удалось войти в bootloader")
                return 1
            
            # Прошивка
            if not flash_firmware(ser, args.firmware, cfg):
                return 1
            
            # TODO: опциональная верификация
            if args.verify:
                print("🔍 Верификация (заглушка) — пропущена")
            
            # Перезагрузка в приложение
            jump_cmd = cfg.get('commands', {}).get('jump_to_app', [])
            if jump_cmd:
                send_command(ser, jump_cmd)
                print("🔁 Перезагрузка в приложение...")
            
            print("🎉 Готово!")
            return 0
            
    except serial.SerialException as e:
        print(f"❌ Ошибка UART: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"❌ Файл не найден: {e}")
        return 1
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())