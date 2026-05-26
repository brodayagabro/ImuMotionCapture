import sys
import numpy as np
import imufusion
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import socket
import threading
import csv
from datetime import datetime
import time
from collections import deque

# ==============================
# Классы для работы с кватернионами и визуализации
# ==============================

class Quaternion:
    """Класс для работы с кватернионами"""
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def __mul__(self, other):
        """Умножение кватернионов"""
        w = self.w*other.w - self.x*other.x - self.y*other.y - self.z*other.z
        x = self.w*other.x + self.x*other.w + self.y*other.z - self.z*other.y
        y = self.w*other.y - self.x*other.z + self.y*other.w + self.z*other.x
        z = self.w*other.z + self.x*other.y - self.y*other.x + self.z*other.w
        return Quaternion(w, x, y, z)
    
    def conjugate(self):
        """Сопряженный кватернион"""
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    def rotate_point(self, point):
        """Вращение точки с помощью кватерниона"""
        # Создаем кватернион для точки (вектора)
        p = Quaternion(0, point[0], point[1], point[2])
        
        # Выполняем вращение: q * p * q_conj
        rotated = self * p * self.conjugate()
        
        return np.array([rotated.x, rotated.y, rotated.z])

class Cube3D:
    """Класс для создания и управления 3D-кубом"""
    def __init__(self, center=(0, 0, 0), size=1):
        self.center = np.array(center)
        self.size = size
        
        # Создаем начальные вершины куба
        self.initial_vertices = self._create_initial_vertices()
        self.vertices = self.initial_vertices.copy()
        
        # Определяем грани куба (индексы вершин)
        self.faces = [
            [0, 1, 2, 3],  # задняя
            [4, 5, 6, 7],  # передняя
            [0, 1, 5, 4],  # нижняя
            [2, 3, 7, 6],  # верхняя
            [0, 3, 7, 4],  # левая
            [1, 2, 6, 5]   # правая
        ]
        
        # Цвета для каждой грани
        self.face_colors = cm.viridis(np.linspace(0, 1, 6))
    
    def _create_initial_vertices(self):
        """Создание начальных вершин куба (без вращения)"""
        s = self.size / 2
        vertices = np.array([
            [-s, -s, -s],  # 0: задняя-нижняя-левая
            [ s, -s, -s],  # 1: задняя-нижняя-правая
            [ s,  s, -s],  # 2: задняя-верхняя-правая
            [-s,  s, -s],  # 3: задняя-верхняя-левая
            [-s, -s,  s],  # 4: передняя-нижняя-левая
            [ s, -s,  s],  # 5: передняя-нижняя-правая
            [ s,  s,  s],  # 6: передняя-верхняя-правая
            [-s,  s,  s]   # 7: передняя-верхняя-левая
        ])
        
        # Смещаем к центру
        vertices += self.center
        return vertices
    
    def set_orientation(self, quaternion_array):
        """Устанавливаем ориентацию куба по кватерниону из массива [w, x, y, z]"""
        # Создаем объект кватерниона
        q = Quaternion(quaternion_array[0], quaternion_array[1], 
                      quaternion_array[2], quaternion_array[3])
        
        # Применяем вращение к каждой начальной вершине
        rotated_vertices = []
        for vertex in self.initial_vertices:
            # Вращаем относительно центра куба
            rotated = q.rotate_point(vertex - self.center) + self.center
            rotated_vertices.append(rotated)
        
        self.vertices = np.array(rotated_vertices)

# ==============================
# Парсер и приемник данных
# ==============================

class TextDataParser:
    def __init__(self):
        # Все поля, которые могут приходить от датчика
        self.field_names = [
            "AccX", "AccY", "AccZ",
            "GyroX", "GyroY", "GyroZ", 
            "AngleX", "AngleY", "AngleZ",
            "MagX", "MagY", "MagZ",
            "Temp1", "Temp2",
            "Status",
            "Pressure",
            "Quat1", "Quat2", "Quat3", "Quat4",
            "Unk1", "Unk2", "Unk3", "Unk4"
        ]
    
    def parse_packet(self, data_str):
        """
        Парсит текстовый пакет от датчика
        Формат: WT4700001010<числа,разделенные,запятами>
        Возвращает словарь со всеми распарсенными значениями
        """
        parsed = {}
        
        try:
            # Извлекаем ID устройства (первые 12 символов)
            if len(data_str) < 12:
                return None
                
            parsed['DeviceID'] = data_str[:12]
            
            # Остальная часть - данные
            data_part = data_str[14:]
            
            # Разделяем значения по запятым
            values = data_part.split(',')
            
            # Преобразуем значения в float
            float_values = []
            for v in values:
                try:
                    v_clean = v.replace('(', '').replace(')', '')
                    float_values.append(float(v_clean))
                except:
                    float_values.append(0.0)
            
            # Сохраняем значения по всем известным полям
            for i, field in enumerate(self.field_names):
                if i < len(float_values):
                    parsed[field] = float_values[i]
                else:
                    parsed[field] = 0.0
            
            return parsed
            
        except Exception as e:
            print(f"Ошибка парсинга: {e}")
            return None

class SensorDataProcessor:
    """Класс для обработки данных с датчика и фильтрации"""
    def __init__(self, sample_rate=20, mag_units=1):
        self.sample_rate = sample_rate
        self.mag_units = mag_units  # 1 = микротеслы
        
        # Инициализация фильтров
        self.bias = imufusion.Bias(sample_rate)
        self.ahrs = imufusion.Ahrs()
        
        # Настройка AHRS фильтра
        self.ahrs.settings = imufusion.AhrsSettings(
            convention=imufusion.CONVENTION_ENU,   # ENU: X-восток, Y-север, Z-вверх
            gain=0.5,
            gyroscope_range=2000,
            acceleration_rejection=30,
            magnetic_rejection=30,
            recovery_trigger_period=5 * sample_rate,
        )
        
        # Время последнего обновления
        self.last_time = None
        
        # Текущий кватернион
        self.current_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        
        # Статистика
        self.packet_count = 0
        self.processed_count = 0
        
    def process_sensor_data(self, data_dict):
        """
        Обрабатывает данные с датчика и возвращает кватернион
        """
        if not data_dict:
            return None
        
        try:
            # Извлекаем сырые данные (с учетом инверсии осей)
            gyro_raw = np.array([
                -data_dict.get('GyroX', 0.0),  # Ось X
                -data_dict.get('GyroY', 0.0),  # Ось Y
                data_dict.get('GyroZ', 0.0)    # Ось Z
            ])
            
            acc_raw = np.array([
                -data_dict.get('AccX', 0.0),   # Ось X
                -data_dict.get('AccY', 0.0),   # Ось Y
                data_dict.get('AccZ', 0.0)     # Ось Z
            ])
            
            mag_raw = np.array([
                data_dict.get('MagX', 0.0),    # Ось X
                data_dict.get('MagY', 0.0),    # Ось Y
                data_dict.get('MagZ', 0.0)     # Ось Z
            ])
            
            # Конвертируем данные магнитометра в микротеслы при необходимости
            if self.mag_units == 1:
                mag_raw = mag_raw / 10.0
            
            # Вычисляем delta_time
            current_time = time.time()
            if self.last_time is None:
                delta_time = 1.0 / self.sample_rate
            else:
                delta_time = current_time - self.last_time
            self.last_time = current_time
            
            # Обработка данных фильтром
            gyro_calibrated = self.bias.update(gyro_raw)
            self.ahrs.update(gyro_calibrated, acc_raw, mag_raw, delta_time)
            
            # Получаем кватернион
            self.current_quaternion = self.ahrs.quaternion.copy()
            
            self.processed_count += 1
            return self.current_quaternion
            
        except Exception as e:
            print(f"Ошибка обработки данных: {e}")
            return None
    
    def get_current_quaternion(self):
        """Возвращает текущий кватернион"""
        return self.current_quaternion.copy()

# ==============================
# UDP-приемник данных
# ==============================

class UDPDataReceiver:
    def __init__(self, port=1399, processor_callback=None):
        self.port = port
        self.is_running = False
        self.socket = None
        self.parser = TextDataParser()
        self.processor_callback = processor_callback
        
        # Потокобезопасные структуры данных
        self.data_lock = threading.Lock()
        self.packet_count = 0
        self.last_data = None
        self.last_quaternion = None
        self.data_queue = deque(maxlen=100)  # Очередь последних данных
        
    def start(self):
        """Запуск UDP-сервера для приёма данных от датчика"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("0.0.0.0", self.port))
        self.socket.settimeout(0.1)
        self.is_running = True
        
        print(f"UDP-сервер запущен на порту {self.port}")
        print("Ожидание данных от датчика...\n")
        
        # Запускаем поток приёма данных
        self.receive_thread = threading.Thread(target=self._receive_loop)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        return self

    def _receive_loop(self):
        """Основной цикл приёма данных от датчика"""
        buffer = b''
        
        while self.is_running:
            try:
                data, addr = self.socket.recvfrom(1024)
                buffer += data
                
                # Пакеты разделяются \r\n
                while b'\r\n' in buffer:
                    packet, buffer = buffer.split(b'\r\n', 1)
                    
                    try:
                        packet_str = packet.decode('ascii', errors='ignore').strip()
                        
                        # Проверяем, что это данные от нашего датчика
                        if packet_str.startswith('WT') and ',' in packet_str:
                            parsed = self.parser.parse_packet(packet_str)
                            
                            if parsed:
                                with self.data_lock:
                                    self.last_data = parsed
                                    self.packet_count += 1
                                    self.data_queue.append(parsed)
                                    
                                    # Обрабатываем данные через callback
                                    if self.processor_callback:
                                        quat = self.processor_callback(parsed)
                                        if quat is not None:
                                            self.last_quaternion = quat
                    
                    except Exception as e:
                        print(f"Ошибка обработки пакета: {e}")
                        continue
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"Ошибка приёма: {e}")
                break

    def get_last_quaternion(self):
        """Получение последнего кватерниона"""
        with self.data_lock:
            return self.last_quaternion
    
    def get_packet_count(self):
        """Получение количества пакетов"""
        with self.data_lock:
            return self.packet_count
    
    def stop(self):
        """Корректная остановка приёмника"""
        self.is_running = False
        if self.socket:
            self.socket.close()
        print(f"\nВсего получено пакетов: {self.packet_count}")

# ==============================
# Визуализация в реальном времени
# ==============================

class RealTimeVisualization:
    def __init__(self, data_receiver, update_interval=50):
        self.data_receiver = data_receiver
        self.update_interval = update_interval  # в миллисекундах
        
        # Создаем фигуру
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Настройка осей
        self.ax.set_title('Визуализация ориентации в реальном времени')
        self.ax.set_xlabel('X (Восток)')
        self.ax.set_ylabel('Y (Север)')
        self.ax.set_zlabel('Z (Вверх)')
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        
        # Создаем куб
        self.cube = Cube3D(center=(0, 0, 0), size=1.5)
        
        # Текстовые элементы для информации
        self.info_text = None
        self.packet_text = None
        
        # Статистика
        self.frame_count = 0
        self.last_update_time = time.time()
        self.update_rate = 0
        
        # Анимация
        self.ani = None
    
    def init_visualization(self):
        """Инициализация визуализации"""
        self.ax.clear()
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel('X (Восток)')
        self.ax.set_ylabel('Y (Север)')
        self.ax.set_zlabel('Z (Вверх)')
        self.ax.set_title('Визуализация ориентации в реальном времени')
        
        # Рисуем оси координат
        axis_length = 2
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, linewidth=2)
        self.ax.text(axis_length, 0, 0, 'X (Восток)', color='r', fontsize=10)
        self.ax.text(0, axis_length, 0, 'Y (Север)', color='g', fontsize=10)
        self.ax.text(0, 0, axis_length, 'Z (Вверх)', color='b', fontsize=10)
        
        # Создаем текстовые элементы
        self.info_text = self.ax.text2D(0.02, 0.98, "", transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        self.packet_text = self.ax.text2D(0.02, 0.85, "", transform=self.ax.transAxes,
                                         fontsize=9, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        return []
    
    def update_visualization(self, frame):
        """Обновление визуализации"""
        # Получаем последний кватернион
        quaternion = self.data_receiver.get_last_quaternion()
        packet_count = self.data_receiver.get_packet_count()
        
        # Очищаем оси
        self.ax.clear()
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([-2, 2])
        self.ax.set_xlabel('X (Восток)')
        self.ax.set_ylabel('Y (Север)')
        self.ax.set_zlabel('Z (Вверх)')
        
        # Обновляем заголовок
        current_time = time.strftime("%H:%M:%S")
        self.ax.set_title(f'Визуализация ориентации в реальном времени\n{current_time}')
        
        # Обновляем куб если есть кватернион
        if quaternion is not None:
            self.cube.set_orientation(quaternion)
            
            # Рисуем куб
            for i, face in enumerate(self.cube.faces):
                # Получаем вершины для грани
                face_vertices = [self.cube.vertices[j] for j in face]
                
                # Создаем полигон для грани
                polygon = Poly3DCollection([face_vertices], alpha=0.8, linewidths=1, edgecolors='k')
                polygon.set_facecolor(self.cube.face_colors[i])
                self.ax.add_collection3d(polygon)
        
        # Рисуем оси координат
        axis_length = 2
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, linewidth=2)
        self.ax.text(axis_length, 0, 0, 'X (Восток)', color='r', fontsize=10)
        self.ax.text(0, axis_length, 0, 'Y (Север)', color='g', fontsize=10)
        self.ax.text(0, 0, axis_length, 'Z (Вверх)', color='b', fontsize=10)
        
        # Обновляем информацию
        if quaternion is not None:
            info_str = (f'Кватернион:\n'
                       f'w={quaternion[0]:.4f}\n'
                       f'x={quaternion[1]:.4f}\n'
                       f'y={quaternion[2]:.4f}\n'
                       f'z={quaternion[3]:.4f}')
            
            # Вычисляем частоту обновления
            current_time = time.time()
            time_diff = current_time - self.last_update_time
            if time_diff > 0:
                self.update_rate = 0.9 * self.update_rate + 0.1 * (1.0 / time_diff)
            self.last_update_time = current_time
        else:
            info_str = "Ожидание данных..."
        
        # Отображаем информацию
        self.info_text = self.ax.text2D(0.02, 0.98, info_str, transform=self.ax.transAxes,
                                       fontsize=10, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        packet_str = (f'Пакетов: {packet_count}\n'
                     f'Частота обновления: {self.update_rate:.1f} Hz\n'
                     f'Кадр визуализации: {self.frame_count}')
        self.packet_text = self.ax.text2D(0.02, 0.85, packet_str, transform=self.ax.transAxes,
                                         fontsize=9, verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Устанавливаем одинаковый масштаб по осям
        self.ax.set_box_aspect([1,1,1])
        
        self.frame_count += 1
        return []
    
    def start(self):
        """Запуск визуализации"""
        print("Запуск визуализации...")
        self.ani = FuncAnimation(self.fig, self.update_visualization,
                                init_func=self.init_visualization,
                                interval=self.update_interval,
                                blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
    
    def stop(self):
        """Остановка визуализации"""
        if self.ani:
            self.ani.event_source.stop()

# ==============================
# Основная программа
# ==============================

def main():
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ ОРИЕНТАЦИИ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("3D куб с использованием кватернионов")
    print("=" * 60)
    
    try:
        # Создаем процессор данных
        print("\nИнициализация процессора данных...")
        processor = SensorDataProcessor(sample_rate=20, mag_units=1)
        
        # Создаем приемник данных
        print("Инициализация UDP приемника...")
        receiver = UDPDataReceiver(port=1399, processor_callback=processor.process_sensor_data)
        
        # Запускаем прием данных
        receiver.start()
        
        # Создаем визуализацию
        print("Инициализация 3D визуализации...")
        visualization = RealTimeVisualization(receiver, update_interval=50)
        
        print("\nГотово к работе!")
        print("-" * 60)
        print("Нажмите Ctrl+C для выхода")
        print("=" * 60)
        
        # Запускаем визуализацию (блокирующая операция)
        visualization.start()
        
    except KeyboardInterrupt:
        print("\nОстановка по команде пользователя...")
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Корректная остановка
        if 'visualization' in locals():
            visualization.stop()
        if 'receiver' in locals():
            receiver.stop()
        
        print("\nПрограмма завершена.")

if __name__ == "__main__":
    # Проверка наличия необходимых библиотек
    try:
        import imufusion
        import matplotlib
    except ImportError as e:
        print(f"Ошибка: Не удалось импортировать необходимые библиотеки: {e}")
        print("Установите недостающие библиотеки:")
        print("pip install numpy matplotlib imufusion")
        exit(1)
    
    # Запуск основной программы
    main()