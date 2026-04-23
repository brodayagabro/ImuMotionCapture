# host/viz/main.py
import sys
from pathlib import Path

# 🔧 АВТО-ФИКС ПУТЕЙ: добавляем корень проекта в sys.path
# Это позволяет запускать скрипт из любой папки и с любым способом вызова
CURRENT_FILE = Path(__file__).resolve()
# Ищем папку 'mocap-system' или любую, содержащую 'host/' и 'firmware/'
for parent in [CURRENT_FILE] + list(CURRENT_FILE.parents):
    if (parent / 'host').is_dir() and (parent / 'firmware').is_dir():
        PROJECT_ROOT = parent
        break
else:
    PROJECT_ROOT = CURRENT_FILE.parent.parent  # fallback: два уровня вверх

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ✅ Теперь импорты работают
import tkinter as tk
from host.viz.app import IMUApp

def main():
    try:
        import imufusion
    except ImportError:
        print("❌ Ошибка: библиотека imufusion не установлена.")
        print("📦 Установите: pip install imufusion")
        exit(1)

    root = tk.Tk()
    app = IMUApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()