import pandas as pd
import matplotlib.pyplot as plt

# Читаем данные из CSV-файла с указанием кодировки Windows-1251
data = pd.read_csv("results.csv", encoding="cp1251")

# Переименовываем столбцы для удобства
data.columns = ["Matrix", "COO_time", "CSR_time", "DIAG_time", "ELLPack_time", "CELL_C_time"]

# Выводим данные для проверки
print(data)

# Построение графика
plt.figure(figsize=(12, 8))

# График для COO_time
plt.plot(data["Matrix"], data["COO_time"], label="COO", marker="o", linestyle="-", color="blue")

# График для CSR_time
plt.plot(data["Matrix"], data["CSR_time"], label="CSR", marker="s", linestyle="--", color="green")

# График для DIAG_time
plt.plot(data["Matrix"], data["DIAG_time"], label="DIAG", marker="^", linestyle="-.", color="red")

# График для LPack_time
plt.plot(data["Matrix"], data["ELLPack_time"], label="LPack", marker="D", linestyle=":", color="purple")

# График для CELL_C_time
plt.plot(data["Matrix"], data["CELL_C_time"], label="LPack", marker="D", linestyle=":", color="purple")

# Настройка графика
plt.title("Время выполнения SpMV для разных форматов матриц", fontsize=16)
plt.xlabel("Название матрицы", fontsize=14)
plt.ylabel("Время выполнения (сек)", fontsize=14)
plt.xticks(rotation=45, ha="right")  # Поворот названий матриц для удобства чтения
plt.legend(fontsize=12)  # Легенда
plt.grid(True, linestyle="--", alpha=0.6)  # Сетка

# Сохранение графика в файл
#plt.tight_layout()
#plt.savefig("spmv_performance.png", dpi=300)

# Показ графика
plt.show()