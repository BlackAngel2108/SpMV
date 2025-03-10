import pandas as pd
import matplotlib.pyplot as plt

# Чтение данных из CSV-файла с указанием имен столбцов
data = pd.read_csv("results.csv", header=None)

# Удаление нулевой строки (первой строки данных)
data = data.drop(0).reset_index(drop=True)

# Назначение имен столбцам вручную (убираем _time)
data.columns = ["Matrix", "COO", "CSR", "DIAG", "ELLPack", "SELL_C", "SELL_C_sigma"]

# Преобразуем данные в числовой формат (если они строки)
for col in data.columns[1:]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Удаляем строки, где все значения (кроме "Matrix") равны нулю
data = data.loc[~(data.iloc[:, 1:] < 0.1).all(axis=1)]

# Удаляем дубликаты строк на основе столбца "Matrix"
data = data.drop_duplicates(subset=["Matrix"], keep="first")

# Проверка данных
print("Данные после предобработки:")
print(data)

# Построение графика
plt.figure(figsize=(14, 8))  # Увеличиваем размер графика для удобства

# Ширина столбцов
bar_width = 0.12  # Уменьшаем ширину столбцов, чтобы все поместились

# Позиции для столбцов
positions = range(len(data["Matrix"]))

# Столбчатая диаграмма для COO
plt.bar(positions, data["COO"], width=bar_width, label="COO", color="blue")

# Столбчатая диаграмма для CSR
plt.bar([p + bar_width for p in positions], data["CSR"], width=bar_width, label="CSR", color="green")

# Столбчатая диаграмма для DIAG
plt.bar([p + 2*bar_width for p in positions], data["DIAG"], width=bar_width, label="DIAG", color="red")

# Столбчатая диаграмма для ELLPack
plt.bar([p + 3*bar_width for p in positions], data["ELLPack"], width=bar_width, label="ELLPack", color="purple")

# Столбчатая диаграмма для SELL_C
plt.bar([p + 4*bar_width for p in positions], data["SELL_C"], width=bar_width, label="SELL-C", color="grey")

# Столбчатая диаграмма для SELL_C_sigma
plt.bar([p + 5*bar_width for p in positions], data["SELL_C_sigma"], width=bar_width, label="SELL-C-sigma", color="orange")

# Настройка графика
plt.title("Время выполнения SpMV для разных форматов матриц", fontsize=16)
plt.xlabel("Название матрицы", fontsize=14)
plt.ylabel("Время выполнения (сек)", fontsize=14)
plt.xticks([p + 2.5*bar_width for p in positions], data["Matrix"], rotation=45, ha="right")  # Поворот названий матриц для удобства чтения
plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')  # Легенда сбоку
plt.grid(True, linestyle="--", alpha=0.6)  # Сетка

# Сохранение графика в файл
plt.tight_layout()
plt.savefig("spmv_performance.png", dpi=300)

# Показ графика
plt.show()

# Шаг 1: Выделяем матрицы, где DIAG лучше других
diag_best = data[data["DIAG"] == data[["COO", "CSR", "DIAG", "ELLPack", "SELL_C"]].min(axis=1)]

# Проверка данных
print("Матрицы, где DIAG лучше других:")
print(diag_best)

# Шаг 2: Построение диаграммы для матриц, где DIAG лучше других
plt.figure(figsize=(14, 8))  # Увеличиваем размер графика для удобства

# Позиции для столбцов
positions_diag = range(len(diag_best["Matrix"]))

# Столбчатая диаграмма для COO
plt.bar(positions_diag, diag_best["COO"], width=bar_width, label="COO", color="blue")

# Столбчатая диаграмма для CSR
plt.bar([p + bar_width for p in positions_diag], diag_best["CSR"], width=bar_width, label="CSR", color="green")

# Столбчатая диаграмма для DIAG
plt.bar([p + 2*bar_width for p in positions_diag], diag_best["DIAG"], width=bar_width, label="DIAG", color="red")

# Столбчатая диаграмма для ELLPack
plt.bar([p + 3*bar_width for p in positions_diag], diag_best["ELLPack"], width=bar_width, label="ELLPack", color="purple")

# Столбчатая диаграмма для SELL_C
plt.bar([p + 4*bar_width for p in positions_diag], diag_best["SELL_C"], width=bar_width, label="SELL-C", color="grey")

# Столбчатая диаграмма для SELL_C_sigma
plt.bar([p + 5*bar_width for p in positions_diag], diag_best["SELL_C_sigma"], width=bar_width, label="SELL-C-sigma", color="orange")

# Настройка графика
plt.title("Время выполнения SpMV для матриц, где DIAG лучше других", fontsize=16)
plt.xlabel("Название матрицы", fontsize=14)
plt.ylabel("Время выполнения (сек)", fontsize=14)
plt.xticks([p + 2.5*bar_width for p in positions_diag], diag_best["Matrix"], rotation=45, ha="right")  # Поворот названий матриц для удобства чтения
plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')  # Легенда сбоку
plt.grid(True, linestyle="--", alpha=0.6)  # Сетка

# Сохранение графика в файл
plt.tight_layout()
plt.savefig("spmv_performance_diag_best.png", dpi=300)

# Показ графика
plt.show()


# Шаг 1: Фильтруем данные, где COO > 0.3 секунды
coo_slow = data[data["COO"] > 0.3]

# Проверка данных
print("Матрицы, где COO > 0.3 секунды:")
print(coo_slow)

# Шаг 2: Построение диаграммы для матриц, где COO > 0.3 секунды
if coo_slow.empty:
    print("Нет матриц, где COO > 0.3 секунды.")
else:
    plt.figure(figsize=(14, 8))  # Увеличиваем размер графика для удобства

    # Позиции для столбцов
    positions_coo = range(len(coo_slow["Matrix"]))

    # Столбчатая диаграмма для COO
    plt.bar(positions_coo, coo_slow["COO"], width=bar_width, label="COO", color="blue")

    # Столбчатая диаграмма для CSR
    plt.bar([p + bar_width for p in positions_coo], coo_slow["CSR"], width=bar_width, label="CSR", color="green")

    # Столбчатая диаграмма для DIAG
    plt.bar([p + 2*bar_width for p in positions_coo], coo_slow["DIAG"], width=bar_width, label="DIAG", color="red")

    # Столбчатая диаграмма для ELLPack
    plt.bar([p + 3*bar_width for p in positions_coo], coo_slow["ELLPack"], width=bar_width, label="ELLPack", color="purple")

    # Столбчатая диаграмма для SELL_C
    plt.bar([p + 4*bar_width for p in positions_coo], coo_slow["SELL_C"], width=bar_width, label="SELL-C", color="grey")

    # Столбчатая диаграмма для SELL_C_sigma
    plt.bar([p + 5*bar_width for p in positions_coo], coo_slow["SELL_C_sigma"], width=bar_width, label="SELL-C-sigma", color="orange")

    # Настройка графика
    plt.title("Время выполнения SpMV для матриц, где COO > 0.3 секунды", fontsize=16)
    plt.xlabel("Название матрицы", fontsize=14)
    plt.ylabel("Время выполнения (сек)", fontsize=14)
    plt.xticks([p + 2.5*bar_width for p in positions_coo], coo_slow["Matrix"], rotation=45, ha="right")  # Поворот названий матриц для удобства чтения
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')  # Легенда сбоку
    plt.grid(True, linestyle="--", alpha=0.6)  # Сетка

    # Сохранение графика в файл
    plt.tight_layout()
    plt.savefig("spmv_performance_coo_slow.png", dpi=300)

    # Показ графика
    plt.show()



    # Шаг 1: Фильтруем данные, где COO <= 0.3 секунды
coo_slow = data[data["COO"] <= 0.3]

# Проверка данных
print("Матрицы, где COO <= 0.3 секунды:")
print(coo_slow)

# Шаг 2: Построение диаграммы для матриц, где COO <= 0.3 секунды
if coo_slow.empty:
    print("Нет матриц, где COO <= 0.3 секунды.")
else:
    plt.figure(figsize=(14, 8))  # Увеличиваем размер графика для удобства

    # Позиции для столбцов
    positions_coo = range(len(coo_slow["Matrix"]))

    # Столбчатая диаграмма для COO
    plt.bar(positions_coo, coo_slow["COO"], width=bar_width, label="COO", color="blue")

    # Столбчатая диаграмма для CSR
    plt.bar([p + bar_width for p in positions_coo], coo_slow["CSR"], width=bar_width, label="CSR", color="green")

    # Столбчатая диаграмма для DIAG
    plt.bar([p + 2*bar_width for p in positions_coo], coo_slow["DIAG"], width=bar_width, label="DIAG", color="red")

    # Столбчатая диаграмма для ELLPack
    plt.bar([p + 3*bar_width for p in positions_coo], coo_slow["ELLPack"], width=bar_width, label="ELLPack", color="purple")

    # Столбчатая диаграмма для SELL_C
    plt.bar([p + 4*bar_width for p in positions_coo], coo_slow["SELL_C"], width=bar_width, label="SELL-C", color="grey")

    # Столбчатая диаграмма для SELL_C_sigma
    plt.bar([p + 5*bar_width for p in positions_coo], coo_slow["SELL_C_sigma"], width=bar_width, label="SELL-C-sigma", color="orange")

    # Настройка графика
    plt.title("Время выполнения SpMV для матриц, где COO <= 0.3 секунды", fontsize=16)
    plt.xlabel("Название матрицы", fontsize=14)
    plt.ylabel("Время выполнения (сек)", fontsize=14)
    plt.xticks([p + 2.5*bar_width for p in positions_coo], coo_slow["Matrix"], rotation=45, ha="right")  # Поворот названий матриц для удобства чтения
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')  # Легенда сбоку
    plt.grid(True, linestyle="--", alpha=0.6)  # Сетка

    # Сохранение графика в файл
    plt.tight_layout()
    plt.savefig("spmv_performance_coo_slow.png", dpi=300)

    # Показ графика
    plt.show()


    # Шаг 3: Фильтруем данные, где ELLPack > 5 * минимальное значение других форматов
ellpack_slow = data[data["ELLPack"] > 5 * data[["COO", "CSR", "DIAG"]].max(axis=1)]

# Проверка данных
print("Матрицы, где ELLPack работает в пять раз медленнее других форматов:")
print(ellpack_slow)

if ellpack_slow.empty:
    print("Нет матриц, где ELLPack работает в пять раз медленнее других форматов.")
else:
    # Построение диаграммы для матриц, где ELLPack > 5 * минимальное значение других форматов
    plt.figure(figsize=(14, 8))  # Увеличиваем размер графика для удобства

    # Позиции для столбцов
    positions_ellpack = range(len(ellpack_slow["Matrix"]))

    # Столбчатая диаграмма для COO
    plt.bar(positions_ellpack, ellpack_slow["COO"], width=bar_width, label="COO", color="blue")

    # Столбчатая диаграмма для CSR
    plt.bar([p + bar_width for p in positions_ellpack], ellpack_slow["CSR"], width=bar_width, label="CSR", color="green")

    # Столбчатая диаграмма для DIAG
    plt.bar([p + 2*bar_width for p in positions_ellpack], ellpack_slow["DIAG"], width=bar_width, label="DIAG", color="red")

    # Столбчатая диаграмма для ELLPack
    plt.bar([p + 3*bar_width for p in positions_ellpack], ellpack_slow["ELLPack"], width=bar_width, label="ELLPack", color="purple")

    # Столбчатая диаграмма для SELL_C
    plt.bar([p + 4*bar_width for p in positions_ellpack], ellpack_slow["SELL_C"], width=bar_width, label="SELL-C", color="grey")

    # Столбчатая диаграмма для SELL_C_sigma
    plt.bar([p + 5*bar_width for p in positions_ellpack], ellpack_slow["SELL_C_sigma"], width=bar_width, label="SELL-C-sigma", color="orange")

    # Настройка графика
    plt.title("Время выполнения SpMV для матриц, где ELLPack работает в пять раз медленнее других форматов", fontsize=16)
    plt.xlabel("Название матрицы", fontsize=14)
    plt.ylabel("Время выполнения (сек)", fontsize=14)
    plt.xticks([p + 2.5*bar_width for p in positions_ellpack], ellpack_slow["Matrix"], rotation=45, ha="right")  # Поворот названий матриц для удобства чтения
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')  # Легенда сбоку
    plt.grid(True, linestyle="--", alpha=0.6)  # Сетка

    # Сохранение графика в файл
    plt.tight_layout()
    plt.savefig("spmv_performance_ellpack_slow.png", dpi=300)

    # Показ графика
    plt.show()