import pandas as pd
import re
from collections import Counter
from pathlib import Path
import csv

# 0. 合法元素集合（你给的这份表）
valid_elements = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
    'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Fl', 'Lv', 'Ts', 'Og'
}

# 1. 读取 CSV
# data_path = r"E:\AMsystem\Project\QCH-RAG\data\HER.csv"
data_path = r"E:\AMsystem\Project\QCH-RAG\data\HER-clear.csv"
df = pd.read_csv(data_path, encoding="gbk")  # 很重要：encoding="gbk"

print("总行数：", len(df))
print("列名：", df.columns.tolist())
print(df.head(3))  # 先看前几行确认没问题


# 2. 提取元素并统计频次
counter = Counter()

def extract_elements(text):
    if pd.isna(text):
        return []
    text = str(text)
    # 用正则找出所有元素符号（如 Ni, Co, Pt, Fe, Pd, Ru, Ir 等）
    candidates = re.findall(r"[A-Z][a-z]?", text)
    # 只保留在 valid_elements 集合中的“合法元素”
    els = [e for e in candidates if e in valid_elements]
    return els

all_elements_per_row = []  # 也顺便保存每行有哪些元素

for idx, row in df.iterrows():
    elements = extract_elements(row["high_entropy_alloy_elements"])
    all_elements_per_row.append(elements)
    counter.update(elements)

# 打印前 15 个最常出现的元素
print("元素出现频次前 15：")
for elem, freq in counter.most_common(15):
    print(f"{elem}: {freq}")

# 3. 把每行的元素列表存回 df（方便以后用）
df["parsed_elements"] = [";".join(els) for els in all_elements_per_row]

# 4. 保存一个“干净版本”的表
output_path = r"E:\AMsystem\Project\QCH-RAG\data\her_hea_literature_clean.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print("清洗后的表已保存为：", output_path)


# 5.
# 定义贵金属 / 非贵金属
precious = {"Pt", "Ir", "Ru", "Rh", "Pd", "Au", "Ag"}

element_rows = []
for elem, freq in counter.most_common():
    elem_type = "PM" if elem in precious else "NM"
    element_rows.append({"Element": elem, "Type": elem_type, "Frequency": freq})

# 保存到新文件
elem_stats_path = r"E:\AMsystem\Project\QCH-RAG\data\hea_element_stats.csv"
with open(elem_stats_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["Element", "Type", "Frequency"])
    writer.writeheader()
    writer.writerows(element_rows)

print("元素统计表已保存为：", elem_stats_path)