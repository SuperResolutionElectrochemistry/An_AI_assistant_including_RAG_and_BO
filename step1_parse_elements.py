import pandas as pd
import re
from collections import Counter
from pathlib import Path
import csv

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



data_path = your csv file
df = pd.read_csv(data_path, encoding="gbk")

print("总行数：", len(df))
print("列名：", df.columns.tolist())
print(df.head(3))  


counter = Counter()

def extract_elements(text):
    if pd.isna(text):
        return []
    text = str(text)
    candidates = re.findall(r"[A-Z][a-z]?", text)
    els = [e for e in candidates if e in valid_elements]
    return els

all_elements_per_row = [] 

for idx, row in df.iterrows():
    elements = extract_elements(row["high_entropy_alloy_elements"])
    all_elements_per_row.append(elements)
    counter.update(elements)

print("元素出现频次前 15：")
for elem, freq in counter.most_common(15):
    print(f"{elem}: {freq}")

df["parsed_elements"] = [";".join(els) for els in all_elements_per_row]

output_path = save path
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(output_path)


element_rows = []
for elem, freq in counter.most_common():
    elem_type = "PM" if elem in precious else "NM"
    element_rows.append({"Element": elem, "Type": elem_type, "Frequency": freq})


elem_stats_path = save path
with open(elem_stats_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["Element", "Type", "Frequency"])
    writer.writeheader()
    writer.writerows(element_rows)
