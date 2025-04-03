import matplotlib.pyplot as plt

with open("blur_evaluation_results/final_results.txt") as f:
    lines = f.readlines()

for _ in range(3):
    lines.pop(0)


vals = {}
while lines:
    line = lines.pop(0)
    print(line)
    line = line.strip()[:-1]
    _, blur_level = line.split()
    for i in range(3):
        line = lines.pop(0)
        val = float(line.split("= ")[-1])
        vals[blur_level] = val
    print(vals)

