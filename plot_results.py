import matplotlib.pyplot as plt

with open("blur_evaluation_results/final_results.txt") as f:
    lines = f.readlines()

for _ in range(2):
    lines.pop(0)

label = lines.pop(0).split(": ")[1]

same_person = {}
diff_person = {}
all_pairs = {}

titles = ["Same Person Evaluation",
"Different Person Evaluation",
"All Pairs Evaluation"]




while lines:
    line = lines.pop(0)
    print(line)
    line = line.strip()[:-1]
    _, blur_level = line.split()
    for metric in (same_person, diff_person, all_pairs):
        line = lines.pop(0)
        val = float(line.split("= ")[-1])
        metric[blur_level] = val
    

for idx, metric in enumerate((same_person, diff_person, all_pairs)):
    plt.plot(metric.keys(), metric.values(), marker='o', label=label)
    plt.xlabel('Blur Sigma')
    # plt.ylabel('Cosine Similarity' if category == 'same' else '1 - Cosine Similarity')
    plt.ylabel('Cosine Similarity')
    plt.title(titles[idx])
    plt.legend()
    plt.grid(True)
    plt.show()
