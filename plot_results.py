import matplotlib.pyplot as plt


files = ["eval_results/eval_tight.txt", "eval_results/eval_arcface.txt"] 

model_vals = {}

for file in files:
    model_vals[file] = {}
    with open(file) as f:
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
            val = float(line.split("= ")[-1].split("(")[0])
            metric[blur_level] = val
        
    model_vals[file]["same"] = same_person
    model_vals[file]["diff"] = diff_person
    model_vals[file]["all"] = all_pairs
    model_vals[file]["label"] = str(label)

# for idx, metric in enumerate((same_person, diff_person, all_pairs)):


for idx, metric in enumerate(("same", "diff", "all")):

    for file in files:
        plt.plot(model_vals[file][metric].keys(), model_vals[file][metric].values(),
                 marker='o', label=model_vals[file]["label"])

    
    plt.xlabel('Blur Sigma')
    plt.ylabel('Cosine Similarity' if metric != 'diff' else '1 - Cosine Similarity')
    plt.title(titles[idx])
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.2,1.2)

    plt.show()
