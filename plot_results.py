import matplotlib.pyplot as plt


files = {
        "eval_results/eval_tight.txt": ("o", "solid"),
        "eval_results/eval_arcface.txt": ("o", "solid"),
        "eval_results/eval_model2.txt": ("X", "dashed"),
        "eval_results/random_crop_random_blur.txt": ("X", "dashed"),
        # "eval_results/random_crop_random_blur_20.txt": ("X", "dashed"),
         }

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
        if not line:
            break
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
                 marker=files[file][0], linestyle=files[file][1],
                 label=model_vals[file]["label"], linewidth=3)

    
    plt.xlabel('Blur Sigma')
    plt.ylabel('Cosine Similarity' if metric != 'diff' else '1 - Cosine Similarity')
    plt.title(titles[idx])
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0,1)
    # plt.ylim(0.0,0.8)

    plt.show()
