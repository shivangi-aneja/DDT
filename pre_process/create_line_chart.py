import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def create_line_chart(ce, ft, vae, xtra, classes, filename):
    fig, ax = plt.subplots()
    ax.plot(classes, ce, marker="o", linewidth=4, markersize=12, label="FS to DF",
            markerfacecolor='firebrick', color='salmon')
    ax.plot(classes, ft, marker="v", markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4,
            label="F2F to DF")
    ax.plot(classes, vae, marker="*", markerfacecolor='green', markersize=14, color='lightgreen', linewidth=4,
            label="NT to DF")
    ax.plot(classes, xtra, marker="^", markerfacecolor='purple', markersize=12, color='mediumslateblue', linewidth=4,
            label="FS, NT, F2F to DF")
    ax.legend()
    fig.tight_layout()
    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    # plt.xlabel("# images fine-tuned with")
    # plt.ylabel('Accuracy')
    plt.savefig(filename)


classes = ['0', '5', '10', '50', '100']
# FS
f2f_ce = [
49.87,
62.72,
73.48,
85.46,
88.60
]

# F2F
f2f_ft = [
52.06,
78.05,
79.97,
84.62,
90.16
]

# NT
f2f_vae = [
68.54,
78.61,
80.13,
87.67,
90.67
]

# All
all_vae = [
72.67,
80.58,
82,
89.14,
91.27]

create_line_chart(f2f_ce, f2f_ft, f2f_vae,all_vae, classes, 'all.png')
