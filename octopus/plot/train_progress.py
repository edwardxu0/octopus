import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib.ticker import AutoMinorLocator, MultipleLocator


class ProgressPlot:
    def __init__(self, figsize=(10, 5)):
        self.ax1 = None
        self.ax2 = None
        self.fig = plt.figure(figsize=figsize)

    def draw_train(self, X1, Y1, X2, Y2, xlim, ylim):
        colors = sns.color_palette("rocket", len(Y1))
        self.ax1 = self.fig.add_subplot()
        for i, se in enumerate(Y1):
            self.ax1.plot(
                X1,
                Y1[se],
                color=colors[i],
                alpha=0.5,
                label=f"Safe ReLU({se})",
            )

        self.ax1.scatter(X2, Y2, marker="x", color="red", label="BS Points")
        self.ax1.legend(loc="center left")

        self.ax1.set_xlim(xlim)
        self.ax1.set_ylim(ylim)
        self.ax1.set_xlabel("Batch ID")
        self.ax1.set_ylabel("Safe ReLU")

    def draw_safe_relu_pp(self, X1, Y1):
        self.ax1.scatter(X1, Y1, marker="v", color="deeppink", label="Safe ReLU PP")
        self.ax1.legend(loc="center left")

    def draw_accuracy(self, X1, Y1, X2, Y2, ylim):
        if self.ax1:
            self.ax2 = self.ax1.twinx()
        else:
            self.ax2 = self.fig.add_subplot()
        

        self.ax2.set_ylim([-0.1,1.1])
        self.ax2.scatter(X1, Y1, marker="o", color="Green", label="Test Accuracy")
        
        data = zip(X2,Y2)
        
        df = pd.DataFrame(data, columns=['id', 'loss'])
        Y2 = df.loss.rolling(10).mean()
        
        self.ax2.plot(X2, Y2, color="orange", alpha=0.25, label="Loss")
        self.ax2.legend(loc="center right")

        for i, txt in enumerate(Y1):
            self.ax2.annotate(
                f"{txt*100:5.2f}%", (X1[i] - len(X2) / 250, Y1[i] - 0.18), rotation=90
            )

    def draw_grid(self, x_stride=100, y_stride=0.1):
        if self.ax1:
            ax = self.ax1
        elif self.ax2:
            ax = self.ax2
        ax.xaxis.set_major_locator(MultipleLocator(x_stride))
        ax.xaxis.set_minor_locator(MultipleLocator(x_stride / 2))
        ax.yaxis.set_major_locator(MultipleLocator(y_stride))
        ax.yaxis.set_minor_locator(MultipleLocator(y_stride / 2))
        ax.grid(
            which="major", axis="x", color="grey", linestyle="--", linewidth=0.5
        )
        ax.grid(
            which="minor", axis="x", color="grey", linestyle=":", linewidth=0.1
        )
        ax.grid(
            which="major", axis="y", color="grey", linestyle="--", linewidth=0.5
        )
        ax.grid(
            which="minor", axis="y", color="grey", linestyle=":", linewidth=0.1
        )

    def save(self, title, path):
        #plt.title(title)
        plt.savefig(path, format="pdf", bbox_inches="tight")

    def clear(self):
        plt.close(self.fig)
