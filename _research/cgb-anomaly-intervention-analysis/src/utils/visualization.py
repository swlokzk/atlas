import matplotlib.pyplot as plt


def set_matplotlib_defaults():
    plt.rcParams.update({"figure.max_open_warning": 0})


def plot_predictions(y_true, y_pred, title: str = "predictions"):
    plt.figure(figsize=(8, 3))
    plt.plot(y_true, label="true")
    plt.plot(y_pred, label="pred")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    return plt
