import matplotlib.pyplot as plt


def plot_stats(stats_array, plot_name, x_name, y_name):
    plt.plot(stats_array)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(plot_name)
    plt.savefig(plot_name + ".png")
    plt.cla()
    plt.clf()

