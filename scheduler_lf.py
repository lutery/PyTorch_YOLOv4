# lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf'] 
import math
import numpy as np
import matplotlib.pyplot as plt

def lf(x: float, epochs: int, lrf=0.01) -> float:
    """
    Cosine learning rate scheduler
    :param x: current epoch
    :param epochs: total epochs
    :param hyp: hyperparameters, unused here but kept for compatibility
    :return: learning rate multiplier
    """
    return ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # hyp['lrf'] is set to 0.01


def main():
    epochs = 300
    lrf = 0.2
    n = epochs + 1

    xs = np.linspace(0, 2 * epochs, n)
    ys = [lf(x, epochs, lrf) for x in xs]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, linewidth=2)
    plt.title(f'Cosine Learning Rate Scheduler, epochs = {epochs}, lrf = {lrf}')
    plt.xlabel("x (epoch)")
    plt.ylabel("lr_factor")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()