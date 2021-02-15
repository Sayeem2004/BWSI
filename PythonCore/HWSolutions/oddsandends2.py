import matplotlib.pyplot as plt;
import numpy as np;

def main():
    data = np.load("mystery-img.npy");
    fig, axs = plt.subplots();
    axs.imshow(data);
    plt.show();

main();
