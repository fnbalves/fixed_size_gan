import matplotlib.pyplot as plt
import numpy as np


def view_samples(samples, labels, ncols, figsize=(5, 5)):
    nrows = int(len(samples) / ncols)
    # ge the figure and the axes
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                             sharey=True, sharex=True)

    # draw the samples
    for index, (ax, img) in enumerate(zip(axes.flatten(), samples)):
        if labels is not None:
            current_label = labels[index]
            ax.set_title(current_label, fontsize=7,
                         color='k', loc='left', pad=1)
        ax.axis('off')

        img = ((img - img.min()) * 255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box')
        ax.imshow(img, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0.35)
    return fig, axes