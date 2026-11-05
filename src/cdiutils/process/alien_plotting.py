import numpy as np
import pylab as plt

from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from cdiutils.plot.slice import plot_volume_slices
import plotly.express as px

###########################################################################################################################################
############################################                   Plot utilites                   ############################################
###########################################################################################################################################

def plot_3D_projections(
    data,
    mask=None,
    alpha_mask=0.3,
    ax=None,
    fig=None,
    cmap=None,
    log_scale=False,
    fig_title=None,
):
    data = np.asarray(data)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data, got shape {data.shape}.")

    # Calculate projections without modifying the original data.
    projections = [
        np.nanmean(data, axis=0),
        np.nanmean(data, axis=1),
        np.nanmean(data, axis=2).T,
    ]

    # Match the original plotting behaviour:
    # take the logarithm after calculating each projection.
    if log_scale:
        projections = [
            np.log(np.maximum(projection, 1e-10))
            for projection in projections
        ]

    if mask is not None:
        mask = np.asarray(mask, dtype=float)
        mask_projections = [
            np.nanmean(mask, axis=0),
            np.nanmean(mask, axis=1),
            np.nanmean(mask, axis=2).T,
        ]
    else:
        mask_projections = [None] * 3

    if fig is None:
        fig, axes = plt.subplots(
            1, 3, figsize=(6, 2), layout="tight"
        )
    else:
        axes = np.asarray(ax).ravel()

    for axis, projection, mask_projection in zip(
        axes, projections, mask_projections
    ):
        axis.imshow(
            projection,
            cmap=cmap or "turbo",
            aspect="equal",
            interpolation="nearest",
            origin="upper",
        )

        if mask_projection is not None:
            mask_plot = (mask_projection != 0).astype(float)
            axis.imshow(
                np.dstack([
                    mask_plot,
                    np.zeros(mask_plot.shape),
                    np.zeros(mask_plot.shape),
                    alpha_mask * mask_plot,
                ]),
                aspect="equal",
                interpolation="nearest",
                origin="upper",
            )

    # cdiutils detector-axis labels
    axes[0].set_xlabel(r"axis$_{2}$, det. horiz.")
    axes[0].set_ylabel(r"axis$_{1}$, det. vert.")

    axes[1].set_xlabel(r"axis$_{2}$, det. horiz.")
    axes[1].set_ylabel(r"axis$_{0}$, rocking curve")

    axes[2].set_xlabel(r"axis$_{0}$, rocking curve")
    axes[2].set_ylabel(r"axis$_{1}$, det. vert.")

    if fig_title is not None:
        fig.suptitle(fig_title)

    fig.tight_layout()
    return fig

    axes[0].set_xlabel(r"axis$_{2}$, det. horiz.")
    axes[0].set_ylabel(r"axis$_{1}$, det. vert.")

    axes[1].set_xlabel(r"axis$_{2}$, det. horiz.")
    axes[1].set_ylabel(r"axis$_{0}$, rocking curve")

    axes[2].set_xlabel(r"axis$_{0}$, rocking curve")
    axes[2].set_ylabel(r"axis$_{1}$, det. vert.")

    if fig_title is not None:
        fig.suptitle(fig_title)

    fig.tight_layout()
    return fig


def plot_3D_slice(data,
                  mask=None,
                  ax=None, fig=None,
                  cmap=None,
                  log_scale=False):
    '''
    Same as plot_3D_projections but with only slices at the center of the data array instead of projections along axis.
    '''

    if fig is None:
        fig, ax = plt.subplots(1,3, figsize=(6,2))

    for n in range(3):
        s = [slice(None) for n in range(3)]
        s[n] = data.shape[n]//2

        if log_scale:
            ax[n].matshow(np.log(data[s]),cmap=cmap, aspect='auto')
        else:
            ax[n].matshow(data[s],cmap=cmap, aspect='auto')

        if mask is not None :
            ax[n].matshow(mask[s],cmap='Reds', alpha=.7, aspect='auto')

    fig.tight_layout()
    return



###########################################################################################################################################
########################                  Plotly functions for a 3D scatter plot of the clusters                   ########################
###########################################################################################################################################

def get_colors(arr, cmap='viridis'):
    '''
    transform an array of integers into an array of different colors

    Parameters
    ----------
    :arr: 1D array of integers
    :cmap: the colormap from which the colors are taken

    Returns
    ----------
    :colors: 1D array of same size as arr containing corresponding colors
    '''
    cmap = cm.get_cmap(cmap)
    n = len(np.unique(arr))
    colornorm = Normalize(vmin=1, vmax=n)
    hex_map = dict()
    for i, cl in enumerate(np.unique(arr)):
        hex_map[cl] = to_hex(cmap(colornorm(i + 1)))
    colors = list(map(lambda x: hex_map[x], arr))
    return colors

def plotly_scatter3D_clusters(pixels, pixels_labels,
                              Nb_pixels_partial = 10000,
                              marker_size=1.5):
    '''
    Make a clusters 3D scatter plot using plotly.express.scatter_3d and a different color for each cluster

    Parameters
    ----------
    :pixels: points positions with shape (number of points, 3)
    :pixels_labels: array of integer corresponding to the cluster label of each pixel. shape (number of points)
    :Nb_pixels_partial: number of pixels used in the plot. We don't use all of them for rapidity issues.
    :marker_size: size of the scattered points
    '''

    from matplotlib import cm
    from matplotlib.colors import Normalize, to_hex
    import plotly.express as px

    if Nb_pixels_partial is not None:
        if len(pixels)<Nb_pixels_partial:
            Nb_pixels_partial = len(pixels)
        rand_indices = np.random.choice(np.arange(pixels.shape[0]),Nb_pixels_partial)
        pixels_partial = pixels[rand_indices]
        pixels_labels_partial = pixels_labels[rand_indices]

    colors= get_colors(pixels_labels_partial, cmap='viridis')

    fig = px.scatter_3d(x=pixels_partial[...,0],y=pixels_partial[...,1],z=pixels_partial[...,2], color=colors)
    fig.update_traces(marker_size = marker_size)
    fig.show()

    return