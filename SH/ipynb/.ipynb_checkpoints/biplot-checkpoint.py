def eigen_scaling(pca, scaling = 0):
    # pca is a PCA object obtained from statsmodels.multivariate.pca
    # scaling is one of [0, 1, 2, 3]
    # the eigenvalues of the pca object are n times the ones computed with R
    # we thus need to divide their sum by the number of rows
    const = ((pca.scores.shape[0]-1) * pca.eigenvals.sum()/ pca.scores.shape[0])**0.25
    if scaling == 0:
        scores = pca.scores
        loadings = pca.loadings
    elif scaling == 1:
        scaling_fac = (pca.eigenvals / pca.eigenvals.sum())**0.5
        scaling_fac.index = pca.scores.columns
        scores = pca.scores * scaling_fac * const
        loadings = pca.loadings * const
    elif scaling == 2:
        scaling_fac = (pca.eigenvals / pca.eigenvals.sum())**0.5
        scaling_fac.index = pca.scores.columns
        scores = pca.scores * const
        loadings = pca.loadings * scaling_fac * const
    elif scaling == 3:
        scaling_fac = (pca.eigenvals / pca.eigenvals.sum())**0.25
        scaling_fac.index = pca.scores.columns
        scores = pca.scores * scaling_fac * const
        loadings = pca.loadings * scaling_fac * const
    else:
        sys.exit("Scaling should either be 0, 1, 2 or 3")
    return([scores, loadings])


def biplot(pca, scaling = 0, plot_loading_labels = True, color = None, alpha_scores = 1):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    scores_, loadings = eigen_scaling(pca, scaling=scaling)
    # Plot scores
    if color is None:
        sns.relplot(
            x = "comp_0",
            y = "comp_1",
            alpha = alpha_scores,
            data = scores_,
        )
    else:
        scores_ = scores_.copy()
        scores_["group"] = color
        sns.relplot(
            x = "comp_0",
            y = "comp_1",
            alpha = alpha_scores,
            data = scores_,
        )

    # Plot loadings
    if plot_loading_labels:
        loading_labels = pca.loadings.index

    for i in range(loadings.shape[0]):
        plt.arrow(
            0, 0,
            loadings.iloc[i, 0],
            loadings.iloc[i, 1],
            color = 'red',
            alpha = 0.7,
            linestyle = '-',
            head_width = loadings.values.max() / 50,
            width = loadings.values.max() / 2000,
            length_includes_head = True
        )
        if plot_loading_labels:
            plt.text(
                loadings.iloc[i, 0]*1.05,
                loadings.iloc[i, 1]*1.05,
                loading_labels[i],
                color = 'black',
                ha = 'center',
                va = 'center',
                fontsize = 10
            );

    # range of the plot
    scores_loadings = np.vstack([scores_.values[:, :2], loadings.values[:, :2]])
    xymin = scores_loadings.min(axis=0) * 1.2
    xymax = scores_loadings.max(axis=0) * 1.2

    plt.axhline(y = 0, color = 'k', linestyle = 'dotted', linewidth=0.75)
    plt.axvline(x = 0, color = 'k', linestyle = 'dotted', linewidth=0.75)
    plt.xlabel("Comp1")
    plt.ylabel("Comp2")
    plt.xlim(xymin[0], xymax[0])
    plt.ylim(xymin[1], xymax[1]);