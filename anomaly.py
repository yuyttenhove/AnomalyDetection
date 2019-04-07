import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import os
from sklearn.utils import shuffle, resample
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from matplotlib.colors import LogNorm
from sklearn.cluster import KMeans

script_dir = os.path.dirname(__file__)

filename = os.path.join(script_dir, "anomaly.csv")
data = np.loadtxt(filename, delimiter=",")
data = shuffle(data)

gmm = None

def build_gmm(n_centers=5, significance_level = .01, data=data):
    gmm = GaussianMixture(5)
    iterdata = data
    converged = False
    l = len(iterdata)

    while not converged:
        gmm.fit(iterdata)

        scores = gmm.score_samples(iterdata)
        
        mask = (scores < np.log(significance_level)).astype(int)

        iterdata = np.array([[i, j] for c, (i, j) in enumerate(iterdata) if not mask[c]])
        
        if len(iterdata) == l:
            converged = True
        else:
            l = len(iterdata)
    return gmm

def score(datapoint, gmm=gmm):
    if gmm == None:
        gmm = build_gmm()
    score = gmm.score_samples([[datapoint["x"], datapoint["y"]]])[0]
    return np.exp(score)

if __name__ == "__main__":
    # False to only save plots and don't show them when running this script
    show_plots = False
    ######################
    ## Data inspection: ##
    ######################

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], 1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter plot of the toy dataset")
    plt.savefig(os.path.join(script_dir, "figures", "data_inspection"))
    if show_plots:
        plt.show()

    #################
    ## GMM method: ##
    #################

    # try to detect the best suited number of centers for the Gaussian mixture model
    max_centers = 9
    
    # Cross validation to detect number of gaussians
    n_splits = 8
    ll_test = np.zeros(max_centers)
    kf = KFold(n_splits=n_splits)
    for train, test in kf.split(data):
        this_ll_test = np.zeros(max_centers)
        for n_centers in range(1, max_centers + 1):
            gmm = GaussianMixture(n_centers)
            gmm.fit(data[train])
            this_ll_test[n_centers-1] = gmm.score(data[test])
        ll_test += this_ll_test/n_splits
    # Plotting...
    plt.figure()
    plt.plot(range(1, max_centers + 1), ll_test)
    plt.xlabel("Number of gaussians in mixture")
    plt.ylabel("log-likelyhood")
    plt.title("Average log-likelyhood after 8-fold cross validation")
    plt.savefig(os.path.join(script_dir, "figures", "cross_validation"))
    if show_plots:
        plt.show()

    # Bootstrap resampling with information criteria to detect number of gaussians
    n_bootstrap_samples = 100
    aic = np.zeros(max_centers)
    bic = np.zeros(max_centers)
    for _ in range(n_bootstrap_samples):
        sample = resample(data)
        this_aic = np.zeros(max_centers)
        this_bic = np.zeros(max_centers)
        for n_centers in range(max_centers):
            gmm = GaussianMixture(n_centers + 1)
            gmm.fit(sample)
            this_aic[n_centers] = gmm.aic(sample)
            this_bic[n_centers] = gmm.bic(sample)
        aic += this_aic/n_bootstrap_samples
        bic += this_bic/n_bootstrap_samples
    # Plotting...
    plt.figure()
    plt.plot(range(1, max_centers + 1), aic)
    plt.plot(range(1, max_centers + 1), bic)
    plt.xlabel("Number of gaussians in mixture")
    plt.ylabel("Information criterion")
    plt.title("Average information criteria over 100 bootstrap samples")
    plt.legend(["AIC", "BIC"])
    plt.savefig(os.path.join(script_dir, "figures", "bootstrapping"))
    if show_plots:
        plt.show()

    # Best number of centers = 5
    n_centers = 5
    significance_level = .01
    gmm = build_gmm(n_centers=5, significance_level=significance_level, data=data)

    scores = gmm.score_samples(data)

    mask = (scores < np.log(significance_level)).astype(int)

    outliers = np.array([[i, j] for c, (i, j) in enumerate(data) if mask[c]])
    normal_data = np.array([[i, j] for c, (i, j) in enumerate(data) if not mask[c]])

    np.savetxt(os.path.join(script_dir, "results", "outliers_gmm.csv"), outliers, delimiter=',')

    # Plotting...
    plt.figure()
    # display predicted scores by the model as a contour plot
    npts = 50000
    xlim = (1.5, 6.5)
    ylim = ((0, 6.5))
    x = np.random.uniform(xlim[0], xlim[1], npts)
    y = np.random.uniform(ylim[0], ylim[1], npts)
    z = np.exp(gmm.score_samples(np.stack([x, y], axis=1)))
    # countourplot
    CS = plt.tricontour(x, y, z, levels=np.logspace(-7, 0, num=8), norm = LogNorm(1e-7, 1), linewidths=.5)
    plt.colorbar(CS, shrink=0.8)
    plt.scatter(normal_data[:, 0], normal_data[:, 1], 1)
    plt.scatter(outliers[:, 0], outliers[:, 1], 7, c="red")
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 10, c="yellow")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["normal point", "outlier"])
    plt.title('Outliers detected by an iterative GMM algorithm')
    plt.savefig(os.path.join(script_dir, "figures", "outliers_gmm"))
    if show_plots:
        plt.show()


    ####################
    ## DBSCAN method: ##
    ####################

    l = len(data)

    distances = np.zeros((l, l))
    for i, p in enumerate(data):
        distances[i, :] = np.sqrt((p[0] - data[:, 0])**2 + (p[1] - data[:, 1])**2)

    # graphically determine min_pts en eps
    distances_sorted = np.sort(distances, axis=1)
    # Plotting...
    plt.figure()
    k_range = range(1, 10, 2)
    for k in k_range:
        plt.semilogy(range(1, l+1), np.sort(distances_sorted[:, k]), linewidth=1)
    plt.legend(["k = {}".format(i) for i in k_range])
    plt.ylabel("Distance")
    plt.xlabel("Point no.")
    plt.title("Sorted distances from points to their N nearest neighbours")
    plt.legend(["N = {}".format(i) for i in range(1, 10, 2)])
    plt.savefig(os.path.join(script_dir, "figures", "nearest_neighbour"))
    if show_plots:
        plt.show()

    min_pts = 5
    eps = np.sort(distances_sorted[:, min_pts])[-20]
    print(eps)

    # helper function for sorting
    def getKey(tup):
        return tup[1]

    # distances[i][j] now contains a tuple of the index of the j-th nearest point to point i 
    # and the distance between that point and point i
    distances = [ sorted(zip(range(l), distances[i]), key=getKey) for i in range(l) ]

    labels = [None for _ in range(l)]

    # helper function to find indices of the points in the eps-neighbourhoud of a point with given index
    def find_neighbours(index, distances=distances, eps=eps):
        N = []
        for t in distances[i]:
            if 0 < t[1] <= eps:
                N.append(t[0])
        return N

    # the dbscan algorithm
    cluster_counter = 0
    for i, _ in enumerate(data):
        if labels[i] == None:
            queu = find_neighbours(i)
            if len(queu) < min_pts:
                labels[i] = -1
            else:
                cluster_counter += 1
                labels[i] = cluster_counter
                while len(queu) > 0:
                    p = queu[-1]
                    if labels[p] == -1:
                        labels[p] = cluster_counter
                    if labels[p] == None:
                        labels[p] = cluster_counter
                        neighbours = find_neighbours(p)
                        if len(neighbours) >= min_pts:
                            queu.extend(neighbours)
                    del queu[-1]

    outliers = np.array([data[i] for i in range(l) if labels[i] == -1])
    normal_data = np.array([data[i] for i in range(l) if labels[i] != -1])

    np.savetxt(os.path.join(script_dir, "results", "outliers_dbscan.csv"), outliers, delimiter=',')

    # Plotting...
    plt.figure()
    plt.scatter(normal_data[:, 0], normal_data[:, 1], 1)
    plt.scatter(outliers[:, 0], outliers[:, 1], 7, c="red")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["normal point", "outlier"])
    plt.title("Outliers detected by the DBSCAN algorithm")
    plt.savefig(os.path.join(script_dir, "figures", "outliers_dbscan"))
    if show_plots:
        plt.show()
