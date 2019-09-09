import numpy as np
import pandas as pd
import math
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, fcluster
from scipy.spatial.distance import squareform, pdist
from matplotlib import pyplot as plt
import seaborn as sns

#-------------------------------------------------------------------------------
# input data in csv format where each row is a solution from the Pareto Front,
# with columns in order:
# 0. id: integer
# 1. profit: float
# 2. cost: float
# 3. reqs: string of boolean of length equal to the number of requirements.
# 4. stks: string of boolean of length equal to the number of stakeholders.
data = pd.read_csv("ParetoFront.csv", sep=",")
#-------------------------------------------------------------------------------

def boolean_distance(str1: str, str2: str) -> int:
    '''
    Return distance between two lists of boolean.
    :param str1: First list of boolean
    :param str2: Second list of boolean
    :return: number of elements in common between the two input strings
    '''
    r1_boolean = np.array(list(str1), dtype=int)
    r2_boolean = np.array(list(str2), dtype=int)
    return sum(r1_boolean != r2_boolean)


def distance(sol1: list, sol2: list) -> float:
    '''
    Distance between two solutions from the Pareto front, using profit, cost, s
    :param sol1: first solution
    :param sol2: second solution
    :return: distance
    '''
    c1 = sol1[1] / max_profit
    c2 = sol2[1] / max_profit
    p1 = sol1[2] / max_cost
    p2 = sol2[2] / max_cost
    euc_dist = math.sqrt((c1 - c2)**2 + (p1 - p2)**2)
    d_req = boolean_distance(sol1[3], sol2[3]) / len(sol1[3])
    d_stk = boolean_distance(sol1[4], sol2[4]) / len(sol1[4])
    return 1*euc_dist + 1*d_req + 1*d_stk


def count_booleans(df: pd.DataFrame):
    '''

    :param df:
    :return:
    '''
    reqs = np.zeros(len(df.iloc[0][3]))
    stks = np.zeros(len(df.iloc[0][4]))
    for i, row in df.iterrows():
        reqs = reqs + np.array(list(row[3]), dtype=int)
        stks = stks + np.array(list(row[4]), dtype=int)
    d = dict()
    d["reqs"] = reqs
    d["stks"] = stks
    return d

def cluster_count(data, clusters, file):
    clusters_counts = dict()
    for c in list(set(clusters)):
        cluster_elements = data[data.cluster == c]
        count = count_booleans(cluster_elements)
        clusters_counts[c] = count
        ccount = len(cluster_elements.profit)
        pm = cluster_elements.profit.mean()
        cm = cluster_elements.cost.mean()
        psd = cluster_elements.profit.std()
        csd = cluster_elements.cost.std()
        print(
            "Cluster{}: Cantidad: {}, Profit Mean: {}, Profit SD: {}, Cost Mean: {}, Cost SD: {}".format(c, ccount, pm,
                                                                                                         psd, cm, csd))
    clusters_counts = pd.DataFrame(clusters_counts)
    clusters_counts.to_csv(file)
    return clusters_counts

max_cost = max(data["cost"])
max_profit = max(data["profit"])

dist_matrix = pdist(data.values, distance)
linkage_matrix = linkage(squareform(dist_matrix), method="complete")
clusters = fcluster(linkage_matrix, 4, criterion='maxclust')
data["cluster"] = clusters
data.to_csv("clusters.csv")
clusters_counts = cluster_count(data, clusters, "cluster_count.csv")

## Plotting----------------------------------------


def plot_histograms(clusters, clusters_counts, dim="reqs"):
    i = 1
    for c in list(set(clusters)):
        count_req = clusters_counts[c][dim]
        freq_vector = []
        for r in range(0, len(count_req)):
            freq_vector.extend([r] * int(count_req[r]))
        plt.subplot(2, 2, i)
        plt.title("Grupo {}".format(c))
        sns.distplot(freq_vector, bins=len(count_req), kde=False)
        i = i + 1
    sns.despine()
    plt.show()

fig = plt.figure(figsize=(25, 10))
dn = dendrogram(linkage_matrix)
plt.show()

plot_histograms(clusters, clusters_counts, dim="reqs")
plot_histograms(clusters, clusters_counts, dim="stks")
sns.lmplot( x="profit", y="cost", data=data, fit_reg=False, hue='cluster', legend=False)
plt.show()

## Filter 2
filtered_clusters = data[data.cluster == 2]
filtered_clusters = filtered_clusters.drop('cluster', 1)
dist_matrix = pdist(filtered_clusters.values, distance)
linkage_matrix = linkage(squareform(dist_matrix), method="complete")
clusters = fcluster(linkage_matrix, 4, criterion='maxclust')
filtered_clusters["cluster"] = clusters
filtered_clusters.to_csv("subclusters.csv")
clusters_counts = cluster_count(filtered_clusters, clusters, "subcluster_count.csv")

fig = plt.figure(figsize=(25, 10))
dn = dendrogram(linkage_matrix)
plt.show()

plot_histograms(clusters, clusters_counts, dim="reqs")
plot_histograms(clusters, clusters_counts, dim="stks")
sns.lmplot( x="profit", y="cost", data=filtered_clusters, fit_reg=False, hue='cluster', legend=False)
plt.show()