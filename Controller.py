import base64
import io
import math
import numpy as np
import pandas as pd
import csv

from itertools import compress
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist

from functools import reduce

class Controller:

    __instance__ = None

    def __init__(self):
        if Controller.__instance__ is None:
            self.front_data = None
            self.stk_data = None
            self.req_data = None
            self.dist_array = None
            self.indexes_stack = [] # this appends list of int
            Controller.__instance__ = self

    @staticmethod
    def get_instance():
        """
        Get an instance of the controller class
        :return: unique controller instance
        """
        if not Controller.__instance__:
            Controller()
        return Controller.__instance__

    def front_is_loaded(self):
        return self.front_data is not None

    def stk_is_loaded(self):
        return self.stk_data is not None

    def req_is_loaded(self):
        return self.req_data is not None

    def load_pareto_front_file(self, file_content):
        self.front_data = self.__load_json_file__(file_content, orient='column', dtype=(float, float, str, str))

    def load_stakeholders_file(self, file_content):
        list_of_rows = self.__load_csv_file__(file_content)
        self.stk_data = [x[0] for x in list_of_rows]

    def load_requirements_file(self, file_content):
        self.req_data = self.__load_json_file__(file_content)
        print(self.req_data)

    def __load_json_file__(self, file_content, orient='column', dtype=True):
        content_type, content_string = file_content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            return pd.read_json(io.StringIO(decoded.decode(encoding='utf-8', errors='replace')), orient=orient, dtype=dtype)
        except Exception as e:
            raise Exception('There is a problem with the json file', e.args)

    def __load_csv_file__(self, file_content):
        content_type, content_string = file_content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            return list(csv.reader(io.StringIO(decoded.decode(encoding='utf-8', errors='replace'))))
        except Exception as e:
            raise Exception('There is a problem with the csv file', e.args)

    def get_last_linkage_matrix(self):
        if self.dist_array is None:
            data = self.front_data.copy()
            data['profit'] = self.min_max_normalization(data['profit'])
            data['cost'] = self.min_max_normalization(data['cost'])
            rows = list(data.itertuples(name='Row', index=False))
            # calculate the distance array using the custom distance function specified below.
            self.dist_array = pdist(rows, lambda x, y: self.__distance(x, y))
            self.indexes_stack.append(list(range(0, len(rows))))

        ind = self.indexes_stack[-1]
        dist_matrix = squareform(self.dist_array)
        dist_matrix = squareform(dist_matrix[np.ix_(ind, ind)])
        link_matrix = linkage(dist_matrix)
        return link_matrix

    def __distance(self, x: tuple, y: tuple) -> float:
        """
        Distance between two quasi-optimal solutions
        :param x: tuple (id, profit, cost, reqs, stks)
        :param y: tuple (id, profit, cost, reqs, stks)
        :return: distance between x and y
        """
        # s and t are strings.
        boolean_distance = lambda s, t: sum(np.array(list(s), dtype=int) != np.array(list(t), dtype=int))
        q = (float(x[0]) - float(y[0])) ** 2 + (float(x[1]) - float(y[1])) ** 2
        euc_dist = math.sqrt(q)
        d_req = boolean_distance(x[2], y[2]) / len(x[2])
        d_stk = boolean_distance(x[3], y[3]) / len(x[3])
        return 1 * euc_dist + 1 * d_req + 1 * d_stk

    def get_clusters(self, linkage_matrix, threshold):
        f = fcluster(linkage_matrix, threshold * max(linkage_matrix[:, 2]), 'distance')
        return f

    def get_current_front(self):
        return self.front_data.iloc[self.indexes_stack[-1]]

    def dive_into_cluster(self, threshold, cluster):
        last_linkage_matrix = self.get_last_linkage_matrix()
        cluster_list = self.get_clusters(last_linkage_matrix, threshold)
        mask = cluster_list == cluster
        self.indexes_stack.append(list(compress(self.indexes_stack[-1], mask)))

    def get_depth(self):
        return len(self.indexes_stack)

    def go_back(self):
        self.indexes_stack.pop()

    def get_data_indexes(self):
        return self.indexes_stack[-1]

    def min_max_normalization(self, column):
        return column - column.min() / (column.max() - column.min())

    def get_members_in_cluster(self, threshold, cluster):
        return list(self.get_clusters(self.get_last_linkage_matrix(),threshold)).count(cluster)

    def get_words_from_cluster(self, threshold, cluster):
        # Indexes of the elements that belong to cluster
        clusters = self.get_clusters(self.get_last_linkage_matrix(), threshold)
        cluster_mask = np.array(clusters) == int(cluster)
        indexes = list(compress(self.indexes_stack[-1], cluster_mask))

        # Stakeholders vectors from those elements
        stk_vectors = self.front_data.stks[indexes]
        # Get the stakeholders in stk_vectors
        list_of_str_2_list_of_int = [list(map(int, x)) for x in map(list, stk_vectors)]
        list_or = lambda x, y: [a or b for a, b in zip(x, y)]
        mask = list(map(bool,reduce(lambda x, y: list_or(x, y), list_of_str_2_list_of_int)))
        stk_words = np.array(self.stk_data)[mask].tolist()
        stk_frequency = np.sum(list_of_str_2_list_of_int, axis=0)

        # TODO: this need to change. The two blocks do exactly the same, refactor is needed.
        # Requeriments vectors from those elements
        req_vectors = self.front_data.reqs[indexes]
        # Get the stakeholders in stk_vectors
        list_of_str_2_list_of_int = [list(map(int, x)) for x in map(list, req_vectors)]
        mask = list(map(bool, reduce(lambda x, y: list_or(x, y), list_of_str_2_list_of_int)))
        req_words = ['R{}'.format(i) for i, x in enumerate(mask) if x]
        req_frequency = np.sum(list_of_str_2_list_of_int, axis=0)

        return (stk_words, stk_frequency), (req_words, req_frequency)

