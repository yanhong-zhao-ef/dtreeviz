import json
import math
from collections import defaultdict
from typing import List, Mapping

import numpy as np
from sklearn.utils import compute_class_weight

from dtreeviz.models.shadow_decision_tree import VisualisationNotYetSupportedError
from dtreeviz.models.shadow_decision_tree import ShadowDecTree
from dtreeviz.utils import safe_isinstance
from catboost.core import Pool

class ShadowCatboostDTree(ShadowDecTree):
    # need to figure out what this global config means might not need it
    ROOT_NODE = 0
    TREE_LEAF = -1
    def __init__(self, model,
                 tree_index: int,
                 x_data,
                 y_data,
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None,
                 pool = None,
                 ):
        self.model = model
        self.tree_index = tree_index
        # need to figure out what is the tree to dataframe is doing for xgboost
        # self.tree_to_dataframe = self._get_tree_dataframe()
        # self.children_left = self._calculate_children(self.__class__.LEFT_CHILDREN_COLUMN)
        # self.children_right = self._calculate_children(self.__class__.RIGHT_CHILDREN_COLUMN)
        # self.config = json.loads(self.booster.save_config())
        # self.node_to_samples = None  # lazy initialized
        self.features = None  # lazy initialized
        self.pool = pool
        super().__init__(model, x_data, y_data, feature_names, target_name, class_names)

    def _splits_by_tree_index(self):
        return self.model._get_tree_splits(self.tree_idx, None)

    def _tree_leaf_values_by_tree_index(self):
        return self.model._get_tree_leaf_values(self.tree_idx)

    def _is_oblivious(self):
        """Detect if catboost built symmetric or asymmetric trees"""
        if self.is_fit():
            if self.model.get_all_params()['grow_policy'] == "SymmetricTree":
                return True
            else:
                return False

    @staticmethod
    def _layer_of_symmetric_tree(id):
        """Given the index in a symmetric tree return the layer of which the node id belongs"""
        return int(math.floor(math.log2(id+1)))
        
    def is_fit(self):
        # check if the catboost tree model is fitted
        return self.model.is_fitted()

    def is_classifier(self):
        if safe_isinstance(self.model, "catboost.core.CatBoostClassifier"):
            return True
        elif safe_isinstance(self.model, "catboost.core.CatBoostRegressor"):
            return False
        else:
            raise VisualisationNotYetSupportedError("is_classifier()", "Catboost")

    def get_class_weights(self):
        if self.is_classifier():
            return self.model.get_all_params()['class_weights']

    def get_thresholds(self):
        thresholds = [self.get_node_split(i) for i in range(0, self.nnodes())]
        return thresholds

    def get_features(self):
        if self.features is not None:
            return self.features

        feature_index = [self.get_node_feature(i) for i in range(0, self.nnodes())]
        self.features = np.array(feature_index)
        return self.features

    def criterion(self):
        return VisualisationNotYetSupportedError("criterion()", "Catboost")

    def nclasses(self):
        if not self.is_classifier():
            return 1
        else:
            return len(np.unique(self.y_data))

    def classes(self):
        if self.is_classifier():
            return np.unique(self.y_data)

    def get_node_samples(self):
        # here leaf is subsituted with value of -1 so it won't show up here as it doesn't match the children left/right ids
        if self.node_to_samples is not None:
            return self.node_to_samples
        if self.pool is None:
            prediction_leaves = self.model.calc_leaf_indexes(self.x_data, self.tree_index, self.tree_index+1)
        else:
            prediction_leaves = self.model.calc_leaf_indexes(self.pool, self.tree_index, self.tree_index + 1)
        node_to_samples = defaultdict(list)
        for sample_i, prediction_leaf in enumerate(prediction_leaves):
            prediction_path = self._get_leaf_prediction_path(prediction_leaf)
            for node_id in prediction_path:
                node_to_samples[node_id].append(sample_i)
        self.node_to_samples = node_to_samples
        return node_to_samples

    def get_node_nsamples(self, id):
        return len(self.get_node_samples()[id])

    def _get_leaf_prediction_path(self, leaf):
        # here leaf is subsituted with value of -1 so it won't show up here as it doesn't match the children left/right ids
        prediction_path = [leaf]
        def walk(node_id):
            if node_id != self.__class__.ROOT_NODE:
                try:
                    parent_node = np.where(self.children_left == node_id)[0][0]
                    prediction_path.append(parent_node)
                    walk(parent_node)
                except IndexError:
                    pass

                try:
                    parent_node = np.where(self.children_right == node_id)[0][0]
                    prediction_path.append(parent_node)
                    walk(parent_node)
                except IndexError:
                    pass
        walk(leaf)
        return prediction_path

    def get_children_left(self):
        """
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i.
        """
        # for symmetric tree this is relatively easy where we just need to traverse the from the root node and construct
        # a list of all left nodes by node index
        children_left = []
        if self._is_oblivious():
            for node_id in range(self.nnodes()-self.model.get_tree_leaf_counts()[self.tree_index]):
                if node_id != self.__class__.ROOT_NODE and node_id % 2 != 0:
                    children_left.append(node_id)
        else:
            # need to work through an asymmetric example
            pass
        return np.array(children_left)

    def get_children_right(self):
        """
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i.
        """
        children_right = []
        if self._is_oblivious():
            for node_id in range(self.nnodes()-self.model.get_tree_leaf_counts()[self.tree_index]):
                if node_id != self.__class__.ROOT_NODE and node_id % 2 == 0:
                    children_right.append(node_id)
        else:
            pass
        return np.array(children_right)

    def _get_node_feature_value(self, id):
        splits = self._splits_by_tree_index()
        if self._is_oblivious():
            # assume index starts from 0
            feature_and_value = splits[self._layer_of_symmetric_tree(id)]
        else:
            feature_and_value = splits[id]
        return feature_and_value

    def get_node_split(self, id) -> (int, float):
        feature_and_value = self._get_node_feature_value(id)
        split_value = float(feature_and_value.split(',')[1].split('=')[1])  # replaced by regex later
        return split_value

    def get_node_feature(self, id) -> int:
        feature_and_value = self._get_node_feature_value(id)
        split_feature = str(feature_and_value.split(',')[0])  # replaced by regex later
        return split_feature

    def get_node_nsamples_by_class(self, id):
        pass

    def get_prediction(self, id):
        pass

    def nnodes(self):
        # following the implementation at plot_tree https://github.com/catboost/catboost/blob/ccf8c0fe58737c9f728e14472fa37277ea4db39c/catboost/python-package/catboost/core.py#L3345
        sum_nodes = 0
        if self._is_oblivious():
            splits = self._splits_by_tree_index()
            num_layers = len(splits)+1
            # sum of a geometric sequence with a0=1 r=2
            sum_nodes += 2^num_layers - 1
        else:
            splits = self._splits_by_tree_index()
            sum_nodes += len(splits)
        return sum_nodes

    def _assign_node_ids(self):
        # deprecate this function later, this is just food for thought for now
        if self._is_oblivious():
            pass
            # if we are dealing with the symmetric tree then we can just pose the node ids as from 0 - number of nodes
            # where node idx 0 is the root node of the tree with index 0
            # each tree then get index  (0,1,2) etc.
        else:
            pass
            # for asymmetric tree building this is more complicated which need to take in account of step node
            # depending on step node list which could be like this [(1, 4), (1, 2), (0, 0), (0, 0), (1, 2), (0, 0), (0, 0)]
            # the node ids might be 0,1,4 in this case with other ids for leaf nodes (we might not need the leaf nodes here)
        return None

    def get_node_criterion(self, id):
        return VisualisationNotYetSupportedError("get_node_criterion()", "Catboost")

    def get_feature_path_importance(self, node_list):
        raise VisualisationNotYetSupportedError("get_feature_path_importance()", "Catboost")

    def get_max_depth(self):
        return self.model.get_all_params()["depth"]

    def get_score(self):
        raise VisualisationNotYetSupportedError("get_score()", "Catboost")

    def get_min_samples_leaf(self):
        raise VisualisationNotYetSupportedError("get_min_samples_leaf()", "Catboost")

    def shouldGoLeftAtSplit(self, id, x):
        return x < self.get_node_split(id)

