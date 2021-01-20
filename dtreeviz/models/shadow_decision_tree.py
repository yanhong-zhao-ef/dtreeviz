from abc import ABC, abstractmethod
from collections import Sequence
from numbers import Number
from typing import List, Tuple, Mapping

import numpy as np
import pandas as pd
import sklearn
import xgboost


class ShadowDecTree(ABC):
    """
    This tree shadows a decision tree as constructed by scikit-learn's and XGBoost's
    DecisionTree(Regressor|Classifier). As part of build process, the
    samples considered at each decision node or at each leaf node are
    saved as a big dictionary for use by the nodes.

    The decision trees for classifiers and regressors from scikit-learn and XGBoost
    are built for efficiency, not ease of tree walking. This class
    is intended as a way to wrap all of that information in an easy to use
    package.

    Field leaves is list of shadow leaf nodes. Field internal is list of shadow non-leaf nodes.
    Field root is the shadow tree root.
    """

    def __init__(self,
                 tree_model,
                 x_data: (pd.DataFrame, np.ndarray),
                 y_data: (pd.Series, np.ndarray),
                 feature_names: List[str] = None,
                 target_name: str = None,
                 class_names: (List[str], Mapping[int, str]) = None):
        """
        Parameters
        ----------
        :param tree_model: sklearn.tree.DecisionTreeRegressor, sklearn.tree.DecisionTreeClassifier, xgboost.core.Booster
            The decision tree to be interpreted
        :param x_data: pd.DataFrame, np.ndarray
            Features values on which the shadow tree will be build.
        :param y_data: pd.Series, np.ndarray
            Target values on which the shadow tree will be build.
        :param feature_names: List[str]
            Features' names
        :param target_name: str
            Target's name
        :param class_names: List[str], Mapping[int, str]
            Class' names (in case of a classifier)

        """

        self.tree_model = tree_model
        if not self.is_fit():
            raise Exception(f"Model {tree_model} is not fit.")

        self.feature_names = feature_names
        self.target_name = target_name
        self.class_names = class_names
        # self.class_weight = self.get_class_weight()
        self.x_data = ShadowDecTree._get_x_data(x_data)
        self.y_data = ShadowDecTree._get_y_data(y_data)
        # self.node_to_samples = self.get_node_samples()
        self.root, self.leaves, self.internal = self._get_tree_nodes()
        if class_names:
            self.class_names = self._get_class_names()

    @abstractmethod
    def is_fit(self) -> bool:
        """Checks if the tree model is already trained."""
        pass

    @abstractmethod
    def is_classifier(self) -> bool:
        """Checks if the tree model is a classifier."""
        pass

    @abstractmethod
    def get_class_weights(self):
        """Returns the tree model's class weights."""
        pass

    @abstractmethod
    def get_thresholds(self) -> np.ndarray:
        """Returns split node/threshold values for tree's nodes.

        Ex. threshold[i] holds the split value/threshold for the node i.
        """
        pass

    @abstractmethod
    def get_features(self) -> np.ndarray:
        """Returns feature indexes for tree's nodes.

        Ex. features[i] holds the feature index to split on
        """
        pass

    @abstractmethod
    def criterion(self) -> str:
        """Returns the function to measure the quality of a split.

        Ex. Gini, entropy, MSE, MAE
        """
        pass

    @abstractmethod
    def get_class_weight(self):
        """
        TOOD - to be compared with get_class_weights
        :return:
        """
        pass

    @abstractmethod
    def nclasses(self) -> int:
        """Returns the number of classes.

        Ex. 2 for binary classification or 1 for regression.
        """
        pass

    @abstractmethod
    def classes(self) -> np.ndarray:
        """Returns the tree's classes values in case of classification.

        Ex. [0,1] in class of a binary classification
        """
        pass

    @abstractmethod
    def get_node_samples(self):
        """Returns dictionary mapping node id to list of sample indexes considered by
        the feature/split decision.
        """
        pass

    @abstractmethod
    def get_node_nsamples(self, id):
        """Returns number of samples for a specific node id."""
        pass

    @abstractmethod
    def get_children_left(self) -> np.ndarray:
        """Returns the node ids of the left child node.

        Ex. children_left[i] holds the node id of the left child of node i.
        """
        pass

    @abstractmethod
    def get_children_right(self) -> np.ndarray:
        """Returns the node ids of the right child node.

        Ex. children_right[i] holds the node id of the right child of node i.
        """
        pass

    @abstractmethod
    def get_node_split(self, id) -> (int, float):
        """Returns node split value.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def get_node_feature(self, id) -> int:
        """Returns feature index from node id.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def get_node_nsamples_by_class(self, id):
        """For a classification decision tree, returns the number of samples for each class from a specified node.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def get_prediction(self, id):
        """Returns the constant prediction value for node id.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def nnodes(self) -> int:
        "Returns the number of nodes (internal nodes + leaves) in the tree."
        pass

    @abstractmethod
    def get_node_criterion(self, id):
        """Returns the impurity (i.e., the value of the splitting criterion) at node id.

        Parameters
        ----------
        id : int
            The node id.
        """
        pass

    @abstractmethod
    def get_feature_path_importance(self, node_list):
        """Returns the feature importance for a list of nodes.

        The node feature importance is calculated based on only the nodes from that list, not based on entire tree nodes.

        Parameters
        ----------
        node_list : List
            The list of nodes.
        """
        pass

    @abstractmethod
    def get_max_depth(self) -> int:
        """The max depth of the tree."""
        pass

    @abstractmethod
    def get_score(self) -> float:
        """
        For classifier, returns the mean accuracy.
        For regressor, returns the R^2.
        """
        pass

    @abstractmethod
    def get_min_samples_leaf(self) -> (int, float):
        """Returns the minimum number of samples required to be at a leaf node."""
        pass

    @abstractmethod
    def shouldGoLeftAtSplit(self, id, x):
        """Return true if it should go to the left node child based on node split criterion and x value"""
        pass

    def is_categorical_split(self, id) -> bool:
        """Checks if the node split is a categorical one.

        This method needs to be overloaded only for shadow tree implementation which contain categorical splits,
        like Spark.
        """
        return False

    def get_split_node_heights(self, X_train, y_train, nbins) -> Mapping[int, int]:
        class_values = np.unique(y_train)
        node_heights = {}
        # print(f"Goal {nbins} bins")
        for node in self.internal:
            # print(node.feature_name(), node.id)
            X_feature = X_train[:, node.feature()]
            overall_feature_range = (np.min(X_feature), np.max(X_feature))
            # print(f"range {overall_feature_range}")
            r = overall_feature_range[1] - overall_feature_range[0]

            bins = np.linspace(overall_feature_range[0],
                               overall_feature_range[1], nbins + 1)
            # bins = np.arange(overall_feature_range[0],
            #                  overall_feature_range[1] + binwidth, binwidth)
            # print(f"\tlen(bins)={len(bins):2d} bins={bins}")
            X, y = X_feature[node.samples()], y_train[node.samples()]
            X_hist = [X[y == cl] for cl in class_values]
            height_of_bins = np.zeros(nbins)
            for i, _ in enumerate(class_values):
                hist, foo = np.histogram(X_hist[i], bins=bins, range=overall_feature_range)
                # print(f"class {cl}: goal_n={len(bins):2d} n={len(hist):2d} {hist}")
                height_of_bins += hist
            node_heights[node.id] = np.max(height_of_bins)

            # print(f"\tmax={np.max(height_of_bins):2.0f}, heights={list(height_of_bins)}, {len(height_of_bins)} bins")
        return node_heights

    def predict(self, x: np.ndarray) -> Tuple[Number, List]:
        """
        Given an x - vector of features, return predicted class or value based upon this tree.
        Also return path from root to leaf as 2nd value in return tuple.

        Recursively walk down tree from root to appropriate leaf by comparing feature in x to node's split value.

        :param
        x: np.ndarray
            Feature vector to run down the tree to a  leaf.
        """

        def walk(t, x, path):
            if t is None:
                return None
            path.append(t)
            if t.isleaf():
                return t
            # if x[t.feature()] < t.split():
            # print(f"shadow node id, x {t.id} , {t.feature()}")
            if self.shouldGoLeftAtSplit(t.id, x[t.feature()]):
                return walk(t.left, x, path)
            return walk(t.right, x, path)

        path = []
        leaf = walk(self.root, x, path)
        return leaf.prediction(), path

    def tesselation(self):
        """
        Walk tree and return list of tuples containing a leaf node and bounding box list of(x1, y1, x2, y2) coordinates.
        """
        bboxes = []

        def walk(t, bbox):
            if t is None:
                return None
            # print(f"Node {t.id} bbox {bbox} {'   LEAF' if t.isleaf() else ''}")
            if t.isleaf():
                bboxes.append((t, bbox))
                return t
            # shrink bbox for left, right and recurse
            s = t.split()
            if t.feature() == 0:
                walk(t.left, (bbox[0], bbox[1], s, bbox[3]))
                walk(t.right, (s, bbox[1], bbox[2], bbox[3]))
            else:
                walk(t.left, (bbox[0], bbox[1], bbox[2], s))
                walk(t.right, (bbox[0], s, bbox[2], bbox[3]))

        # create bounding box in feature space (not zeroed)
        f1_values = self.x_data[:, 0]
        f2_values = self.x_data[:, 1]
        overall_bbox = (np.min(f1_values), np.min(f2_values),  # x,y of lower left edge
                        np.max(f1_values), np.max(f2_values))  # x,y of upper right edge
        walk(self.root, overall_bbox)

        return bboxes

    def get_leaf_sample_counts(self, min_samples=0, max_samples=None):
        """
        Get the number of samples for each leaf.

        There is the option to filter the leaves with samples between min_samples and max_samples.

        Parameters
        ----------
        min_samples: int
            Min number of samples for a leaf
        max_samples: int
            Max number of samples for a leaf

        :return: tuple
            Contains a numpy array of leaf ids and an array of leaf samples
        """

        max_samples = max_samples if max_samples else max([node.nsamples() for node in self.leaves])
        leaf_samples = [(node.id, node.nsamples()) for node in self.leaves if
                        min_samples <= node.nsamples() <= max_samples]
        x, y = zip(*leaf_samples)
        return np.array(x), np.array(y)

    def get_leaf_criterion(self):
        """Get criterion for each leaf

        For classification, supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
        For regression, supported criteria are “mse”, “friedman_mse”, “mae”.
        """

        leaf_criterion = [(node.id, node.criterion()) for node in self.leaves]
        x, y = zip(*leaf_criterion)
        return np.array(x), np.array(y)

    def get_leaf_sample_counts_by_class(self):
        """ Get the number of samples by class for each leaf.

        :return: tuple
            Contains a list of leaf ids and a two lists of leaf samples(one for each class)
        """

        leaf_samples = [(node.id, node.n_sample_classes()[0], node.n_sample_classes()[1]) for node in self.leaves]
        index, leaf_sample_0, leaf_samples_1 = zip(*leaf_samples)
        return index, leaf_sample_0, leaf_samples_1

    def _get_class_names(self):
        if self.is_classifier():
            if isinstance(self.class_names, dict):
                return self.class_names
            elif isinstance(self.class_names, Sequence):
                return {i: n for i, n in enumerate(self.class_names)}
            else:
                raise Exception(f"class_names must be dict or sequence, not {self.class_names.__class__.__name__}")

    def _get_tree_nodes(self):
        # use locals not args to walk() for recursion speed in python
        leaves = []
        internal = []  # non-leaf nodes
        children_left = self.get_children_left()
        children_right = self.get_children_right()

        def walk(node_id):
            if children_left[node_id] == -1 and children_right[node_id] == -1:  # leaf
                t = ShadowDecTreeNode(self, node_id)
                leaves.append(t)
                return t
            else:  # decision node
                left = walk(children_left[node_id])
                right = walk(children_right[node_id])
                t = ShadowDecTreeNode(self, node_id, left, right)
                internal.append(t)
                return t

        root_node_id = 0
        root = walk(root_node_id)
        return root, leaves, internal

    @staticmethod
    def _get_x_data(x_data):
        if isinstance(x_data, pd.DataFrame):
            x_data = x_data.values  # We recommend using :meth:`DataFrame.to_numpy` instead.
        return x_data

    @staticmethod
    def _get_y_data(y_data):
        if isinstance(y_data, pd.Series):
            y_data = y_data.values
        return y_data

    @staticmethod
    def get_shadow_tree(tree_model, x_data, y_data, feature_names, target_name, class_names=None, tree_index=None):
        if isinstance(tree_model, ShadowDecTree):
            return tree_model
        elif isinstance(tree_model, (sklearn.tree.DecisionTreeRegressor, sklearn.tree.DecisionTreeClassifier)):
            from dtreeviz.models import sklearn_decision_trees
            return sklearn_decision_trees.ShadowSKDTree(tree_model, x_data, y_data, feature_names,
                                                        target_name, class_names)
        elif isinstance(tree_model, xgboost.core.Booster):
            from dtreeviz.models import xgb_decision_tree
            return xgb_decision_tree.ShadowXGBDTree(tree_model, tree_index, x_data, y_data,
                                                    feature_names, target_name, class_names)
        else: raise ValueError(f"Tree model must be in (DecisionTreeRegressor, DecisionTreeClassifier, xgboost.core.Booster, but was {tree_model.__class__.__name__}")


class ShadowDecTreeNode():
    """
    A node in a shadow tree. Each node has left and right pointers to child nodes, if any.
    As part of tree construction process, the samples examined at each decision node or at each leaf node are
    saved into field node_samples.
    """

    def __init__(self, shadow_tree: ShadowDecTree, id: int, left=None, right=None):
        self.shadow_tree = shadow_tree
        self.id = id
        self.left = left
        self.right = right

    def split(self) -> (int, float):
        """Returns the split/threshold value used at this node."""

        return self.shadow_tree.get_node_split(self.id)

    def feature(self) -> int:
        """Returns feature index used at this node"""

        return self.shadow_tree.get_node_feature(self.id)

    def feature_name(self) -> (str, None):
        """Returns the feature name used at this node"""

        if self.shadow_tree.feature_names is not None:
            return self.shadow_tree.feature_names[self.feature()]
        return None

    def samples(self) -> List[int]:
        """Returns samples indexes from this node"""

        return self.shadow_tree.get_node_samples()[self.id]

    def nsamples(self) -> int:
        """
        Return the number of samples associated with this node. If this is a leaf node, it indicates the samples
        used to compute the predicted value or class . If this is an internal node, it is the number of samples used
        to compute the split point.
        """

        return self.shadow_tree.get_node_nsamples(self.id)

    # TODO
    # implementation should happen in shadow tree implementations, we already have methods for this
    # this implementation will work also for validation dataset.... think how to separate model interpretation using training vs validation dataset.
    def n_sample_classes(self):
        """Used for binary classification only.

        Returns the sample count values for each classes.
        """

        samples = np.array(self.samples())
        if samples.size == 0:
            return [0, 0]

        node_y_data = self.shadow_tree.y_data[samples]
        unique, counts = np.unique(node_y_data, return_counts=True)

        if len(unique) == 2:
            return [counts[0], counts[1]]
        elif len(unique) == 1:  # one node can contain samples from only on class
            if unique[0] == 0:
                return [counts[0], 0]
            elif unique[0] == 1:
                return [0, counts[0]]

    def criterion(self):
        return self.shadow_tree.get_node_criterion(self.id)

    def split_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the list of indexes to the left and the right of the split value."""

        samples = np.array(self.samples())
        if len(samples) == 0:
            # there are no samples in this node
            left = []
            right = []
        else:
            node_X_data = self.shadow_tree.x_data[samples, self.feature()]
            split = self.split()
            left = np.nonzero(node_X_data < split)[0]
            right = np.nonzero(node_X_data >= split)[0]
        return left, right

    def isleaf(self) -> bool:
        return self.left is None and self.right is None

    def isclassifier(self) -> bool:
        return self.shadow_tree.is_classifier()

    def is_categorical_split(self) -> bool:
        return self.shadow_tree.is_categorical_split(self.id)

    def prediction(self) -> (Number, None):
        """Returns leaf prediction.

        If the node is an internal node, returns None
        """

        if not self.isleaf():
            return None
        # if self.isclassifier():
        #     counts = self.shadow_tree.get_prediction_value(self.id)
        #     return np.argmax(counts)
        # else:
        #     return self.shadow_tree.get_prediction_value(self.id)
        return self.shadow_tree.get_prediction(self.id)

    def prediction_name(self) -> (str, None):
        """
        If the tree model is a classifier and we know the class names, return the class name associated with the
        prediction for this leaf node.

        Return prediction class or value otherwise.
        """

        if self.isclassifier():
            if self.shadow_tree.class_names is not None:
                return self.shadow_tree.class_names[self.prediction()]
        return self.prediction()

    def class_counts(self) -> (List[int], None):
        """
        If this tree model is a classifier, return a list with the count associated with each class.
        """

        if self.isclassifier():
            if self.shadow_tree.get_class_weight() is None:
                # return np.array(np.round(self.shadow_tree.tree_model.tree_.value[self.id][0]), dtype=int)
                return np.array(np.round(self.shadow_tree.get_node_nsamples_by_class(self.id)), dtype=int)
            else:
                return np.round(
                    self.shadow_tree.get_node_nsamples_by_class(self.id) / self.shadow_tree.get_class_weights()).astype(
                    int)
        return None

    def __str__(self):
        if self.left is None and self.right is None:
            return "<pred={value},n={n}>".format(value=round(self.prediction(), 1), n=self.nsamples())
        else:
            return "({f}@{s} {left} {right})".format(f=self.feature_name(),
                                                     s=round(self.split(), 1),
                                                     left=self.left if self.left is not None else '',
                                                     right=self.right if self.right is not None else '')


class VisualisationNotYetSupportedError(Exception):
    def __init__(self, method_name, model_name):
        super().__init__(f"{method_name} is not implemented yet for {model_name}")
