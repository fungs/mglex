u"""
 The types and methods for describing the distribution of label-type data. We use a modified Naive Bayesian Model
 which  considers hierarchical labels using a weighting scheme. However, the weighting of the individual labels
 is handled externally which leaves much freedom for shaping the actual PMF. Weights could for instance be set by
 consideration phylogenetic distances.
"""

__author__ = "johannes.droege@uni-duesseldorf.de"

from .. import common, types
import numpy as np
# from collections import deque
from sys import argv, exit, stdin, stdout, stderr, exit

# label data type
label_index_type = np.uint32  # TODO: check range
support_type = np.float64  # TODO: check range


class Context(object):
    """Container for information which is shared between Data and Model"""

    def __init__(self):
        self.labels = []
        self.levelindex = []

    def __str__(self):
        return str(self.labels)

    @property
    def num_features(self):
        return len(self.labels)


class Data(object):  # TODO: use deque() for large append-only lists
    """Data container for hierarchical label type data"""

    def __init__(self, context=Context()):
        self.context = context
        self._label_mapping = common.NestedCountIndex()
        self._labels = []  # TODO: operate on separate deque objects?
        self.labels = None
        self.mean_support = None

        # initialize indices and labels from context
        for l in context.labels:
            self._label_mapping[l]

        self.num_features = len(self._label_mapping)

    def deposit(self, features):
        features = tuple((self._label_mapping[path], support_type(support)) for path, support in features)
        self._labels.append(features)

    def parse(self, inseq):  # TODO: add load_data from generic with data-specific parse_line function
        for line in inseq:
            feature_list = []
            if line:
                for entry in line.split(" "):
                    # print(entry)
                    path, support = entry.split(":", 2)[:2]
                    path = path.split(".")
                    feature_list.append((path, support))
            self.deposit(feature_list)
        return self.prepare()

    def prepare(self):
        # calculate internal (ordered) label index
        del self.context.labels[:]  # clear context and rewrite in correct order

        index2index = np.empty(len(self._label_mapping), dtype=label_index_type)
        newindex = 0
        for levelindex, level in enumerate(self._label_mapping.items_nested()):  # skip root level?
            for path, oldindex in sorted(level):
                index2index[oldindex] = newindex
                self.context.labels.append(path)
                newindex += 1
            self.context.levelindex.append(newindex)  # whenever a deeper level is reached, the index is saved

        # reset temporary index which is not needed any more
        self.num_features = len(self._label_mapping)
        self._label_mapping = common.NestedCountIndex()

        # replace old by new indices and create numpy arrays inplace TODO: make this two real 2d arrays
        support_total_sum = 0.
        for i, features in enumerate(self._labels):
            index_col = np.empty(len(features), dtype=label_index_type)
            support_col = np.empty(len(features), dtype=support_type)
            for j, (index_orig, support) in enumerate(features):
                index_col[j] = index2index[index_orig]
                support_col[j] = support

            # not necessary but order by increasing index for better debug output
            si = index_col.argsort()
            index_col = index_col[si]
            support_col = support_col[si]

            self._labels[i] = (index_col, support_col)

            support_total_sum += support_col.sum()

        del index2index  # runs out of scope and should be garbage-collected anyway

        self.labels = self._labels
        self._labels = []

        self.mean_support = support_total_sum/self.num_data

        # print(self.context.levelindex)
        return self

    @property
    def num_data(self):
        return len(self.labels)

    def __len__(self):
        return self.num_data


class Model(object):
    """A variant of a Naïve Bayes classifier model for hierarchical labels"""

    def __init__(self, params, context, initialize=True, pseudocount=True):
        assert type(context) == Context
        assert params.shape[0] == context.num_features

        self.context = context
        self.params = np.array(params, dtype=support_type)  # TODO: use large unsigned integer first, then cut down
        self.labels = context.labels[:]
        self._levelindex = np.asarray(context.levelindex, dtype=label_index_type)
        self.levelsum = np.empty(params.shape, dtype=support_type)
        self._pseudocount = pseudocount
        self._log_frequencies = None

        if initialize:
            self.update()

    def update_context(self):  # reorder and resize matrix
        # print(self.labels, file=stderr)
        # print(self.context.labels, file=stderr)
        if self.labels != self.context.labels:
            print("updating context!", file=stderr)
            mapping = dict(zip(self.context.labels, range(self.context.num_features)))
            newparams = np.zeros(shape=(self.context.num_features, self.num_components), dtype=support_type)
            newparams[[mapping[i] for i in self.labels]] = self.params[:]
            self.params = newparams
            self._levelindex = np.asarray(self.context.levelindex, dtype=label_index_type)
            self.labels = self.context.labels[:]
            self.update()

    def update(self):  # update parameters without change of feature set
        # print("params")
        # print(self.params.shape)
        # print(self.params, file=stderr)
        for i, j in zip(self._levelindex, self._levelindex[1:]):  # TODO: advanced slicing with np.r_?
            self.levelsum[i:j] = self.params[i:j].sum(axis=0)
        # print("levelsum")
        # print(self.levelsum.shape)
        # print(self.levelsum)
        self._log_frequencies = np.log(self.params / self.levelsum)
        return False  # indicates whether a dimension change occurred

    def log_likelihood(self, data):  # TODO: check
        loglike = np.empty((len(data), self.num_components), dtype=types.logprob_type)
        for i, (indexcol, supportcol) in enumerate(data.labels):  # TODO: vectorize 3d?

            if not indexcol.size:  # no label == no observation == perfect fit
                loglike[i] = 0
                continue

            #denominator = supportcol.sum()  # replace by relative weights in data and length normalization
            #assert np.all(denominator != 0.0)
            ll = np.dot(supportcol, self._log_frequencies[indexcol])

            # if not np.all(numerator != 0.):
            #     print(pretty_probvector(numerator), file=stderr)
            #     print(pretty_probvector(denominator), file=stderr)
            #     print(pretty_probvector(indexcol), file=stderr)
            #     print(pretty_probvector(supportcol), file=stderr)

            # if not denominator.all():  # TODO: turn back into assertion
            # print("datum: %i\n  numerator: %s /\n denominator %s" % (i, numerator, denominator), file=stderr)
            # print("datum %i" % i, file=stderr)
            # si = indexcol.argsort()
            # indexcol_sorted = indexcol[si]
            # print(indexcol, file=stderr)
            # print([self.labels[i] for i in indexcol], file=stderr)
            # print(supportcol, file=stderr)
            # print(self.params[indexcol], file=stderr)
            # print(self.levelsum[indexcol], file=stderr)
            #    exit(1)

            #print(pretty_probvector(numerator),file=stderr)
            #print(pretty_probvector(denominator), file=stderr)

            # if self._pseudocount:
            #     probs = (numerator+1)/(denominator+1)
            # else:
            #     probs = numerator/denominator

            # ll = np.log(probs)  # TODO: or log - log
            loglike[i] = ll/data.mean_support

        assert np.all(loglike <= .0)
        return loglike

    def get_labels(self, indices=None):
        if not indices:
            for i in self.params.argmax(axis=0):
                yield str(i)
        else:
            for i in self.params[indices].argmax(axis=0):
                yield str(i)

    def maximize_likelihood(self, data, responsibilities, weights, cmask=None):  # TODO: unite responsibilities and weights and change everywhere
        # TODO: input as combined weights, not responsibilities and data.sizes

        if not (cmask is None or cmask.shape == () or np.all(cmask)):  # cluster reduction
            responsibilities = responsibilities[:, cmask]
            self.params = self.params[:, cmask]

        if self._pseudocount:
            self.params[:] = 1
        else:
            self.params[:] = 0  # zero out values

        for res, (index_col, support_col) in zip(responsibilities, data.labels):
            # common.print_probvector(res, file=stderr)
            # print(index_col, file=stderr)
            # print(support_col, file=stderr)
            # common.print_probmatrix(np.dot(support_col[:, np.newaxis], res[np.newaxis, :]), file=stderr)
            # print(self.params.shape)
            # print(support_col[:, np.newaxis].shape)
            # print(res.T)
            # print(np.vdot(support_col, res).shape)
            # print(res, file=stderr)
            # print(support_col[:, np.newaxis], file=stderr)
            # print(np.dot(support_col[:, np.newaxis], res[np.newaxis, :]), file=stderr)
            self.params[index_col] += np.dot(support_col[:, np.newaxis], res[np.newaxis, :])  # TODO: check shape match
            # print(index_col, file=stderr)
            # print([self.labels[i] for i in index_col], file=stderr)
            # print(self.params[index_col], file=stderr)

        weights_combined = responsibilities * weights

        dimchange = self.update()  # create cache for likelihood calculations
        ll = self.log_likelihood(data)
        # common.write_probmatrix(ll, file=stdout)
        std_per_class = np.sqrt(common.weighted_variance(ll, weights_combined))
        weight_per_class = weights_combined.sum(axis=0, dtype=types.large_float_type)
        relative_weight_per_class = np.asarray(weight_per_class / weight_per_class.sum(), dtype=types.prob_type)
        combined_std = np.dot(std_per_class, relative_weight_per_class)
        # stderr.write("Weighted stdev was: %s\n" % common.pretty_probvector(std_per_class))
        # stderr.write("Weighted combined stdev was: %.2f\n" % combined_std)
        stderr.write("LOG %s: class likelihood standard deviation is %.2f\n" % (self._short_name, combined_std))
        self.stdev = combined_std
        return dimchange, ll

    @property
    def num_components(self):
        return self.params.shape[1]

    @property
    def num_features(self):
        return self.params.shape[0]

    @property
    def names(self):
        return list(self.get_labels())

    _short_name = "LD_model"


def load_model(instream):
    all_clists = []
    # samples = input.next().rstrip().split("\t")
    for line in instream:
        if not line or line[0] == "#":
            continue
        clist = line.rstrip().split("\t")
        if clist:
            all_clists.append(list(map(int, clist)))
    return Model(all_clists)


def load_data(input, samples):  # TODO: add load_data from generic with data-specific parse_line function
    store = Data(samples)
    for line in input:
        if not line or line[0] == "#":  # skip empty lines and comments
            continue
        seqname, coverage_field = line.rstrip().split("\t", 2)[:2]
        feature_list = []
        for sample_group in coverage_field.split(" "):
            sample_name, coverage = sample_group.split(":", 2)[:2]
            coverage = list(map(int, coverage.split(",")))  # TODO: use sparse numpy objects...
            feature_list.append((sample_name, coverage))
        store.deposit(seqname, feature_list)
    return store.prepare()


def empty_model(cluster_number, context, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    params = np.zeros(shape=(context.num_features, cluster_number), dtype=support_type)
    return Model(params, context, initialize=False, **kwargs)


def random_model(cluster_number, context, low, high, **kwargs):
    assert cluster_number > 0
    assert type(context) == Context
    params = np.random.random_integers(low=low, high=high, size=(context.num_features, cluster_number))
    return Model(params, context, **kwargs)


def load_data_file(filename, **kwargs):
    d = Data(**kwargs)
    return common.load_data_file(filename, d)
