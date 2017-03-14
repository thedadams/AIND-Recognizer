import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, words: dict, hwords: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=None, verbose=False):
        self.words = words
        self.hwords = hwords
        self.sequences = words[this_word]
        self.X, self.lengths = hwords[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states, xs=None, lengths=None):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if xs is None:
            xs = self.X
        if lengths is None:
            lengths = self.lengths
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(xs, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_model_score = float("inf")
        for n in range(self.min_n_components, self.max_n_components + 1):
            model = None
            this_score = 0.0
            try:
                model = self.base_model(n)
                if model is None:
                    break
                this_score = -2.0 * model.score(self.X, self.lengths) + len(self.X[0]) * np.log(len(self.lengths))
                if this_score < best_model_score:
                    best_model_score = this_score
                    best_model = model
            except Exception as e:
                if self.verbose:
                    print(e)
                    print("There was a problem splitting the data.")
            if model is None:
                break
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def train_test_data_split(self, train_indexes, test_indexes):
        train_x = []
        test_x = []
        train_l = []
        test_l = []
        global_index = 0
        for index, length in enumerate(self.lengths):
            if index in train_indexes:
                for j in range(length):
                    train_x.append(self.X[global_index + j])
                train_l.append(length)
            else:
                for j in range(length):
                    test_x.append(self.X[global_index + j])
                test_l.append(length)
            global_index += length
        return train_x, train_l, test_x, test_l


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        best_model = None
        best_model_score = float("-inf")
        split_method = KFold(2)
        for n in range(self.min_n_components, self.max_n_components + 1):
            model = None
            this_score = 0.0
            try:
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    train_x, train_lengths, test_x, test_lengths = self.train_test_data_split(cv_train_idx, cv_test_idx)
                    model = self.base_model(n, train_x, train_lengths)
                    if model is None:
                        break
                    this_score += model.score(test_x, test_lengths)
                if this_score > best_model_score:
                    best_model_score = this_score
                    best_model = model
            except Exception as e:
                if self.verbose:
                    print(e)
                    print("There was a problem splitting the data.")
            if model is None:
                break
        return best_model
