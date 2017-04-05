import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self, selector_func=None):
        """For BIC and DIC, we can use this select method by passing the corresponding
           selector_func.

            params:
                selector_func - a function with signature func(model) for scoring purposes

            returns:
                the best model for the selection method
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # If selector_func is None, then we assume this is SelectorConstant
        if selector_func is None:
            return self.base_model(self.n_constant)

        best_model = None
        best_model_score = float("-inf")
        for n in range(self.min_n_components, self.max_n_components + 1):
            model = None
            this_score = 0.0
            try:
                # Get the base model
                model = self.base_model(n)
                # If the model initialization failed because there are too many components
                # then we stop.
                if model is None:
                    break
                # Get the BIC for this model.
                this_score = selector_func(model)
                # If this model is better than the previous best, then we use this model.
                # Note that I changed the formula so that we are always maximizing.
                if this_score > best_model_score:
                    best_model_score = this_score
                    best_model = model
            except Exception as e:
                if self.verbose:
                    print(e)
                    print("There was a problem splitting the data.")
            # If the model initialization failed because there are too many components
            # then we stop.
            if model is None:
                break
        return best_model

    def base_model(self, num_states, xs=None, lengths=None):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if xs is None or lengths is None:
            xs = self.X
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
        return super().select()


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    NOTE: I changed the formula to BIC = 2 * logL - p * logN so that we are always maximizing
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        # Call the select method for the super class, passing the bic scoring function.
        return super().select(selector_func=self.bic)

    def bic(self, model):
        """The scoring formula for BIC.

        params:
            model - the model we are testing

        returns:
            float - the score for this model.
        """
        return 2.0 * model.score(self.X, self.lengths) - (model.n_components * (model.n_components - 1) + model.n_components * model.n_features) * np.log(len(self.lengths))


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    # Call the select method for the super class, passing the dic scoring function.

    def select(self):
        return super().select(selector_func=self.dic)

    def dic(self, model):
        """The scoring formula for DIC.

        params:
            model - the model we are testing

        returns:
            float - the score for this model.
        """
        scores = [model.score(y[0], y[1]) for k, y in self.hwords.items() if k != self.this_word]
        this_score = sum(scores) / float(len(scores))
        this_score = model.score(self.X, self.lengths) - this_score
        return this_score


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def train_test_data_split(self, train_indexes, test_indexes):
        """Given the train and test indexes from the split, this function returns the corresponding
        Xs and lengths

        param:
            train_indexes - the indexes for training
            test_indexes - the indexes for testing

        returns:
            list - Xs for training
            list - lengths for training
            list - Xs for testing
            list - lengths for training
        """
        train_x = []
        test_x = []
        train_l = []
        test_l = []
        global_index = 0
        for index, length in enumerate(self.lengths):
            # If this is a train index, then we add the xs and lengths for training.
            if index in train_indexes:
                for j in range(length):
                    train_x.append(self.X[global_index + j])
                train_l.append(length)
            # Otherwise we add the xs and lengths for training.
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
                    train_x, train_lengths, test_x, test_lengths = self.train_test_data_split(
                        cv_train_idx, cv_test_idx)
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
        return best_model
