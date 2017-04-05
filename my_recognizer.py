import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for item in sorted(test_set.get_all_sequences().keys()):
        X, lengths = test_set.get_item_Xlengths(item)
        probs = dict()
        best_word = ""
        best_log_prob = float("-inf")
        for word, model in models.items():
            try:
                log_prob = model.score(X, lengths)
                probs[word] = log_prob
                # Keep track of the best word so far so we can add it to guesses
                if best_log_prob < log_prob:
                    best_log_prob = log_prob
                    best_word = word
            except Exception as e:
                probs[word] = float("-inf")
        # Add this data to the returned variables
        probabilities.append(probs.copy())
        guesses.append(best_word)
    return probabilities, guesses
