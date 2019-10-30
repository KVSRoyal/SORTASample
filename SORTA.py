# A custom SORTA implementation
import pathlib
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk


class SORTA:
    def __init__(self):
        # Make a stemmer for all the stuff we have to do later
        self.__stemmer = SnowballStemmer('english')

        # Download stopwords for the tokenizing
        nltk.download('stopwords')
        nltk.download('punkt')  # Needed for the tokenizer
        self.__stop_words = set(stopwords.words('english'))

        # Add some custom stop words
        self.__stop_words.add('congenital')
        self.__stop_words.add('elsewhere')
        self.__stop_words.add('classified')
        self.__stop_words.add('unspecified')
        self.__stop_words.add('specified')
        self.__stop_words.add('associated')
        self.__stop_words.add('site')
        self.__stop_words.add('type')

        # Load the list of words we want to weight
        self.__vip_words = self.__load_vip_words()

        # Load all terms and their ids from hobo
        self.__terms_to_ids_and_bigrams = {}
        hobo_path = pathlib.Path(__file__).parent / 'rsrc' / 'hobo.obo'
        with open(hobo_path, 'r', encoding='utf-8') as file:
            file = file.read()

            # Get all [Term]s
            paragraphs = [p.strip() for p in re.split(r'\[Term\]', file)[1:]]

            # Get the id and name from the first two lines of each term
            for paragraph in paragraphs:
                lines = paragraph.split('\n')
                term_id = lines[0].split(' ', maxsplit=1)[1].split(':')[1]
                name = lines[1].split(' ', maxsplit=1)[1]

                # Normalize and split the word into tokens for the bigram matching
                tokens = self.__to_bigram_tokens(name, True)

                # Store the id and tokens by name
                self.__terms_to_ids_and_bigrams[name] = [term_id, tokens]

        # Load all of the original terms and their ids from hpo
        self.__id_to_name = {}
        hpo_path = pathlib.Path(__file__).parent / 'rsrc' / 'hpo.obo'
        with open(hpo_path, 'r', encoding='utf-8') as file:
            file = file.read()

            # Get all [Term]s
            paragraphs = [p.strip() for p in re.split(r'\[Term\]', file)[1:]]

            # Get the id and name from the first two lines of each term
            for paragraph in paragraphs:
                lines = paragraph.split('\n')
                term_id = lines[0].split(' ', maxsplit=1)[1].split(':')[1]
                name = lines[1].split(' ', maxsplit=1)[1]

                # Store the id and tokens by name
                self.__id_to_name[term_id] = name

        print('Ready for Matching!\n\n\n')


    def get_matches(self, string):
        """
        Returns a list of SORTA matches for this string with score >= 20%, their score, and their HPO code
        :param string: The string we want to match to other strings
        :return: A list of SORTA matches for this string with score >= 20%, their score, and their HPO code
        """
        lhs = self.__to_bigram_tokens(string, False)

        # For debugging purposes
        # print('LHS ===================================================')
        # print(lhs)

        matches = []
        for key in self.__terms_to_ids_and_bigrams.keys():
            rhs = self.__terms_to_ids_and_bigrams[key][1]
            score = self.__score_match(lhs, rhs)

            # If the score is greater than 20, add this term to a list of matches
            if score > 30:  # 80 is good for debugging purposes
                # For debugging purposes
                # print('RHS ===================================================')
                # print(rhs)

                hpo_code = self.__terms_to_ids_and_bigrams[key][0][:7]  # This might be one of our fancy "new" codes
                descriptor = self.__id_to_name[hpo_code]
                matches.append([descriptor, hpo_code, score])

        # print('\n\n')  # For debugging purposes

        # Finally, return all matches in order from largest to smallest score
        matches.sort(key=lambda match: -match[-1])
        return matches

    def __to_bigram_tokens(self, string, is_og):
        """
        Returns a list of bigram tokens for this string
        :param string: The string we want to split into bigram tokens
        :param is_og: True if this is an ontology term. Otherwise, this should be False
        :return: a list of bigram tokens for this string
        """
        string = string.lower()  # VVVVVI

        # Remove these from any input terms, but not ontology terms
        if not is_og:
            string = re.split(r' with ', string)[0]  # Get rid of any "with"s
            string = re.split(r' w/ ', string)[0]
            string = re.split(r' without ', string)[0]  # Get rid of any "without"s
            string = re.split(r' w/o ', string)[0]
            string = re.split(r' due to ', string)[0]  # Get rid of any "due to"s
            string = re.split(r' following ', string)[0]  # Get rid of any "following"s

        # Normalize and split the word into tokens by whitespace. Remove stop words and non-alphanumeric characters
        # nltk pretends hyphens and '/' don't exist. bad nltk. bad
        tokens = word_tokenize(string.replace('-', ' ').replace('/', ' '))
        tokens = [re.sub(r'[\W|\(|\)]+', '', token).strip() for token in tokens]
        tokens = [token for token in tokens if token != '' and token not in self.__stop_words]
        tokens = self.__weight_words(tokens)  # Weight tokens as necessary

        # Stem the tokens
        tokens = ['^' + (self.__stemmer.stem(token).upper() if token.isupper() else self.__stemmer.stem(token).lower())
                  + '$' for token in tokens]

        bigram_tokens = []  # Duplicates are allowed according to the SORTA paper
        for token in tokens:
            bigrams = [token[index:index + 2] for index in range(len(token) - 1)]
            bigram_tokens.extend(bigrams)

        return bigram_tokens

    @staticmethod
    def __score_match(lhs, rhs):
        """
        Returns a similarity score for these two lists of bigrams, according to SORTA's algorithm.
        :param lhs: The lhs list of bigram tokens
        :param rhs: The rhs list of bigram tokens
        :return: A similarity score for these two lists of bigrams, according to SORTA's algorithm.
        """
        # Don't do anything to the original lists
        lhs = lhs.copy()
        rhs = rhs.copy()
        lhs_size = len(lhs)
        rhs_size = len(rhs)

        # Count the number of shared tokens. Because duplicates are allowed, if we find a match,
        # remove it from the other list so it won't pair with anything else
        num_shared_tokens = 0
        for i in range(len(lhs)):
            for j in range(len(rhs)):
                if lhs[i] == rhs[j]:
                    rhs.pop(j)
                    j -= 1
                    num_shared_tokens += 1
                    break

        # Finally, put everything into the formula and return our score
        similarity = (num_shared_tokens * 2) / (lhs_size + rhs_size)
        return similarity * 100  # Return a percentage

    def __weight_words(self, tokens):
        """
        Returns a weighted version of this list of tokens. (All important words will occur twice.)
        :param tokens: The list of tokens we want to way as necessary
        :return: A weighted version of this list of tokens. (All important words will occur twice.)
        """
        weighted_tokens = []

        # Weigh words by making them upper case
        for token in tokens:
            weighted_tokens.append(token)  # Add each token at least once
            # Check if this token contains a vip word. If it does, upper case that word
            for word in self.__vip_words:
                if re.search(word, token) is not None:
                    token = re.sub(word, word.upper(), token)
                    weighted_tokens.append(token.upper())
                    break

        return weighted_tokens

    @staticmethod
    def __load_vip_words():
        """
        Returns a set of all terms that should be given more importance in an analysis
        :return: A set of all terms that should be given more importance in an analysis
        """
        # The total set to return
        vip_words = set()

        # Load the list of organ tokens, normalize them, and add them our list of weight words
        organs_path = pathlib.Path(__file__).parent / 'rsrc' / 'organs.txt'
        with open(organs_path, 'r', encoding='utf-8') as file:
            file = file.read()
            for line in file.split('\n'):
                tokens = line.split(' ')
                for token in tokens:
                    vip_words.add(token.strip().lower())

        # Add anything else we think is important
        vip_words.add('mental')

        return vip_words

