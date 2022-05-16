# -*- coding: utf-8 -*-
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from collections import Counter
from typing import List

#import spacy
from catboost import CatBoostClassifier
from nltk.stem.snowball import SnowballStemmer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# https://github.com/igorbrigadir/stopwords/blob/master/en/alir3z4.txt
with open("stopwords.txt") as stop_words_file:
    STOP_WORDS_ALIR3Z4 = stop_words_file.read().split("\n")

# https://github.com/first20hours/google-10000-english/blob/master/google-10000-english-no-swears.txt
with open("popular-words.txt") as popular_words_file:
    POPULAR_WORDS = popular_words_file.read().split("\n")

POPULAR_TAGS = list(set(POPULAR_WORDS) - set(STOP_WORDS_ALIR3Z4))


class BaseWordsProcessor(ABC):
    @abstractmethod
    def procc(self, word: str) -> str:
        ...


class EmptyWordsProcessor(BaseWordsProcessor):
    def __init__(self):
        pass

    def procc(self, word: str) -> str:
        return word


class NonemptyWordsProcessor(BaseWordsProcessor, ABC):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor

    @abstractmethod
    def procc(self, word: str) -> str:
        ...


class SpacyWordsProcessor(NonemptyWordsProcessor):
    def __init__(self, preprocessor):
        super().__init__(preprocessor)

    def procc(self, word: str) -> str:
        word = self.preprocessor(word)
        word = word[0].lemma_
        return word


class NltkWordsProcessor(NonemptyWordsProcessor):
    def __init__(self, preprocessor):
        super().__init__(preprocessor)

    def procc(self, word: str) -> str:
        return self.preprocessor.stem(word)


def extract_words(
    text: str,
    alphabet: str,
    min_length: int = 3,
    stop_words: List[str] = None,
    preprocessor: BaseWordsProcessor = NltkWordsProcessor(
        SnowballStemmer(language="english")
    ),
):
    """Split text into word."""
    stop_words = stop_words or []

    # filter symbols
    text = "".join((c if c in alphabet else " ") for c in text.lower())

    # split to words
    words = text.split()

    # filter words
    return [
        preprocessor.procc(word)
        for word in words
        if word not in stop_words and len(word) >= min_length
    ]


class BaseTagger(ABC):
    @abstractmethod
    def get_tags(self, texts: List[str]) -> List[List[str]]:
        """['Text1', 'Text2', ...] -> [['text1_tag1', 'text1_tag2', ...], ...]"""
        ...


class BaseChoiceTagger(BaseTagger, ABC):
    def __init__(self, tags: List[str]):
        self.tags = tags


class BaseSeparateTagger(BaseTagger, ABC):
    @abstractmethod
    def get_tags_from_text(self, text: str) -> List[str]:
        """'Text' -> ['text_tag1', 'text_tag2', ...]"""
        ...

    def get_tags(self, texts: List[str]) -> List[List[str]]:
        result = []
        for text in texts:
            tags = self.get_tags_from_text(text)
            result.append(tags)
        return result


class MostFrequentWordsTagger(BaseSeparateTagger):
    default_stop_words = STOP_WORDS_ALIR3Z4
    words_alphabet = "abcdefghijklmnopqrstuvwxyz-'"

    def __init__(self, stop_words: list = None, max_tags_per_text: int = 5):
        self.stop_words = stop_words or self.default_stop_words
        self.max_tags_per_text = max_tags_per_text

    def get_tags_from_text(self, text: str) -> List[str]:
        # now we do stemming of our words. This should be more correct approach
        words = extract_words(
            text, alphabet=self.words_alphabet, min_length=3, stop_words=self.stop_words
        )
        words_counter = Counter(words)

        tags = []
        result = words_counter.most_common()
        if len(result) == 0:
            return []

        word, max_count = result[0]
        i = 0
        while result[i][1] == max_count:
            tags.append(result[i][0])
            i += 1

        return tags[: self.max_tags_per_text]


class FindSpecialWordsTagger(BaseSeparateTagger, BaseChoiceTagger):
    default_tags_candidates = POPULAR_TAGS
    words_alphabet = "abcdefghijklmnopqrstuvwxyz-'"

    def __init__(self, tags: List[str] = None, max_tags_per_text: int = 5):
        super().__init__(tags=tags or self.default_tags_candidates)
        self.max_tags_per_text = max_tags_per_text

    def get_tags_from_text(self, text: str) -> List[str]:
        words = extract_words(text, alphabet=self.words_alphabet, min_length=3)

        found_tags = []
        for tag in self.tags:
            found_tags.append((tag, words.count(tag)))

        found_tags.sort(key=lambda o: o[1], reverse=True)
        found_tags = found_tags[: self.max_tags_per_text]

        return [tag for tag, count in found_tags]


class CatBoostClassifierTagger(BaseChoiceTagger):
    default_stop_words = STOP_WORDS_ALIR3Z4
    words_alphabet = "abcdefghijklmnopqrstuvwxyz-'"
    default_tags_candidates = [
        "alt.atheism",
        "sci.space",
        "soc.religion.christian",
        "sci.med",
        "talk.politics.guns",
        "sci.electronics",
    ]

    def __init__(
        self,
        tags: List[str] = None,
        stop_words=None,
        clf_pretrained=None,
        confidence_measures=None,
        use_gpu=False,
    ):
        super().__init__(tags=tags or self.default_tags_candidates)
        self.stop_words = stop_words or self.default_stop_words
        if confidence_measures:
            self.confidence_measures = confidence_measures
        else:
            self.confidence_measures = 1 / len(self.tags)

        self.twenty_train = fetch_20newsgroups(
            subset="train", categories=self.tags, shuffle=True
        )
        self.train_data = [
            " ".join(
                extract_words(
                    z, self.words_alphabet, min_length=1, stop_words=self.stop_words
                )
            )
            for z in self.twenty_train.data
        ]

        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(self.train_data)

        self.tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

        if clf_pretrained:
            self.clf = clf_pretrained
        else:
            if use_gpu:
                self.clf = CatBoostClassifier(
                    task_type="GPU", devices="0:1", num_trees=100, depth=11
                )
            else:
                self.clf = CatBoostClassifier(num_trees=100, depth=11)
            self.clf.fit(self.X_train_tfidf, self.twenty_train.target, silent=False)

    def save_model(self, model_name="tagger_model.pkl"):
        with open(model_name, "wb") as file:
            pickle.dump(self.clf, file)

    def load_model(self, model_name="tagger_model.pkl"):
        with open(model_name, "rb") as file:
            self.clf = pickle.load(file)

    def ret_model(self):
        return self.clf

    def get_model(self, clf):
        self.clf = clf

    def get_tags(self, texts: List[str]) -> List[List[str]]:
        texts = [
            " ".join(
                extract_words(
                    z, self.words_alphabet, min_length=1, stop_words=self.stop_words
                )
            )
            for z in texts
        ]
        X_new_counts = self.count_vect.transform(texts)
        X_new_tfidf = self.tfidf_transformer.transform(X_new_counts)
        predicted = []
        for text_probs in self.clf.predict_proba(X_new_tfidf):
            tmp = []
            for num, j in enumerate(text_probs):
                if j >= self.confidence_measures:
                    tmp.append(num)
            predicted.append(tmp)
        tags = [
            [self.twenty_train.target_names[category] for category in text_ind_cat]
            for text_ind_cat in predicted
        ]
        return tags


if __name__ == "__main__":
    classifier = CatBoostClassifierTagger(use_gpu=False)
    model = classifier.ret_model()
    new_classifier = CatBoostClassifierTagger(clf_pretrained=model)

    test_1 = """
    Gun legislation in the United States is augmented by judicial interpretations of the Constitution.
    The Second Amendment of the United States Constitution reads: "A well regulated Militia, being necessary to the security of a free State, the right of the people to keep and bear Arms, shall not be infringed.
    In 1791, the United States adopted the Second Amendment, and in 1868 adopted the Fourteenth Amendment.
    The effect of those two amendments on gun politics was the subject of landmark U.S. Supreme Court decisions in District of Columbia v. Heller (2008), where the Court affirmed for the first time that the second amendment guarantees an individual right to possess firearms independent of service in a state militia and to use them for traditionally lawful purposes such as self-defense within the home, and in McDonald v. City of Chicago (2010), where the Court ruled that the Second Amendment is incorporated by the Due Process Clause of the Fourteenth Amendment and thereby applies to both state and federal law. In so doing, it endorsed the so-called 'individual-right' theory of the Second Amendment's meaning and rejected a rival interpretation, the 'collective-right' theory, according to which the amendment protects a collective right of states to maintain militias or an individual right to keep and bear arms in connection with service in a militia.
    """

    test_2 = """
    The Trinity is an essential doctrine of mainstream Christianity.
    
    From earlier than the times of the Nicene Creed (325) Christianity advocated[78] the triune mystery-nature of God as a normative profession of faith.
    
    According to Roger E. Olson and Christopher Hall, through prayer, meditation, study and practice, the Christian community concluded "that God must exist as both a unity and trinity", codifying this in ecumenical council at the end of the 4th century.[79][80]
    """

    test_3 = """
    The local interstellar medium is a region of space within 100 parsecs (pc) of the Sun, which is of interest both for its proximity and for its interaction with the Solar System. This volume nearly coincides with a region of space known as the Local Bubble, which is characterized by a lack of dense, cold clouds.
    It forms a cavity in the Orion Arm of the Milky Way galaxy, with dense molecular clouds lying along the borders, such as those in the constellations of Ophiuchus and Taurus.
    (The actual distance to the border of this cavity varies from 60 to 250 pc or more.)
    This volume contains about 104–105 stars and the local interstellar gas counterbalances the astrospheres that surround these stars, with the volume of each sphere varying depending on the local density of the interstellar medium.
    The Local Bubble contains dozens of warm interstellar clouds with temperatures of up to 7,000 K and radii of 0.5–5 pc.[88] 
    """

    test_4 = """
    Electronic components have a number of electrical terminals or leads.
    These leads connect to other electrical components, often over wire, to create an electronic circuit with a particular function (for example an amplifier, radio receiver, or oscillator).
    Basic electronic components may be packaged discretely, as arrays or networks of like components, or integrated inside of packages such as semiconductor integrated circuits, hybrid integrated circuits, or thick film devices.
    The following list of electronic components focuses on the discrete version of these components, treating such packages as components in their own right. 
    """

    print(MostFrequentWordsTagger().get_tags([test_4]))
    print(FindSpecialWordsTagger().get_tags([test_4]))
    print(new_classifier.get_tags([test_1, test_2, test_3, test_4]))
