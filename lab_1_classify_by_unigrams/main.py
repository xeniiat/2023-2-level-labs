"""
Lab 1
Language detection
"""


def tokenize(text: str) -> list[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """
    if not isinstance(text, str):
        return None

    for element in text:
        text = ''.join(element for element in text if element.isalpha())
    letters = list(text.lower())
    return letters


def calculate_frequencies(tokens: list[str] | None) -> dict[str, float] | None:
    """
    Calculates frequencies of given tokens
    :param tokens: a list of tokens
    :return: a dictionary with frequencies
    """
    freq_dic = {}
    if not isinstance(tokens, list):
        return None
    for letter in tokens:
        if not isinstance(letter, str):
            return None
    for letter in tokens:
        freq_dic[letter] = (1 if letter not in freq_dic else freq_dic[letter]+1)
    for letter in freq_dic:
        freq_dic[letter] /= len(tokens)
    return freq_dic





def create_language_profile(language: str, text: str) -> dict[str, str | dict[str, float]] | None:
    """
    Creates a language profile
    :param language: a language
    :param text: a text
    :return: a dictionary with two keys – name, freq
    """
    if not isinstance(text, str) or not isinstance(language, str):
        return None
    frequencies = calculate_frequencies(tokenize(text))
    if isinstance(frequencies, dict):
        return {'name': language, 'freq': frequencies}
    return None


def calculate_mse(predicted: list, actual: list) -> float | None:
    """
    Calculates mean squared error between predicted and actual values
    :param predicted: a list of predicted values
    :param actual: a list of actual values
    :return: the score
    """
    if not isinstance(predicted, list) or not isinstance(actual, list):
        return None
    if len(predicted) != len(actual):
        return None
    total_number = 0
    for index, value in enumerate(actual):
        total_number += (value - predicted[index])**2
    mse = total_number / len(predicted)
    return mse





def compare_profiles(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_to_compare: dict[str, str | dict[str, float]]
) -> float | None:
    """
    Compares profiles and calculates the distance using symbols
    :param unknown_profile: a dictionary of an unknown profile
    :param profile_to_compare: a dictionary of a profile to compare the unknown profile to
    :return: the distance between the profiles
    """
    if not isinstance(unknown_profile, dict) or not isinstance(profile_to_compare, dict):
        return None
    if "name" not in unknown_profile or "freq" not in unknown_profile \
        or "name" not in profile_to_compare or "freq" not in profile_to_compare:
        return None
    all_letters = set(unknown_profile["freq"].keys()).union(set(profile_to_compare["freq"].keys()))
    language_1 = []
    language_2 = []
    for letter in all_letters:
        language_1.append(unknown_profile["freq"].get(letter, 0))
        language_2.append(profile_to_compare["freq"].get(letter, 0))
    difference = calculate_mse(language_1, language_2)
    return difference
def detect_language(
        unknown_profile: dict[str, str | dict[str, float]],
        profile_1: dict[str, str | dict[str, float]],
        profile_2: dict[str, str | dict[str, float]],
) -> str | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param profile_1: a dictionary of a known profile
    :param profile_2: a dictionary of a known profile
    :return: a language
    """
    if not isinstance(unknown_profile, dict) or not isinstance(profile_1, dict)\
        or not isinstance(profile_2, dict):
        return None
    difference_1 = compare_profiles(unknown_profile, profile_1)
    difference_2 = compare_profiles(unknown_profile, profile_2)
    name_1 = profile_1["name"]
    name_2 = profile_2["name"]
    names = [name_1, name_2]
    if isinstance(difference_1, float) and isinstance(difference_2, float):
        if difference_1 < difference_2:
            return str(name_1)
        if difference_1 > difference_2:
            return str(name_2)
        if difference_1 == difference_2:
            if isinstance(name_1, str) and isinstance(name_2, str):
                return str(names.sort()[0])
    return None

def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
