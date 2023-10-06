"""
Lab 1
Language detection
"""
import json


def tokenize(text: str) -> list[str] | None:
    """
    Splits a text into tokens, converts the tokens into lowercase,
    removes punctuation, digits and other symbols
    :param text: a text
    :return: a list of lower-cased tokens without punctuation
    """
    if not isinstance(text, str):
        return None
    letters = [element.lower() for element in text if element.isalpha()]
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
    if not isinstance(frequencies, dict):
        return None
    return {'name': language, 'freq': frequencies}


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
    name_1 = str(profile_1["name"])
    name_2 = str(profile_2["name"])
    if not isinstance(difference_1, float) or not isinstance(difference_2, float):
        return None
    if difference_1 < difference_2:
        return str(name_1)
    if difference_1 > difference_2:
        return str(name_2)
    if difference_1 == difference_2:
        names = [name_1, name_2]
        names.sort()
        return names[0]



def load_profile(path_to_file: str) -> dict | None:
    """
    Loads a language profile
    :param path_to_file: a path to the language profile
    :return: a dictionary with at least two keys – name, freq
    """
    if not isinstance(path_to_file, str):
        return None
    with open(path_to_file, "r", encoding="utf-8") as file_to_read:
        language_profile = json.load(file_to_read)
    if not isinstance(language_profile, dict):
        return None
    return language_profile


def preprocess_profile(profile: dict) -> dict[str, str | dict] | None:
    """
    Preprocesses profile for a loaded language
    :param profile: a loaded profile
    :return: a dict with a lower-cased loaded profile
    with relative frequencies without unnecessary ngrams
    """
    if not isinstance(profile, dict) or "name" not in profile.keys()\
        or "freq" not in profile.keys() or "n_words" not in profile.keys():
        return None
    freq_dict = {}
    total_number = profile["n_words"][0]
    for gramm in profile["freq"]:
        if len(gramm) == 1:
            if gramm.lower() not in freq_dict:
                freq_dict[gramm.lower()] = profile["freq"][gramm] / total_number
            else:
                freq_dict[gramm.lower()] += profile["freq"][gramm] / total_number
    preprocessed_profile = {"name": profile["name"], "freq": freq_dict}
    return preprocessed_profile


def collect_profiles(paths_to_profiles: list) -> list[dict[str, str | dict[str, float]]] | None:
    """
    Collects profiles for a given path
    :paths_to_profiles: a list of strings to the profiles
    :return: a list of loaded profiles
    """
    if not isinstance(paths_to_profiles, list):
        return None
    processed_profiles = []
    for path in paths_to_profiles:
        language_profile = load_profile(path)
        if isinstance(language_profile, dict):
            processed_profile = preprocess_profile(language_profile)
            if isinstance(processed_profile, dict):
                processed_profiles += [processed_profile]
    return processed_profiles


def detect_language_advanced(unknown_profile: dict[str, str | dict[str, float]],
                             known_profiles: list) -> list | None:
    """
    Detects the language of an unknown profile
    :param unknown_profile: a dictionary of a profile to determine the language of
    :param known_profiles: a list of known profiles
    :return: a sorted list of tuples containing a language and a distance
    """
    if not isinstance(unknown_profile, dict) or not isinstance(known_profiles, list):
        return None
    detected_language = []
    for profile in known_profiles:
        mse_value = compare_profiles(profile, unknown_profile)
        detected_language.append((profile["name"], mse_value))
    detected_language.sort(key=lambda x: (x[1], x[0]))
    return detected_language


def print_report(detections: list[tuple[str, float]]) -> None:
    """
    Prints report for detection of language
    :param detections: a list with distances for each available language
    """
    if not isinstance(detections, list):
        for detection in detections:
            name = detection[0]
            score = round(detection[1], 5)
            print(f"{name}: MSE {score}")
