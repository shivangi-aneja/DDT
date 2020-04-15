"""
 init class for all the generators for all the datasets
"""

from latent_classifier.archs.networks import *

CLASSIFIERS = {"classifier1", "classifier2", "classifier3", "classifier4"}


def get_available_classifiers():
    """
    lists all the available discriminators
    :return: None
    """
    return sorted(CLASSIFIERS)


def make_classifier(name, *args, **kwargs):
    """
    creates the autoencoder based on the name
    :param name: string name of the autoencoder
    :param args: params for the autoenocoder object
    :param kwargs: params for the autoenocoder object
    :return: the autoencoder object
    """
    name = name.strip().lower()
    if not name in CLASSIFIERS:
        raise ValueError("invalid autoencoder architecture: '{0}'".format(name))

    elif name == "classifier1":
        return Classifier1(*args, **kwargs)
    elif name == "classifier2":
        return Classifier2(*args, **kwargs)
    elif name == "classifier3":
        return Classifier3(*args, **kwargs)
