"""
 init class for all the generators for all the datasets
"""

from siamese.archs.networks import *

SIAMESE = {"siamese1", "siamese2"}


def get_available_siamese():
    """
    lists all the available discriminators
    :return: None
    """
    return sorted(SIAMESE)


def make_siamese_network(name, *args, **kwargs):
    """
    creates the autoencoder based on the name
    :param name: string name of the autoencoder
    :param args: params for the autoenocoder object
    :param kwargs: params for the autoenocoder object
    :return: the autoencoder object
    """
    name = name.strip().lower()
    if not name in SIAMESE:
        raise ValueError("invalid siamese network architecture: '{0}'".format(name))

    elif name == "siamese1":
        return SiameseNetwork1(*args, **kwargs)

    elif name == "siamese2":
        return SiameseNetwork2(*args, **kwargs)