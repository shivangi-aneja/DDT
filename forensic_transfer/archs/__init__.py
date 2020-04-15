"""
 init class for all the generators for all the datasets
"""

from forensic_transfer.archs.networks import *

AUTOENCODERS = {"ft1", "ft2"}


def get_available_autoencoders():
    """
    lists all the available discriminators
    :return: None
    """
    return sorted(AUTOENCODERS)


def make_autoencoder(name, *args, **kwargs):
    """
    creates the autoencoder based on the name
    :param name: string name of the autoencoder
    :param args: params for the autoenocoder object
    :param kwargs: params for the autoenocoder object
    :return: the autoencoder object
    """
    name = name.strip().lower()
    if not name in AUTOENCODERS:
        raise ValueError("invalid autoencoder architecture: '{0}'".format(name))

    elif name == "ft1":
        return Autoencoder1(*args, **kwargs)

    elif name == "ft2":
        return Autoencoder2(*args, **kwargs)

