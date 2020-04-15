"""
 init class for all the generators for all the datasets
"""

from dsne_ae.archs.networks import *

AUTOENCODERS = {"dsne1"}


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

    elif name == "dsne1":
        return Autoencoder1(*args, **kwargs)
