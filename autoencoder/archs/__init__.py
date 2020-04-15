"""
 init class for all the generators for all the datasets
"""

from autoencoder.archs.networks import *

AUTOENCODERS = {"test_autoencoder1", "test_autoencoder2", "test_autoencoder3", "cifar"}


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

    elif name == "test_autoencoder1":
        return TestAutoencoder1(*args, **kwargs)
    elif name == "test_autoencoder2":
        return TestAutoencoder2(*args, **kwargs)
    elif name == "test_autoencoder3":
        return TestAutoencoder3(*args, **kwargs)
    elif name == "cifar":
        return AutoencoderCifar(*args, **kwargs)

