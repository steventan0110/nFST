Welcome to nFST documentation!
===================================

nFST (neuralized finite state transducer) is a sequence modeling toolkit
that allows users to define their own latent variable models (with help of finite state machines) to
train sequence models. Using FST to represent latent structure is a generic approach that
can be applied to various tasks including transliteration, slot filtering (tagging), cipher decoding, etc.
In this documentation, we provide tutorials as well as configuration for sequence modeling using FST.

.. note::

   This project is under active development.

Contents
--------
.. toctree::
    :maxdepth: 1
    :caption: Installation
    
    install
    
.. toctree::
    :maxdepth: 1
    :caption: Preprocessing
    
    preprocess

.. toctree::
    :maxdepth: 1
    :caption: Training
    
    train

.. toctree::
    :maxdepth: 1
    :caption: Decoding
    
    decode

.. toctree::
    :maxdepth: 1
    :caption: Evaluation
    
    eval
