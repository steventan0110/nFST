Welcome to nFST documentation!
===================================

nFST (neuralized finite state transducer) is a sequence modeling toolkit
that allows users to define their own latent variable models (with help of finite state machines) to
train sequence models. Using FST to represent latent structure is a generic approach that
can be applied to various tasks including transliteration, slot filtering (tagging), cipher decoding, etc.
In this documentation, we provide tutorials as well as configuration for sequence modeling using FST.

Installation & Setup
--------------------
After cloning the repository, install packages by

.. code-block: console
   
   (.venv) $ pip install -r /path/to/nFST/requirements.txt

.. _packages:

Packages
~~~~~~~~~
Note that mfst packgaes cannot be properly installed by pip, please refer to the 
`note <https://github.com/matthewfl/openfst-wrapper/blob/master/notes.txt>`_ to clone
the project and install it.

This project models the joint probablity of the sequences. Since it does not support
conditional model yet, we relies on other sequence models to generate hypothesis
and perform reranking to compare the performance. This means third-party generated
hypothesis is required. In our research, we used LSTM models trained with Fairseq
toolkit as the hypothesis-provider. To learn more about Fairseq, please refer to
its `tutorials <https://fairseq.readthedocs.io/en/latest/>`_

.. note::

   This project is under active development.

Contents
--------
.. toctree::
    :maxdepth: 5
    :caption: Preprocessing
    
    preprocess

.. toctree::
    :maxdepth: 5
    :caption: Training
    
    train

.. toctree::
    :maxdepth: 5
    :caption: Decoding
    
    decode

.. toctree::
    :maxdepth: 5
    :caption: Evaluation
    
    eval
