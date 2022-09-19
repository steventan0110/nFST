Installation & Setup
=====

Packages
------------
After cloning the repository from GitHub, install the required
environment files using requirement

.. code-block:: console

   (.venv) $ pip install -r /path/to/nFST/requirements.txt

Note that mfst packgaes cannot be properly installed by pip, please refer to the 
`note <https://github.com/matthewfl/openfst-wrapper/blob/master/notes.txt>` to clone
the project and install it.

This project models the joint probablity of the sequences. Since it does not support
conditional model yet, we relies on other sequence models to generate hypothesis
and perform reranking to compare the performance. This means third-party generated
hypothesis is required. In our research, we used LSTM models trained with Fairseq
toolkit as the hypothesis-provider. To learn more about Fairseq, please refer to
its `tutorials <https://fairseq.readthedocs.io/en/latest/>`_.