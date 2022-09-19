Define Your FST
================
Our toolkit already provides serialization, training, decoding, and evaluation code. With that being said, the most work that you need to do is preparing dataset as well writing a python script to define your finite state machines under the folder ``nFST/fsm``, where you can also check our already-defined machines for transliteration and slot-filling tasks. In your customized machine, you need to implement several functionalities including.
 * Define input/output symbols (and any optional control symbols) and build a vocab/dictionary on it.
 * Define the strcuture of your machine
To perform first task, you need to retrieve all possible input/output characters/words/etc. from the data you prepared and simply do ``Vocab.add_word(x)``. Note that you might need to define control symbols (for various finite state operations such as concatenation, compositions, or it could simply be information that you hope your model can exploit). In our case config file :ref:`preprocess_config`, we already pre-defined some useful control symbols such as ``input-mark``, ``insertion-mark``, etc. These special marks are seen by models during training. To define the structure of your machine, you need to understand how to write fst with `mfst<https://github.com/matthewfl/openfst-wrapper>`_ toolkit (which is a wrapper for openfst). The general process is as following:

.. code-block:: console
 
  from mfst import FST
  fst = FST(semiring_class=your_semiring)
  # define state and arcs
  state1 = fst.add_state()
  state2 = fst.add_state()
  fst.add_arc(state1,state2, input_lable={some token}, output_label={some token}, weight={depends on your semiring}
  
In our case, we use a self-defined semiring called ``RefWeight`` which allows for most FST operations on tuple of marks (your input/output token). Please refer to ``nFST/fsm/tr.py`` for examples.

Serialize FST
==============
To serialize your customized FST with input data, you need to add a file under ``nFST/preprocess`` that extends our implementation ``preprocess.py``. A helpful example to look at is our ``tr.py`` file. The major functionality of ``preprocess.py`` is to:
  * serialize the fst machine itself
  * serialize composed machine xTy (where x,y are input/output and T is your defined machine)

We already handled the first case (serializing fst machine itself) in the implementation of ``preprocess.py`` and it should take care of most use cases. In your customized preprocessing file, you will need to compose your input (need to convert it into tokens with your Vocab file) with the FST machine you defined. If you correctly implemented the FST as described in the previous section, it is very simple to do so by calling ``(x.compose(T)).compose(y)``. In our ``tr.py`` files, you will be able to copy over most codes for serialization. It is likely that all you need to do is convert input sequence with Vocab file.

.. _preprocess_config:
Configuration
=============
Parameters setup can be found under ``nFST/conf/preprocess/standard.yaml``. Throughout the project, we use *hydra* to manage arguments instead of using command line argparser. We use it because it allows easier management of parameter passing, especially when we need to change any setup.


