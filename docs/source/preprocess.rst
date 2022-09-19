Define Your FST
================


Serialize FST
==============

To serialize your customized FST with input data, you need to add a file under ``nFST/preprocess`` that extends our implementation ``preprocess.py``. A helpful example to look at is our ``tr.py`` file. The major functionality of ``preprocess.py`` is to:
  * serialize the fst machine itself
  * serialize composed machine xTy (where x,y are input/output and T is your defined machine)

We already handled the first case (serializing fst machine itself) in the implementation of ``preprocess.py`` and it should take care of most use cases. In your customized preprocessing file, you will need to compose your input (need to convert it into tokens with your Vocab file) with the FST machine you defined. If you correctly implemented the FST as described in the previous section, it is very simple to do so by calling ``(x.compose(T)).compose(y)``. In our ``tr.py`` files, you will be able to copy over most codes for serialization. It is likely that all you need to do is convert input sequence with Vocab file.
 
Configuration
=============
Parameters setup can be found under ``nFST/conf/preprocess/standard.yaml``. Throughout the project, we use *hydra* to manage arguments instead of using command line argparser. We use it because it allows easier management of parameter passing, especially when we need to change any setup.


