
#


```
$ python --version
Python 3.5.3 :: Anaconda 2.5.0 (x86_64)

# tensorflow v1.1 install
$ pip uninstall tensorflow
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0rc1-py3-none-any.whl
$ pip install --ignore-installed --upgrade $TF_BINARY_URL

$ python
>>> import tensorflow as tf
>>> tf.__version__
'1.1.0-rc1'

$ jupyter notebook
```
