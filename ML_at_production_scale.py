
# ----------------------------------------------------------------------------------------------------------------------
# Build a linear model with Estimators  |  TensorFlow
# ----------------------------------------------------------------------------------------------------------------------
# https://www.tensorflow.org/tutorials/estimators/linear



import tensorflow as tf
import tensorflow.feature_column as fc
import os
import sys
import matplotlib.pyplot as plt
from IPython.display import clear_output

tfe = tf.enable_eager_execution()


# ! pip install -q requests
# ! git clone --depth 1 https://github.com/tensorflow/models


models_path = os.path.join(os.getcwd(), 'models')
sys.path.append(models_path)

from official.wide_deep import census_dataset
from official.wide_deep import census_main
census_dataset.download("/tmp/census_data/")

#export PYTHONPATH=${PYTHONPATH}:"$(pwd)/models"
#running from python you need to set the `os.environ` or the subprocess will not see the directory.
if "PYTHONPATH" in os.environ:
    os.environ['PYTHONPATH'] += os.pathsep +  models_path
else:
    os.environ['PYTHONPATH'] = models_path


!python -m official.wide_deep.census_main --help

!python -m official.wide_deep.census_main --model_type=wide --train_epochs=2



# 增加一行



















































































































































































































