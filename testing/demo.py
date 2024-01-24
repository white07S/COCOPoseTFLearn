import sys
sys.path.append('/home/preetam/projects/project2/COCOPoseTFLearn')
from model import cmu_model
from gui.config_reader import config_reader
import cv2
import matplotlib
import pylab as plt
import numpy as np
from gui import util

model = cmu_model.get_testing_model()
model.load_weights('weights.best.h5')
