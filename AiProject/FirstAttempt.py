#import keras
#import numpy as np
#from parser import load_data
#---------------------------------------

import os

#training_data=load_data('data\smalltrain')
#validation



path =  os.getcwd()
filenames = os.listdir(path)

x=1

for filename in filenames:
    os.rename(filename, filename.replace(" ","-"))
    x=x+1