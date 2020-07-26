Assignment 3
James Hooper ~ NETID: jah171230
Hritik Panchasara ~ NETID: hhp160130
------------------------------------------------------------------------------------------------------------------------------------
- For this assignment we used PyCharm to create/edit/run the code. 
- In PyCharm the code should run by simply pressing the Run dropdown, then clicking run making sure you are running NeuralNetFinal
- The dataset is public on a Github account that one of us own. 
- In if __name__ == '__main__' , you can select the parameters you wish to choose. 
- Some issues occur with too big of sample size or k-value due to memory allocation. Not sure how to fix this it may be a problem dependent on system or IDE.
Part 1:
- You can set the random state for constant results.
- The two functions called are both implmenting kmeans, but one will create an elbow method visual and the other will create a silhouette score visual. This is for understanding which k-value is best.
- The process data function simply prepares the data for the kmeans algorithm by removing the class column and scaling the data.
Part 2:
- You can set the k-value, the random state for constant results, and the number of images used (for this assignment we obviously used 3).
- The process data function retrieves the images to be quantized and sends it back to main.
- The color quantization function will return the quantized images back to main.
- In main itself will print out how long it takes for each image to be quanitzed by the function.
- Once all images are gathered & quantized the print_images function will be called to finally display the images with matplot and save the images to the python folder for file size analysis.
------------------------------------------------------------------------------------------------------------------------------------
Link to Datasets/Images:
~ Part 1 Dataset: https://archive.ics.uci.edu/ml/datasets/HTRU2
~ Part 2 Images: http://www.utdallas.edu/~axn112530/cs6375/unsupervised/images
- Reminder that these are all being housed in one of our githubs for ease of use within PyCharm.
------------------------------------------------------------------------------------------------------------------------------------
Libraries Used:
Part 1:
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part 2:
import numpy as np
from sklearn import cluster
from skimage import io
import matplotlib.pyplot as plt
from time import time
import sys
------------------------------------------------------------------------------------------------------------------------------------
Just in case. To import libraries/packages in PyCharm.
- Go to File.
- Press Settings.
- Press Project drop down.
- Press Project Interpreter.
- Press the plus sign on the top right box, should be to the right of where it says "Latest Version".
- Search and Install packages as needed.
- For this assignment the packages are: pandas, numpy, matplotlib, scikit-image, and scikit-learn/sklearn.