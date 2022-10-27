import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap
import copy
import os.path

# RdBu colormap
RdBu_cmap = copy.copy(cm.get_cmap("RdBu_r"))
RdBu_cmap.set_under('w')
RdBu_cmap.set_bad('w')

DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template")

# Planck colormap
Planck_template = np.loadtxt(os.path.join(DATAPATH, "Planck_RGB.txt"))
Planck_cmap = ListedColormap(Planck_template/255.)
Planck_cmap.set_bad("gray") # color of missing pixels
Planck_cmap.set_under("white") # color of background, necessary if you want to use
