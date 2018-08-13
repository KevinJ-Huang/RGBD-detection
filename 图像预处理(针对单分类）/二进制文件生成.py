from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
data_dir = '2.png'
img = np.array(plt.imread(data_dir))
print(img.shape)