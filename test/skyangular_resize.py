import numpy as np
from envmap import EnvironmentMap, downscaleEnvmap
from matplotlib import pyplot as plt

img_latlong = np.zeros((1024, 2048,3), np.float64) + 1
# img_latlong[int(img_latlong.shape[0]/2):, :]=0
# img_latlong[::2, 1::2]=0

# Create Envmap
e_ll = EnvironmentMap(np.copy(img_latlong), 'latlong')

# Downscale
sao = e_ll.solidAngles()
sat = EnvironmentMap(512, 'Latlong').solidAngles()
e_llD = downscaleEnvmap(e_ll, sao, sat, 1)
#import pdb; pdb.set_trace()
plt.imshow(e_llD.data)
plt.show()

# Convert Envmap to Skyangular
e_saD = e_llD.convertTo('skyangular')
plt.imshow(e_llD.data)
plt.show()