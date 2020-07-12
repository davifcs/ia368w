#

#   Copyright (C) 2014 Eleri Cardozo

#

#  This program is free software: you can redistribute it and/or modify

#  it under the terms of the GNU General Public License as published by

#  the Free Software Foundation, either version 3 of the License, or

#  (at your option) any later version.

#

#  This program is distributed in the hope that it will be useful,

#  but WITHOUT ANY WARRANTY without even the implied warranty of

#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the

#  GNU General Public License for more details.

#

#  You should have received a copy of the GNU General Public License

#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

#



# function result = FeatureDetection(global_poses, L, Pose)

# 

# argumentos:

# global_poses: /perception/laser/n/global_poses

# L (landmarks): L.x e L.y [mm]

# Pose: pose com th em rad

# 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from SplitAndMerge_distances import SplitAndMerge

# --------------------------------------

# Passo 1 - Split and Merge

def FeatureDetection(distances, laser_range):
  PointsX, PointsY, NPoints = SplitAndMerge(distances, laser_range)

  if (len(NPoints) < 3):   # unica reta ?
    print("Nenhuma feature detectada\n")
    return

  # Landmarks detectados no momento atual pelo sensor (com erro)
  Features = []
  range_bearing = []
  for x, y, n in zip(PointsX,PointsY,NPoints):
    if n < 3:    # despreza retas comapenas 2 pontos
      continue
    r = np.sqrt(x**2 + y**2)
    b = np.arctan2(y,x)
    if b > np.pi:
      b = b - 2 * np.pi
    elif  b < -np.pi:
      b = 2*np.pi + b
    range_bearing.append(r)
    range_bearing.append(b)
    if len(Features) == 0:
      Features = range_bearing
    else:
      Features = np.vstack((Features,range_bearing))
    range_bearing = []

  return Features[1:]
  
# import restthru

# host = 'http://127.0.0.1:4950'
# restthru.http_init()
# distances_request = "/perception/laser/1/distances?range=-90:90:1"
# distances,_ = restthru.http_get(host+distances_request)

# laser_range = np.array([-90,90,1])
# pose_request = "/motion/pose"
# pose, _ = restthru.http_get(host+pose_request)

# Features = FeatureDetection(distances,laser_range)
# print(Features)
# fig, ax = plt.subplots()
# print(Features[:,0],Features[:,1])
# ax.scatter(Features[:,0],Features[:,1])
# plt.show()
