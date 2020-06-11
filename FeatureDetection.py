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
from SplitAndMerge import SplitAndMerge

# --------------------------------------

epsRange = 100

epsBearing = 4*np.pi/180

# ---------------------------------------

# Passo 1 - Split and Merge

def FeatureDetection(global_poses, L, Pose):
  Points = SplitAndMerge(global_poses)

  if (len(Points) < 3):   # unica reta ?
    print("Nenhuma feature detectada\n")
    return

  # Passo 2 - Calculo da MAXIMA VEROSSIMILHANCA

  # As features dos landmarks verdadeiros relativas a posicao atual:

  Z_range   = np.sqrt((L[0] - Pose['x'])**2 + (L[1] -  Pose['y'])**2) # [mm]
  Z_bearing = np.arctan2(L[1] - Pose['y'] , L[0] - Pose['x']) - Pose['th'] # [rad]

  # Landmarks detectados no momento atual pelo sensor (com erro)

  points_len = len(Points[0])

  l_x = []
  l_y = []

  for i in range(0,points_len,2):
    if Points[2][i] > 2:    # despreza retas comapenas 2 pontos
      l_x.append(Points[0][i])
      l_y.append(Points[1][i])

  l_x = np.asarray(l_x)
  l_y = np.asarray(l_y)

  # Features atuais

  z_range = np.sqrt((l_x - Pose['x'])**2 + (l_y - Pose['y'])**2) # [mm]

  z_bearing = np.arctan2(l_y - Pose['y'], l_x - Pose['x']) - Pose['th'] # [rad]

  Features = []

  for i in range(0,len(z_range)):
    for j in range(0,len(Z_range)):
        deltaRange = abs(z_range[i] - Z_range[j])
        deltaBearing = abs(z_bearing[i] - Z_bearing[j])
        if (deltaRange < epsRange and deltaBearing < epsBearing):
          Features.append(l_x[i])
          Features.append(l_y[i])
          Features.append(L_x[j])
          Features.append(L_y[j])
          Features.append(deltaBearing)

  plt.scatter(L_x , L_y,color='r')
  plt.scatter(l_x , l_y,color='b')
  plt.show()

  return Features
  