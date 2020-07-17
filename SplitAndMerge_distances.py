#
#   Copyright (C) 2014 Leonardo Rocha Olivi, Eleri Cardozo
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

# function result = SplitAndMerge(global_poses)
# 
# argumento:
# global_poses: /perception/laser/n/global_poses

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def SplitAndMerge(distances, laser_range):
    # --------------------------------------
    # Distancia maxima para considerar que um ponto pertence a uma reta
    # epsilon = 20 # [mm] 
    # epsilon = 50 # [mm] 
    epsilon = 80 # [mm] 
    # ---------------------------------------

    # ---------------
    # Split And Merge
    # O Split-And-Merge retorna as retas que formam os obstaculos
    # ---------------

    # Numero total de medidas adquiridas pelo sensor
    n = len(distances)
    ang = laser_range[0]

    X = []
    Y = []
    # obtem os pontos da reta no frame global
    for i in range(n):
        X.append(distances[i]*np.cos(ang*np.pi/180))
        Y.append(distances[i]*np.sin(ang*np.pi/180))
        ang += laser_range[2]

    X = np.asarray(X)
    Y = np.asarray(Y)

    # Os indices dos pontos inicial e final para analise.
    # Dois pontos formam uma reta.
    i1 = 0 # indice do primeiro ponto que forma a reta em analise
    i2 = n # indice do segundo ponto que forma a reta em analise

    # Auxiliar que armazena os pontos que formam a reta encontrada
    PointsX = [] 
    PointsY = [] 
    NPoints = []

    # Inicio do SPLIT AND MERGE
    while(i1 is not i2): # enquanto i1 for diferente de i2
        
        # ------------------------------------------------------------
        # Obtenha os pontos P1(x1,y1) e P2(x2,y2) para formar uma reta
        # ------------------------------------------------------------
        # Ponto 1
        x1 = X[i1]
        y1 = Y[i1]
        # Ponto 2
        x2 = X[i2-1]
        y2 = Y[i2-1]
        
        # Equacao geral da reta: a * x + b * y + c = 0
        # a = y1 - y2
        a = y1 - y2
        # b = x2 - x1
        b = x2 - x1
        # c = x1.y2 - x2.y1
        c = x1 * y2 - x2 * y1

        d = []
        # -------------------------------------------------------------------
        # Calculo das distancias de cada ponto dentro do intervalo de analise
        # -------------------------------------------------------------------
        # Distancias de cada um dos pontos P0(x0,y0) contidos no intervalo de 
        # analise (ou seja, entre i1 e i2) ate a reta obtida:
        # d = |a * x0 + b * y0 + c | / sqrt(a^2 + b^2)
        for i in range(i1,i2):
            d.append(abs(a*X[i] + b * Y[i] + c)/np.sqrt(a**2+b**2))

        # -------------------------------------------------------------------
        # Encontre a maior distancia entre a reta e os pontos P0
        # -------------------------------------------------------------------
        maxd = max(d)

        d = np.asarray(d)

        # -------------------------------------------------------------------
        # Avaliar se o ponto de maior distancia pertence a reta de analise
        # -------------------------------------------------------------------
        if (maxd > epsilon): # Se o ponto nao pertence a reta
            # -------------
            # Fase de SPLIT
            # -------------
            # O ponto em questao nao pertence a reta de analise.
            # Este ponto deve ser utilizado para formar outra reta de analise
            
            # Encotre o indice do ponto que possui a maior distancia.
            # Se houver mais de um ponto, utilize o primeiro que encontrar.
            ind = np.where( maxd == d)
            # Gera novo subgrupo de analise (SPLIT)
            # O indice encontrado refere-se ao subgrupo analisado e nao ao
            # indice global, ou seja, referente a todos os dados.
            # Por isso, soma-se o indice i1 ao indice encontrado. Desconta-se 1
            # porque a primeira posicao de um vetor do MATLAB difere do C++. 
            # No C++ o primeiro indice de um vetor vale 0. No MATLAB vale 1.
            i2 = i1 + ind[0][0] 
            # O indice i1 permanece inalterado nesta fase.
        else: # Se o ponto pertence a reta
            # -----
            # Merge
            # -----
            # A reta foi encontrada. Armazene-a para analises posteriores.
            PointsX.append(x1)
            PointsX.append(x2)
            PointsY.append(y1)
            PointsY.append(y2)
            NPoints.append((i2-i1))
            NPoints.append(0)
            # Altere os indices do intervalo de analise para continuar
            i1 = i2 # i1 assume o valor de i2
            i2 = n  # i2 assume novamente o ultimo valor possivel

    # Fim do SPLIT AND MERGE


    # Plotagem das retas (opcional)
    # plt.plot(PointsX, PointsY)
    # plt.show()

    # retorna pontos encontrados
    return PointsX, PointsY, NPoints