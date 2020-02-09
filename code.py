#Chargement de dépendances
import numpy as np
import matplotlib.pyplot as plt
import math

#Discrétisation
A=0
B=500
N=101 #Nombre de points de discrétisation 
Delta = (B-A)/(N-1)
discretization_indexes = np.arange(N) 
discretization = discretization_indexes*Delta 

#Paramètres du modèle
mu=-5
a = 50 
sigma2 = 12

#Données
observation_indexes = [0,20,40,60,80,100] 
depth = np.array([0,-4,-12.8,-1,-6.5,0])
observation = [discretization[i] for i in observation_indexes]
#Indices des composantes correspondant aux observations et aux componsantes non observées
unknown_indexes=list(set(discretization_indexes)-set(observation_indexes))

#Question 1

def covariance(dist, a, sigma2):
    return sigma2 * math.exp(- dist / a)

#Question 2

def mat_distances(x):
    mat_dist = np.zeros((len(x), len(x)))
    for (i, x_i) in enumerate(x):
        for (j, x_j) in enumerate(x):
            mat_dist[i, j] = abs(x_j-x_i)
    return mat_dist

#Question 3

def mat_covariance(x, a, sigma2):
    mat_cov = np.zeros((len(x), len(x)))
    mat_dist = mat_distances(x)
    for i in range(len(x)):
        for j in range(i, len(x)):
            dist = mat_dist[i, j]
            cov = covariance(dist, a, sigma2)
            mat_cov[i, j] = cov
            mat_cov[j, i] = cov
    return mat_cov


#print(mat_covariance(observation, a, sigma2))

#Question 4
def mat_distances2(x, x2):
    mat_dist = np.zeros((len(x), len(x2)))
    for (i, x_i) in enumerate(x):
        for (j, x_j) in enumerate(x2):
            mat_dist[i, j] = abs(x_j-x_i)
    return mat_dist

def mat_covariance_xy(x, x2, a, sigma2):
    mat_cov = np.zeros((len(x), len(x2)))
    mat_dist = mat_distances2(x, x2)
    for i in range((len(x))):
        for j in range(len(x2)):
            dist = mat_dist[i, j]
            cov = covariance(dist, a, sigma2)
            mat_cov[i, j] = cov
    return mat_cov

#print(mat_covariance_xy(discretization, observation, a, sigma2))

print(np.transpose(mat_covariance_xy(discretization, observation, a, sigma2)) == mat_covariance_xy(observation, discretization, a, sigma2))

