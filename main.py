# We want to be able to reduce the dimensionality of data in order to make accurate predictions based on a dataset
# Principal component analysis will accomplish this goal
# To project onto two dimensional space, we want to choose a plane that maximizes the variance in projected data (allows us to distinguish different points)
# This plane is a subspace spanned by the eigenvectors of the data's covariance matrix. Variance of the data is the sum of the eigenvalues

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
optimalVariance takes in a covariance matrix, C, and a float pctPreserved in (0,1) specifying what proportion of original variance is desired

Returns a list of eigenvectors eigenList = [u_1, u_2, ... , u_k], such that eigenList has the fewest number of eigenvectors (up to k) to achieve pctPreserved of the original variance of the dataset
'''

def optimalBasis(C, pctPreserved):

	if pctPreserved <= 0 or pctPreserved >= 1:
		print("Invalid pctPreserved entered.")
		return

	eVals, eVectors = np.linalg.eig(C)
	currentPctPreserved = 0.00
	currentVariance = 0.00
	totalVariance = np.sum(eVals)
	eigenList = []
	counter = 0
	
	for i in range(eVals.size):
		counter += 1
		currentVariance += eVals[np.argsort(eVals)[-(i+1)]]
		eigenList.append(eVectors[:, np.argsort(eVals)[-(i+1)]])
		currentPctPreserved = currentVariance / totalVariance
		if currentPctPreserved >= pctPreserved:
			print("Successfully found " + str(counter) + " eigenvectors to meet requirements. Percent of variance preserved if projected to new subspace: " + str(round(currentPctPreserved,6)))
			return eigenList

'''
createPlot takes in up to four 2xn matrices Z1, Z2, Z3, and Z4, and plots their data on a two-dimensional plane, saving the figure as responsesComp.png
'''
def createPlot2D(Z1=[],Z2=[],Z3=[],Z4=[]):
  fig = plt.figure()
  graph = fig.add_subplot(111)

  if len(Z1) > 0:
    Y1 = np.reshape(Z1,(2,-1))
    graph.scatter(Y1[0,:], Y1[1,:], s=2, c='b', marker="o")
  if len(Z2) > 0:
    Y2 = np.reshape(Z2,(2,-1))
    graph.scatter(Y2[0,:], Y2[1,:], s=2, c='r', marker="o")
  if len(Z3) > 0:
    Y3 = np.reshape(Z3,(2,-1))
    graph.scatter(Y3[0,:], Y3[1,:], s=100, c='g', marker="*")
  if len(Z4) > 0:
    Y4 = np.reshape(Z4,(2,-1))
    graph.scatter(Y4[0,:], Y4[1,:], s=100, c='purple', marker="*")

  fig.savefig("responsesComp.png") 

'''

BEGIN MAIN CODE HERE

'''

# Dataset. 1000 respondents, 100 questions (question #101 is -1, 0, or 1 to classify type of respondent)
surveyData = pd.read_csv('surveyData.csv',header=None)

negativeResponses = surveyData.loc[surveyData[100]==0].drop(columns=100).values.transpose()
positiveResponses = surveyData.loc[surveyData[100]==1].drop(columns=100).values.transpose()
totalResponses = surveyData.loc[surveyData[100]>=0].drop(columns=100).values.transpose()
patientOneResponses = surveyData.loc[surveyData[100]<0].drop(columns=100).values[0,:]
patientTwoResponses = surveyData.loc[surveyData[100]<0].drop(columns=100).values[1,:]
numRespondents = totalResponses.shape[1]

# Covariance matrix for dataset
C = (1/(numRespondents-1))*(totalResponses @ np.transpose(totalResponses))

# basisEigVectors is a list of eigenvectors that make up a basis for the optimal projectional subspace
basisEigVectors = optimalBasis(C,0.999)

# U is the transformation matrix which will shrink our data to the proper number of dimensions
U = np.array(basisEigVectors)

if U.shape[0] == 2:

	negativeResponses2D = U @ negativeResponses
	positiveResponses2D = U @ positiveResponses
	patientOneResponses2D = U @ patientOneResponses
	patientTwoResponses2D= U @ patientTwoResponses

	createPlot2D(negativeResponses2D, positiveResponses2D, patientOneResponses2D, patientTwoResponses2D)

else:

	print("Cannot display projected data (projection is not 2-dimensional).")