# Principal Component Analysis
Analyzing multidimensional cancer survey data to predict which of two respondents is at a higher risk (using principal component analysis)

### Notes:
- Data for this project was pulled from a school assignment. As such, it has been carefully selected so as to be centered around the origin, which makes construction of a covariance matrix possible.
- I skimmed over a lot of the finer mathematical details, specifically as to why this entire process *works* in the first place. For more reading on the subject, [this article](https://builtin.com/data-science/step-step-explanation-principal-component-analysis) may prove useful.

### Overview:

Principal component analysis is an innovative tool that was developed to solve the *curse of dimensionality*: How can we identify clusters, classify data points, and make inferences when working with high-dimensional data? 

In this project, a 100-question survey was issued to 1000 individuals, who were tracked for 15 years to see if they developed cancer or not. Using these survey responses, along with the knowledge of who did and did not test positive for cancer in the 15-year period, we hope to make predictions about two patients who also took the questionnaire (but whose likelihood of contracting cancer is presently unknown). Survey responses were numericized for mathematical analysis.

Accordingly, we have a 100-dimensional dataset that we hope to draw conclusions from. Using principal component analysis, we hope to reduce this data to a more manageable form, and thus make an informed prediction about the two patients in question.

### Goals:

1. Reduce the dimensionality of our dataset while maximally preserving its variance (average of squared distances from the mean, helps to distinguish data points).

2. Plot our data in two dimensions so that we may visually identify clusters, then use these patterns to conclude if two patients with unknown status are likely to develop cancer.

#### Reducing Dimensionality:

The desired subspace which maximizes the variance in our data can be constructed with the eigenvectors of the data's *covariance matrix*. In this case, our final variance can be expressed by the sum of these eigenvectors' corresponding eigenvalues.

In a two-dimensional context, this ideal subspace we are searching for is actually the data's *line of best fit*. In working with higher dimensions, however, a covariance matrix is needed.

First, we import the proper libraries, then pull in our data for further manipulation:

```

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

surveyData = pd.read_csv('surveyData.csv',header=None)

negativeResponses = surveyData.loc[surveyData[100]==0].drop(columns=100).values.transpose()
positiveResponses = surveyData.loc[surveyData[100]==1].drop(columns=100).values.transpose()
totalResponses = surveyData.loc[surveyData[100]>=0].drop(columns=100).values.transpose()
patientOneResponses = surveyData.loc[surveyData[100]<0].drop(columns=100).values[0,:]
patientTwoResponses = surveyData.loc[surveyData[100]<0].drop(columns=100).values[1,:]
```

A covariance matrix `C` can be constructed using the following formula, where `X` is the matrix of data and `n` is the number of data points:

![Covariance Formula](https://github.com/spencermyoung513/Principal-Component-Analysis/blob/main/Equation%20Images/Equation1.PNG)

The following lines of code use this formula to build a covariance matrix for the entire dataset:

```
numRespondents = totalResponses.shape[1]
C = (1/(numRespondents-1))*(totalResponses @ np.transpose(totalResponses))
```

Our next step is to examine whether a two-dimensional representation of our data is practical. How much variance in our data should we aim to preserve in this process? 99.9% feels like a good benchmark. 

Keeping in mind that total variance can be calculated by finding the sum of the eigenvalues associated with the eigenvectors we use to construct our subspace, the following function takes in a covariance matrix and a desired level of precision, then returns a list of eigenvectors that will match this criteria:

```
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
			return eigenList
```

After running `optimalBasis(C, 0.999)`, it is easy to check that two eigenvectors will form the basis of an adequate subspace without losing a significant amount of variance. This will allow us to make a two-dimensional projection of our survey data!

Forming the matrix `U` with these eigenvectors, we can then project each of our data groups onto a 2D subspace:

```
U = np.array(basisEigVectors)
negativeResponses2D = U @ negativeResponses
positiveResponses2D = U @ positiveResponses
patientOneResponses2D = U @ patientOneResponses
patientTwoResponses2D= U @ patientTwoResponses
```

#### Plotting our Data:

Now that our data is in two dimensions, we can plot it on a simple x-y plane:

```
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
```

Running `createPlot2D(negativeResponses2D, positiveResponses2D, patientOneResponses2D, patientTwoResponses2D)` produces the following figure (blue dots are patients who never developed cancer, red dots are patients who did):

![Comparison Plot](https://github.com/spencermyoung513/Principal-Component-Analysis/blob/main/responsesComp.png)

### Conclusion:

Patient one, represented as a green star in the figure, is much closer to the cluster of individuals who were positive for cancer, while patient two, represented with a purple star, is much closer to the individuals who were negative for cancer. From this quick look at the data, it is clear to see that, based on their responses to the questionnaire, patient one is more likely to develop cancer than patient two. 
