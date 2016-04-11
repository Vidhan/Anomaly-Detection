import math
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

def getTrainingSet():
	x = np.array([[1, 29],[30, 4],[15, 6],[7,18],[9,10],[11,12],[12,14],[1,16],[17,1],[9,20],[21,2],[23,24],[25,62],[2,20],[29,35],[30,32]])
	m, n = np.shape(x)
	x=x.transpose()
	return x, m, n

def calculateMu (x,m):
	mu = x.sum(axis=1)
	mu = mu/m
	mu = [mu]
	mu = np.transpose(mu)
	return mu

def calculateCovariance (x,m,n,mu):
	shape = (n,n)
	cov = np.zeros(shape)
	for i in range(0,m):
		arr = x[:,[i]] - mu
		arrTranspose = np.transpose(arr)
		cov = cov +  np.dot(arr,arrTranspose)
	cov = cov/m
	return cov

def calculateP(input,n, mu, cov):
	input = np.transpose(input)
	exponent1 = input-mu
	exponent2 = np.transpose (exponent1)
	exponent3 = inv(cov)
	firstMultiply = np.dot(exponent2,exponent3)
	secondMultiply = np.dot(firstMultiply,exponent1)
	power = ((-1/2.0)*secondMultiply).item(0)
	return math.exp(power)/(math.pow(det(cov),0.5)/math.pow((math.pi*2), -1*n/2.0))

def main():
	x,m,n = getTrainingSet()
	print
	print "The training set consists of "+str(m)+" data samples"
	print "Each data sample has "+str(n)+" features"
	print "******************************************************"
	print
	print "Calculating Multivariate Gaussian Distribution Parameters"
	print
	mu=calculateMu(x,m)
	print "Mu is"
	print mu
	print
	cov=calculateCovariance(x,m,n,mu)
	print "Covariance matrix is"
	print cov
	print "******************************************************"
	print
	input = np.array([[2,10]])
	p = calculateP(input,n,mu,cov)
	print p

main()	

