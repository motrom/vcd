# -*- coding: utf-8 -*-
"""
Classes that predict vehicle's current and future position.

Two state formats:
    full - the entire state, determines motion (along with any fixed parameters)
    XY - [x,y,angle] used to check for collisions
    
functions in each MM (Motion Model) class:
    fullToXY
    expected - return expected value of state at a time
    sample - return independent random samples of full state after motion
    UTstep - take the expected value a set of unscented points over a timestep,
             then compute the resulting distribution and the next set of
             unscented points

functions in each initialState class:
    mean - the expected position of the vehicle
    utpoints - the unscented transform points for that
    sample() - return independent random samples of full state at current time
"""
import numpy as np
#from scipy.stats import multivariate_normal as rmvnorm
from roadLoc import roadLoc
import UnscentedTransform as UT
        

class MM_LineCV():
    """ Vehicle follows a predetermined road. The roads are based on an
        intersection in SUMO, as explained in the roadLoc module.
        X_{t+del_t} = F_{del_t} X_t + N * del_t
        F_{del_t} = [1, del_t; 0, 1]
        X_t = state at time t
        N = noiseMatrix
        
        state = [displacement, velocity]
        noiseMatrix = per-second variance in motion equation """
        
    def __init__(self, route, noiseMatrix):
        self.cov = noiseMatrix
        self.route = route
        points, weights = UT.getUTpoints(2, 1.)
        self.points = points
        self.weights = weights
        self.ndim = 2
        
    def fullToXY(self, state):
        if state.ndim == 1:
            return np.array(roadLoc(state[0], self.route))
        else:
            return roadLoc(state[:,0], self.route)
        
    def expected(self, state, time):
        state[...,0] += state[...,1]*time
        
    def sample(self, state, time, nsamples = 1):
        self.expected(state, time)
        if state.ndim == 1:
            state += np.random.multivariate_normal([0,0], self.cov*time, size=nsamples)
        else:
            state += np.random.multivariate_normal([0,0], self.cov*time, size=state.shape[0])
        
    def UTstep(self, points, time):
        self.expected(points, time)
        mean = self.weights.dot(points)
        points -= mean
        cov = (points.T * self.weights).dot(points) + self.cov*time
        return mean + self.points.dot(np.linalg.cholesky(cov).T)

        
class MM_Bicycle():
    """
    The bicycle model has been used a lot for car motion, a few papers defining
    it were cited in the paper.
    X_{t+del_t} = B(X_t , del_t) + N * del_t
    B = bicycle model
    X_t = state at time t
    N = noiseMatrix
    
    state = [x, y, angle, velocity, acceleration, angular velocity]
    noiseMatrix = covariance matrix of motion noise (for roughly 1 second)
    """
    
    def __init__(self, noiseMatrix):
        self.cov = noiseMatrix
        points, weights = UT.getUTpoints(6, 1.)
        self.points = points
        self.weights = weights
        self.ndim = 6
        
    def fullToXY(self, state):
        if state.ndim==1:
            return state[:3]
        else:
            return state[:,:3]
        
    def expected(self, state, time):
        state[...,0] += np.cos(state[...,2])*state[...,3]*time
        state[...,1] += np.sin(state[...,2])*state[...,3]*time
        state[...,2] += state[...,5]*time
        state[...,3] += state[...,4]*time
            
    def sample(self, state, time, nsamples = 1):
        self.expected(state, time)
        if state.ndim == 1:
            state += np.random.multivariate_normal([0.]*6, self.cov*time, size=nsamples)
        else:
            state += np.random.multivariate_normal([0.]*6, self.cov*time, size=state.shape[0])
            
    def UTstep(self, points, time):
        self.expected(points, time)
        mean = self.weights.dot(points)
        # compute angle mean properly
        cosmean = self.weights.dot(np.cos(points[:,2]))
        sinmean = self.weights.dot(np.sin(points[:,2]))
        mean[2] = np.arctan2(sinmean, cosmean)
        #
        points -= mean
        # reformat angle residuals to be in +-pi
        rectify(points[:,2])
        #
        cov = (points.T * self.weights).dot(points) + self.cov*time
        outpoints = self.points.dot(np.linalg.cholesky(cov).T) + mean
        # reformat angles to be in +-pi
        rectify(outpoints[:,2])
        return outpoints
        
            
        
class initialState_normal():
    """ The vehicle's current state X_0 is normally distributed. """
    
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
        chol = np.linalg.cholesky(cov).T
        utpoints, utweights = UT.getUTpoints(mean.shape[0])
        self.utpoints = mean + utpoints.dot(chol)
    def sample(self, nsamples):
        return np.random.multivariate_normal(self.mean, self.cov, size=nsamples)
    
    
def rectify(theta):
    theta[theta >  np.pi] -= 2*np.pi
    theta[theta < -np.pi] += 2*np.pi
    return theta