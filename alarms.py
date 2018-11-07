# -*- coding: utf-8 -*-
"""
Functions that calculate probability of collision, or make an alarm decision.

vehicle1,vehicle2 = initial_state objects, contain:
    mean, expected, utpoints, sample()
MM1,MM2 = motion model objects, contain:
    ndim, fullToXY() , expected() , sample() , pdf() , UTstep()
times = times at which to check for collision
    
Some models may have additional arguments.
    
returns (
         collision probability or 0-1 guess ,
         time it took to run the significant (repeating) part of the alarm )
"""
from collisionCheck import check as collisionCheck
import numpy as np
import time
import UnscentedTransform as UT

def _forceProb(prob):
    return min(max(prob, 0.), 1.)
    
def alarm_MCS(vehicle1, vehicle2, MM1, MM2, times, nSamples):
    """ Algorithm 1 in paper """
    starttime = time.time()
    state1 = vehicle1.sample(nSamples)
    state2 = vehicle2.sample(nSamples)
    for dt in times:
        MM1.sample(state1, dt)
        MM2.sample(state2, dt)
        collided = collisionCheck(MM1.fullToXY(state1),
                                  MM2.fullToXY(state2))
        state1 = state1[collided==False]
        state2 = state2[collided==False]
    totalCollided = 1 - float(state1.shape[0]) / nSamples
    return ( totalCollided , time.time() - starttime )
    
def alarm_MCS_2(vehicle1, vehicle2, MM1, MM2, times, nSamples):
    """ This version keeps checking samples that already had a collision.
        It's faster on average if collisions are unlikely. """
    starttime = time.time()
    state1 = vehicle1.sample(nSamples)
    state2 = vehicle2.sample(nSamples)
    Collided = np.zeros((nSamples,), dtype=bool) # array set to False
    for dt in times:
        MM1.sample(state1, dt)
        MM2.sample(state2, dt)
        collided = collisionCheck(MM1.fullToXY(state1), MM2.fullToXY(state2))
        Collided = Collided | collided
    totalCollided = float(np.sum(Collided)) / nSamples
    return ( totalCollided , time.time() - starttime )
    
    
def alarm_expected(vehicle1, vehicle2, MM1, MM2, times):
    """ The most naive approach, ignoring any probabilistic effects. """
    starttime = time.time()
    state1 = vehicle1.mean.copy()
    state2 = vehicle2.mean.copy()
    collided = False
    for dt in times:
        MM1.expected(state1, dt)
        MM2.expected(state2, dt)
        newcollide = collisionCheck(MM1.fullToXY(state1), MM2.fullToXY(state2))
        collided = collided | newcollide
    return ( collided+0. , time.time() - starttime )

def alarm_EVbig(vehicle1, vehicle2, MM1, MM2, times):
    """ The most naive approach, ignoring any probabilistic effects. """
    starttime = time.time()
    state1 = vehicle1.mean.copy()
    state2 = vehicle2.mean.copy()
    collided = False
    for dt in times:
        MM1.expected(state1, dt)
        MM2.expected(state2, dt)
        newcollide = collisionCheck(MM1.fullToXY(state1), MM2.fullToXY(state2), 3., 1.5)
        collided = collided | newcollide
    return ( collided+0. , time.time() - starttime )
    
    
def alarm_UT_1(vehicle1, vehicle2, MM1, MM2, times):
    """ Propagates the unscented sample points through each timestep and
        checks each for collision. Each sample point is effectively mapped to
        a boolean, which are merged into the final probability estimate. """
    points, weights = UT.getUTpoints(MM1.ndim + MM2.ndim)
    npts = points.shape[0]
    v1 = points[:,:MM1.ndim]
    v2 = points[:,MM1.ndim:]
    
    starttime = time.time()
    v1 = v1.dot(np.linalg.cholesky(vehicle1.cov).T) + vehicle1.mean
    v2 = v2.dot(np.linalg.cholesky(vehicle2.cov).T) + vehicle2.mean
    collided = np.zeros((npts,), dtype=bool)
    
    for dtime in times:
        MM1.expected(v1, dtime)
        MM2.expected(v2, dtime)
        xy1 = MM1.fullToXY(v1)
        xy2 = MM2.fullToXY(v2)
        newcollide = collisionCheck(xy1,xy2)
        collided = collided | newcollide
    totalCollided = np.sum(weights[collided])
    totalCollided = _forceProb(totalCollided)
    return ( totalCollided , time.time() - starttime )

    
def alarm_UT_2(vehicle1, vehicle2, MM1, MM2, times):
    """
    A different take on using the unscented transform for collision detection.
    This is not mathematically valid (it completely screws up the
    time-dependency) but gets much better results.
    """
    npoints = vehicle1.utpoints.shape[0]
    weights = np.tile(MM1.weights, (MM2.weights.shape[0],)) *\
              np.repeat(MM2.weights, MM1.weights.shape[0])
    
    starttime = time.time()
    v1 = vehicle1.utpoints.copy()
    v2 = vehicle2.utpoints.copy()
    collided = np.zeros((npoints**2,), dtype=bool)
    
    for dtime in times:
        v1 = MM1.UTstep(v1, dtime)
        v2 = MM2.UTstep(v2, dtime)
        xy1 = MM1.fullToXY(v1)
        xy2 = MM2.fullToXY(v2)
        xy1 = np.tile(xy1, (npoints,1))
        xy2 = np.repeat(xy2, npoints, 0)
        newcollide = collisionCheck(xy1,xy2)
        collided = collided | newcollide
    totalCollided = np.sum(weights[collided])
    totalCollided = _forceProb(totalCollided)
    return ( totalCollided , time.time() - starttime )


def alarm_UT_3(vehicle1, vehicle2, MM1, MM2, times):
    """
    A different take on using the unscented transform for collision detection.
    This is not mathematically valid (it completely screws up the
    time-dependency)
    uses UT for xy transformation as well
    """
    utpoints, weights = UT.getUTpoints(6) # 3 vehicle parts -> 6 total
    collided = np.zeros((utpoints.shape[0],), dtype=bool)
    outpoints = np.empty((utpoints.shape[0], 6))
    
    starttime = time.time()
    v1 = vehicle1.utpoints.copy()
    v2 = vehicle2.utpoints.copy()
    collided[:] = False
    totalCollided = 0.
    
    for dtime in times:
        v1 = MM1.UTstep(v1, dtime)
        v2 = MM2.UTstep(v2, dtime)
        xy1 = MM1.fullToXY(v1).copy()
        xy2 = MM2.fullToXY(v2).copy()
        meanxy1 = MM1.weights.dot(xy1[:,:2])
        meanxy2 = MM2.weights.dot(xy2[:,:2])
        meana1 = np.arctan2(MM1.weights.dot(np.sin(xy1[:,2])),
                            MM1.weights.dot(np.cos(xy1[:,2])))
        meana2 = np.arctan2(MM2.weights.dot(np.sin(xy2[:,2])),
                            MM2.weights.dot(np.cos(xy2[:,2])))
        xy1[:,:2] -= meanxy1
        xy1[:,2] -= meana1
        xy2[:,:2] -= meanxy2
        xy2[:,2] -= meana2
        # reformat angle residuals to be in +-pi
        xy1[:,2] = np.mod(xy1[:,2] + np.pi, 2*np.pi) - np.pi
        xy2[:,2] = np.mod(xy2[:,2] + np.pi, 2*np.pi) - np.pi
        cov1 = np.einsum(xy1, [0,1], xy1, [0,2], MM1.weights, [0], [1,2])
        eigval, eigvec = np.linalg.eigh(cov1)
        eigval = np.maximum(eigval, 0)
        U1 = (eigvec * np.sqrt(eigval)).T
        cov2 = np.einsum(xy2, [0,1], xy2, [0,2], MM2.weights, [0], [1,2])
        eigval, eigvec = np.linalg.eigh(cov2)
        eigval = np.maximum(eigval, 0)
        U2 = (eigvec * np.sqrt(eigval)).T
        outpoints[:,:3] = utpoints[:,:3].dot(U1)
        outpoints[:,:2] += meanxy1
        outpoints[:,2] += meana1
        outpoints[:,3:] = utpoints[:,3:].dot(U2)
        outpoints[:,3:5] += meanxy2
        outpoints[:,5] += meana2
        newcollide = collisionCheck(outpoints[:,:3] , outpoints[:,3:])
        #totalCollided = max(totalCollided, np.sum(weights[newcollide]))
        collided = collided | newcollide
    totalCollided = np.sum(weights[collided])
    totalCollided = _forceProb(totalCollided)
    return ( totalCollided , time.time() - starttime )