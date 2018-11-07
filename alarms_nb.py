#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This code is equivalent to the Monte Carlo alarm from alarms.py, but uses the Python-compiling package numba. For Monte Carlo algorithms with few samples, Python's baggage from loops, numpy preprocessing, etc was the main bottleneck and this code is substantially faster. For high-sample MC algorithms, the difference is minor.
last mod 6/18/18
"""

from collisionCheck import collisionCheck
import numpy as np
import numba

@numba.jit(numba.f8[:](numba.f8[:,:]), nopython=True)
def randMvNorm(cholesky):
    result = np.zeros((cholesky.shape[0],))
    for k in range(cholesky.shape[0]):
        result[k:] += cholesky[k:,k] * np.random.standard_normal()
    return result

@numba.jit(numba.void(numba.f8[:],numba.f8), nopython=True)
def moveBicycle(state, dt):
    state[0] += np.cos(state[2])*state[3]*dt
    state[1] += np.sin(state[2])*state[3]*dt
    state[2] += state[5]*dt
    state[3] += state[4]*dt

@numba.jit(numba.void(numba.f8[:],numba.f8, numba.f8[:,:]), nopython=True)
def sampleBicycle(state, dt, noise):
    moveBicycle(state, dt)
    state += randMvNorm(noise)*np.sqrt(dt)

@numba.jit(nopython=True)
def singleBicycle(mean1, cov1, mean2, cov2, ntimes, dt, noise):
    sample1 = mean1 + randMvNorm(cov1)
    sample2 = mean2 + randMvNorm(cov2)
    for t in range(ntimes):
        sampleBicycle(sample1, dt, noise)
        sampleBicycle(sample2, dt, noise)
        if collisionCheck(sample1, sample2): return 1
    return 0


@numba.jit(numba.void(numba.f8[:],numba.f8, numba.f8[:,:]), nopython=True)
def sampleRoad(state, dt, noise):
    state[0] += state[1]*dt
    state += randMvNorm(noise)*np.sqrt(dt)
    
@numba.jit(numba.f8[:](numba.f8[:]), nopython=True)
def roadLocLR(state):
    return np.array((state[0], -5., 0.))

@numba.jit(numba.f8[:](numba.f8[:]), nopython=True)
def roadLocRD(state):
    pos = state[0]
    if pos <= -8:
        return np.array((-pos, 1.65, 3.141593))
    elif pos > 7.15818455:
        return np.array((-1.65, -.8418154 - pos))
    else:
        p = pos + 8.
        return np.array((8 - 9.65*np.sin(p/9.65),
                         9.65*np.cos(p/9.65) - 8,
                         p*.0779235 - np.sin(p*.31169401)*.2 - 3.141593))
  
@numba.jit(nopython=True)
def singleRoad(mean1, cov1, mean2, cov2, ntimes, dt, noise):
    sample1 = mean1 + randMvNorm(cov1)
    sample2 = mean2 + randMvNorm(cov2)
    for t in range(ntimes):
        sampleRoad(sample1, dt, noise)
        sampleRoad(sample2, dt, noise)
        if collisionCheck(roadLocLR(sample1), roadLocRD(sample2)): return 1
    return 0


@numba.jit(nopython=True)
def alarm_MCSbike(mean1, cov1, mean2, cov2, ntimes, dt, noiseMatrix, nsamples):
    totalCollided = 0
    cov1 = np.linalg.cholesky(cov1)
    cov2 = np.linalg.cholesky(cov2)
    noise = np.linalg.cholesky(noiseMatrix)
    for k in range(nsamples):
        totalCollided += singleBicycle(mean1, cov1, mean2, cov2, ntimes, dt, noise)
    return float(totalCollided) / nsamples
@numba.jit(nopython=True)
def alarm_MCSroad(mean1, cov1, mean2, cov2, ntimes, dt, noiseMatrix, nsamples):
    totalCollided = 0
    cov1 = np.linalg.cholesky(cov1)
    cov2 = np.linalg.cholesky(cov2)
    noise = np.linalg.cholesky(noiseMatrix)
    for k in range(nsamples):
        totalCollided += singleRoad(mean1, cov1, mean2, cov2, ntimes, dt, noise)
    return float(totalCollided) / nsamples
