#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Vehicles following set paths along a road, with a constant velocity model.

Longer prediction than the first simulation.
"""

import numpy as np
import alarms, alarms_nb, motionModels, scoring
from collections import defaultdict
from time import time as timer
from regressor_fixedtime import Model as StepModel; stepmodel = StepModel('DT')
from regressor_2 import Model; model = Model('logit', True)


np.random.seed(64)
## the number of simulations to perform
nsims = 1000
## how often to utilize the motion model and to check for collision. (s)
timeres = .1
## the time duration of each simulation, and of the alarms' predictions. (s)
timelen = 2.5
## Each vehicle is started within a certain coordinate range, then moved
## backwards for 'timeback' seconds. Because the forward motion is partially
## random, this does not constrain the vehicles' positions to be within this
## coordinate range.
timeback = 2.
## The uncertainty in each vehicle's initial state is assumed to be Gaussian
## with this covariance matrix.
initialcovariance = np.diag([2., .5]) * .1
## name with which to save plots and tables; None = display but don't save
savename = 'sim2' # None #

times = np.array([timeres] * int(timelen/timeres))
noise = np.diag([2., .5])
MM1 = motionModels.MM_LineCV('left-right', noise)
MM2 = motionModels.MM_LineCV('right-down', noise)

v1 = np.random.normal(10., 5., nsims)
x1 = np.random.normal(3, 2., nsims) - timeback * v1 # was 5
vehicle1 = np.array((x1,v1)).T
v2 = np.random.normal(10., 5., nsims)
x2 = np.random.normal(-5, 2., nsims) - timeback * v2
vehicle2 = np.array((x2,v2)).T


        
print "running"
    
truth = []
results = defaultdict(list)
for sim in range(nsims):
    veh1 = vehicle1[sim,:]
    veh2 = vehicle2[sim,:]
    veh1 = motionModels.initialState_normal(veh1, initialcovariance)
    veh2 = motionModels.initialState_normal(veh2, initialcovariance)
    
    # find 'true' collision occurrences by using very high-res MCS alarm
    pred, rt = alarms.alarm_MCS(veh1, veh2, MM1, MM2, np.array(times), 10000)
    truth += [pred]
    results['optimal'].append((pred, rt))
    
    # estimators
    starttime = timer()
    result = alarms_nb.alarm_MCSroad(vehicle1[sim], initialcovariance, vehicle2[sim],
                                    initialcovariance, times.shape[0], timeres, noise, 10)
    time = timer() - starttime
    results['MC 10'].append((result, time))
    
    starttime = timer()
    result = alarms_nb.alarm_MCSroad(vehicle1[sim], initialcovariance, vehicle2[sim],
                                    initialcovariance, times.shape[0], timeres, noise, 100)
    time = timer() - starttime
    results['MC 100'].append((result, time))
    
    result = alarms.alarm_MCS(veh1, veh2, MM1, MM2, times, 1000)
    results['MC 1k'].append(result)
        
    result = alarms.alarm_UT_1(veh1, veh2, MM1, MM2, times)
    results['UT 1'].append(result)
    
    result = alarms.alarm_UT_2(veh1, veh2, MM1, MM2, times)
    results['UT 2'].append(result)
    
    result = alarms.alarm_UT_3(veh1, veh2, MM1, MM2, times)
    results['UT 3'].append(result)
    
    result = alarms.alarm_expected(veh1, veh2, MM1, MM2, times)
    results['EV'].append(result)
            
    result = alarms.alarm_EVbig(veh1, veh2, MM1, MM2, times)
    results['EV1m'].append(result)
    
    result = model.alarm(veh1, veh2, MM1, MM2, times)
    results['ML'].append(result)
    
    result = stepmodel.alarm(veh1, veh2, MM1, MM2, times)
    results['UT+ML'].append(result)



truth = np.array(truth)
ProbabilityOfCollision = sum(truth)/nsims
print "collisions "+str(ProbabilityOfCollision)

#scoring.plotROC(truth, results, savename)
bigTable = scoring.bigTable(truth, results, z_cost_vals = [1,10,100])
print bigTable
if not savename is None:
    bigTable.to_csv(savename+'_scores.csv')
