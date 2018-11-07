# -*- coding: utf-8 -*-
"""
Simulations for vehicles following the bicycle model.

Format is very similar to simulate_1, which has more documentation.
"""

import numpy as np
import alarms, alarms_nb, motionModels, scoring
from collections import defaultdict
from time import time as timer
from regressor_fixedtime import Model as StepModel; stepmodel = StepModel('DT')
from regressor_3 import Model; model = Model('MLP')


np.random.seed(64)
nsims = 1000
timeres = .1
timelen = 1.
timeback = 0.5
initialnoise = np.diag([1., 1., .5, .1, 0.05, 0.01]) * .1
savename = 'sim3'


times = np.zeros((int(timelen/timeres),)) + timeres
noise = np.diag([1., 1., .5, .1, 0.05, 0.01])
MM1 = motionModels.MM_Bicycle(noise)
MM2 = motionModels.MM_Bicycle(noise)

a = np.random.normal(0., 1., nsims)
w = np.random.uniform(-.5, .5, nsims)
v = np.random.normal(10., 5., nsims) - timeback * a
th = np.random.uniform(-np.pi, np.pi, nsims)
x = np.random.normal(0., 5., nsims) - timeback * v * np.cos(th)
y = np.random.normal(0., 5., nsims) - timeback * v * np.sin(th)
vehicle1 = np.array((x,y,th,v,a,w)).T
a = np.random.normal(0., 1., nsims)
w = np.random.uniform(-.5, .5, nsims)
v = np.random.normal(10., 5., nsims) - timeback * a
th = np.random.uniform(-np.pi, np.pi, nsims)
x = np.random.normal(0., 5., nsims) - timeback * v * np.cos(th)
y = np.random.normal(0., 5., nsims) - timeback * v * np.sin(th)
vehicle2 = np.array((x,y,th,v,a,w)).T


    
truth = []
results = defaultdict(list)
for sim in range(nsims):
    veh1 = vehicle1[sim,:]
    veh2 = vehicle2[sim,:]
    veh1 = motionModels.initialState_normal(veh1, initialnoise)
    veh2 = motionModels.initialState_normal(veh2, initialnoise)
    
    # find 'real' collision occurrences by using very high-res particle alarm
    pred, rt = alarms.alarm_MCS(veh1, veh2, MM1, MM2, times, 10000)
    truth += [pred]
    results['optimal'].append((pred, rt))
    
    # estimators
    starttime = timer()
    result = alarms_nb.alarm_MCSbike(vehicle1[sim], initialnoise, vehicle2[sim],
                                    initialnoise, times.shape[0], timeres, noise, 10)
    time = timer() - starttime
    results['MC 10'].append((result, time))
    
    starttime = timer()
    result = alarms_nb.alarm_MCSbike(vehicle1[sim], initialnoise, vehicle2[sim],
                                    initialnoise, times.shape[0], timeres, noise, 100)
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
    results['expected'].append(result)
            
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