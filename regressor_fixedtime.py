""" This experiment was not presented in the paper. It simulated two normally
distributed vehicles in a single time frame, then developed a model for the
probability of collision in that time frame. This could be used in tandem with
movement predictions, for instance the unscented transform, to more quickly predict
collisions. While its runtime was excellent, its accuracy was poor despite being
trained on 17x10^6 examples. I suspect that training a model that is both accurate
and generalizable (that works on multiple scenarios, not just the one simulated
in training) will require a great deal of computational resources.

                   

current idea: shrink space by using relative angle, only positive x,y,angle
harder to get distribution - must use UT
full UT = 12 dim
fake
"""

from collisionCheck import check as collisionCheck
import numpy as np
from sklearn.model_selection import train_test_split
from scoring import AUC,ROC
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import time
import UnscentedTransform as UT

model_name = 'step'

xy_bound = 15
std_xy_bound = 10
std_angle_bound = np.pi / 3**.5

def rectify(theta):
    theta[theta > np.pi] -= 2*np.pi
    theta[theta > np.pi] -= 2*np.pi
    theta[theta < -np.pi] += 2*np.pi
    theta[theta < -np.pi] += 2*np.pi
    return theta

def createTruth(npoints, nrepeats=3000):
    """
    Generates a range of initial states for both vehicles, then simulates from
    each one a certain number of times.
    """
    ## generating a grid of features
    ndim = 8
    res = int(npoints**(1./ndim))
    npoints = res**ndim
    X_0 = np.empty((npoints, ndim))
    x_0 = np.linspace(-1.,1.,res)
    for feature in range(4):
        X_0[:,feature] = np.repeat(np.tile(x_0, res**feature), res**(ndim-feature-1))
    x_0 = np.linspace(0.,1.,res)
    for feature in range(4,8):
        X_0[:,feature] = np.repeat(np.tile(x_0, res**feature), res**(ndim-feature-1))
    ## randomly generating features
#    X_0 = np.random.uniform(-1,1,size=(npoints,4))
#    X_0 = np.append(X_0, np.random.uniform(0,1,size=(npoints,4)),axis=1)
    
    egocar = np.zeros((nrepeats,3))
    altcar = np.zeros((nrepeats,3))
    samples = np.random.normal(size=(4,nrepeats))
    
    y_0 = np.empty((npoints,))
    for point in range(npoints):
        feats = X_0[point,:]
        altcar[:,0] = feats[0] * xy_bound + std_xy_bound * feats[4] * samples[0]
        altcar[:,1] = feats[1] * xy_bound + std_xy_bound * feats[5] * samples[1]
        angle = feats[2] * np.pi + std_angle_bound * feats[6] * samples[2]
        egocar[:,2] = rectify(angle)
        angle = feats[3] * np.pi + std_angle_bound * feats[7] * samples[3]
        altcar[:,2] = rectify(angle)
        y_0[point] = np.sum(collisionCheck(egocar, altcar))/nrepeats
    
    np.save(model_name+'_X.npy', X_0)
    np.save(model_name+'_Y.npy',y_0)
    
    
Vlen = 2.5
Vwid = 1.
angle_var = np.pi / 3**.5
def vectorTruth(feat, cov, chol, samples, truths, raw_samples, k):
    feat[8] = (k % 3)*.45 - .45
    k = k // 3
    feat[7] = (k % 3)*.45 - .45
    k = k // 3
    feat[6] = (k % 3)*.45 - .45
    k = k // 3
    feat[5] = (k % 4)*angle_var / 3. + .01
    k = k // 4
    feat[4] = (k % 4) * 3.33 + .01
    k = k // 4
    feat[3] = (k % 4) * 3.33 + .01
    k = k // 4
    feat[2] = (k % 10) * np.pi / 9.
    k = k // 10
    feat[1] = (k % 20) * .5
    k = k // 20
    feat[0] = k * .5
    cov[0,0] = feat[3] * feat[3]
    cov[0,1] = feat[6] * feat[3] * feat[4]
    cov[0,2] = feat[7] * feat[3] * feat[5]
    cov[1,0] = cov[0,1]
    cov[1,1] = feat[4] * feat[4]
    cov[1,2] = feat[8] * feat[4] * feat[5]
    cov[2,0] = cov[0,2]
    cov[2,1] = cov[1,2]
    cov[2,2] = feat[5] * feat[5]
    chol[:] = np.linalg.cholesky(cov)
    samples[:,0] = feat[0] + raw_samples[:,0] * chol[0,0]
    samples[:,1] = feat[1] + raw_samples[:,0] * chol[1,0] + raw_samples[:,1]*chol[1,1]
    samples[:,2] = feat[2] + raw_samples[:,0] * chol[2,0] +\
                    raw_samples[:,1]*chol[2,1] + raw_samples[:,2] * chol[2,2]
    samples[:,2] = np.mod(samples[:,2], np.pi)
    samples[:,3] = np.cos(samples[:,2])
    samples[:,5] = np.abs(samples[:,3])
    samples[:,4] = np.sin(samples[:,2])
    samples[:,6] = np.abs(samples[:,4])
    truths[:] = True
    truths &= abs(samples[:,3]*samples[:,0] + samples[:,4]*samples[:,1]) -\
                    samples[:,5]*Vlen - samples[:,6]*Vwid < Vlen
    truths &= abs(samples[:,4]*samples[:,0] - samples[:,3]*samples[:,1]) -\
                    samples[:,6]*Vlen - samples[:,5]*Vwid < Vwid
    truths &= samples[:,0] - samples[:,5]*Vlen - samples[:,6]*Vwid < Vlen
    truths &= samples[:,1] - samples[:,6]*Vlen - samples[:,5]*Vwid < Vwid
    return np.mean(truths)
        
def createTruth2():
    size = 20*20*10*4*4*4*3*3*3
    X_0 = np.empty((size, 9))
    y_0 = np.empty((size,))
    raw_samples = np.random.standard_normal(size=(2000 , 3))
    samples = np.empty((2000, 7))
    truths = np.empty((2000,), dtype=bool)
    cov = np.zeros((3,3))
    chol = np.zeros((3,3))
    
    for k in range(size):
        y_0[k] = vectorTruth(X_0[k], cov, chol, samples, truths, raw_samples, k)
    
    np.save(model_name+'_X.npy', X_0)
    np.save(model_name+'_Y.npy',y_0)
    
    
def scoreModel(truth, pred, logit):
    """ Gathers R-squared (regression quality measure) and area-under-ROC
    (classification quality measure); optimal value for both is 1"""
    if logit:
        pred = 1./(1+np.exp(-pred))
    else:
        pred = np.minimum(np.maximum(pred,0.),1.)
    print( "test R^2 {:.3f}".format(
        1-np.sum((pred - truth)**2.)/np.sum((np.mean(truth)-truth)**2.) ))
    print( "AUC {:.3f}".format( AUC(*ROC(truth, pred)) ))
     
    
def trainModels():
    """ Loads the data created by createTruth() and tries several regressors
    scores each and saves models (currently only saving MLP) """
    X_0 = np.load(model_name+'_X.npy')
    y_0 = np.load(model_name+'_Y.npy')
    
    ## Transforms the probability values using the logit function, which is
    ## often used in classification models. Did not have a major effect.
    logit = False
    
    print( "y mean = {:f} , RMSR = {:f}".format(np.mean(y_0), np.std(y_0) ))
    
    X_train, X_test, y_train, y_test = train_test_split(X_0,y_0,test_size=.001)
    if logit:
        y_train[y_train > .9999] = .9999
        y_train[y_train < .0001] = .0001
        y_train = np.log(y_train / (1 - y_train))
    
    ## multi-layer perceptron
    print("MLP")
    mlp = MLPRegressor(hidden_layer_sizes=(100,4), tol=1e-10, max_iter=500)
    mlp.fit(X_train, y_train)
    print( "train R^2 {:.3f}".format( mlp.score(X_train, y_train) ))
    scoreModel(y_test, mlp.predict(X_test), logit)
    ff = time.time()
    mlp.predict(X_0[:100])
    print( "runtime for 100 samples: {:.1e}".format( time.time()-ff ))

    ## gradient boosting, a slow but highly regarded ensemble regressor
    print("GBR")
    gbr = GradientBoostingRegressor(max_depth=4)# default 3
    gbr.fit(X_train,y_train)
    print( "train R^2 {:.3f}".format( gbr.score(X_train, y_train) ))
    scoreModel(y_test, gbr.predict(X_test), logit)
    ff = time.time()
    gbr.predict(X_0[:100])
    print( "runtime for 100 samples: {:.1e}".format( time.time()-ff ))

    ## regression trees, very fast in training and use, but easily overfit data
    print("DT")
    dt = DecisionTreeRegressor(max_depth=15)
    dt.fit(X_train,y_train)
    print( "train R^2 {:f}".format( dt.score(X_train, y_train) ))
    scoreModel(y_test, dt.predict(X_test), logit)
    ff = time.time()
    dt.predict(X_0[:100])
    print( "runtime for 100 samples: {:.1e}".format( time.time()-ff ))
    
    ## only saved MLP
    np.save(model_name+'_MLP.npy',mlp)
    
    
class Model():
    """
    The formatting of features was difficult to generalize across examples, so
    each example has an object that formats a scenario and applies the model.
    """
    
    def __init__(self, name='MLP'):
        self.model=np.load(model_name+'_'+name+'.npy').item()
        self.bin = np.zeros((0,9))
        
    def alarm(self, vehicle1, vehicle2, MM1, MM2, times):
        utpoints, weights = UT.getUTpoints(4) # 3 vehicle parts -> 6 total
#        XY1 = np.empty((utpoints.shape[0], 3))
#        XY2 = np.empty((utpoints.shape[0], 3))
        xyr = np.empty((utpoints.shape[0], 4))
        covr = np.zeros((4,4))
        temp_x = np.empty((utpoints.shape[0],))
        features = np.empty((len(times),8))
        
        starttime = time.time()
        v1 = vehicle1.utpoints.copy()
        v2 = vehicle2.utpoints.copy()
        totalpred = 0.
        
        for j,dtime in enumerate(times):
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
            cov2 = np.einsum(xy2, [0,1], xy2, [0,2], MM2.weights, [0], [1,2])
#            eigval, eigvec = np.linalg.eigh(cov1)
#            eigval = np.sqrt(np.maximum(eigval, 0))
#            np.einsum(utpoints[:,:3], [1,0], eigval, [0], eigvec, [2,0], [1,2],
#                      out=XY1)
#            XY1[:,:2] += meanxy1
#            XY1[:,2] += meana1
#            cos1 = np.cos(XY1[:,2])
#            sin1 = np.sin(XY1[:,2])
#            XY1[:,0] -= cos1*2.5
#            XY2[:,0] -= sin1*2.5
#            eigval, eigvec = np.linalg.eigh(cov2)
#            eigval = np.sqrt(np.maximum(eigval, 0))
#            np.einsum(utpoints[:,:3], [1,0], eigval, [0], eigvec, [2,0], [1,2],
#                      out=XY2)
#            XY2[:,:2] += meanxy2
#            XY2[:,2] += meana2
#            cos2 = np.cos(XY2[:,2])
#            sin2 = np.sin(XY2[:,2])
#            XY1[:,0] -= cos2*2.5
#            XY2[:,0] -= sin2*2.5
#            # relative positions
#            XY2 -= XY1
#            temp_xy20 = XY2[:,0]*cos1 + XY2[:,1]*sin1
#            XY2[:,1] = XY2[:,1]*cos1 - XY2[:,0]*sin1
#            XY2[:,0] = temp_xy20
#            XY2[:,:2] = abs(XY2[:,:2]) # use symmetry
#            meanxy2 = weights.dot(XY2[:,:2])
#            meana2 = np.arctan2(weights.dot(cos1*sin2-sin1*cos2), 
#                                weights.dot(cos1*cos2+sin1*sin2))
#            if meana2 < 0: meana2 += np.pi
#            XY2[:,:2] -= meanxy2
#            XY2[:,2] -= meana2
#            XY2[:,2] = np.mod(XY2[:,2], np.pi)
#            cov2 = np.einsum(XY2, [0,1], XY2, [0,2], weights, [0], [1,2])

            covr[:2,:2] = cov1[:2,:2] + cov2[:2,:2]
            covr[2,:2] = -cov1[2,:2]
            covr[2,2] = cov1[2,2]
            covr[:2,2] = -cov1[:2,2]
            covr[3,:2] = cov2[2,:2]
            covr[3,3] = cov2[2,2]
            covr[:2,3] = cov2[:2,2]
            eigval, eigvec = np.linalg.eigh(covr)
            eigval = np.sqrt(np.maximum(eigval, 0))
            np.einsum(utpoints, [1,0], eigval, [0], eigvec, [2,0], [1,2],
                      out=xyr)
            xyr += [meanxy2[0]-meanxy1[0], meanxy2[1]-meanxy1[1], meana1, meana2]
            cos1 = np.cos(xyr[:,2])
            sin1 = np.sin(xyr[:,2])
            cos2 = np.cos(xyr[:,3])
            sin2 = np.sin(xyr[:,3])
            xyr[:,0] += (cos1 - cos2)*2.5
            xyr[:,1] += (sin1 - sin2)*1.
            temp_x[:] = np.abs(xyr[:,0]*cos1 + xyr[:,1]*sin1)
            xyr[:,1] = np.abs(xyr[:,1]*cos1 - xyr[:,0]*sin1)
            xyr[:,0] = temp_x
            xyr[:,3] -= xyr[:,2]
            meanxyr = weights.dot(xyr[:,:2])
            meanar = np.arctan2(weights.dot(np.abs(cos1*sin2-sin1*cos2)), 
                                weights.dot(cos1*cos2+sin1*sin2))
            xyr[:,:2] -= meanxyr
            xyr[:,3] -= meanar
            cov2 = np.einsum(xyr[:,[0,1,3]], [0,1], xyr[:,[0,1,3]], [0,2],
                             weights, [0], [1,2])
            
            s,u = np.linalg.eigh(cov2)
            s[s < 0] = 0.
            cov2 = (u * s).dot(u.T)
            std2 = np.sqrt(np.diagonal(cov2)) + 1e-8
            
            features = np.array([(#meanxy2[0], meanxy2[1], meana2,
                             meanxyr[0], meanxyr[1], meanar, 
                             std2[0], std2[1], std2[2], cov2[0,1]/std2[0]/std2[1],
                             cov2[0,2]/std2[0]/std2[2], cov2[1,2]/std2[1]/std2[2])])
            pred = self.model.predict(features)
            totalpred = max(totalpred, pred[0])
            if totalpred > .51: break # at this point you are definitely going to warn
        endtime = time.time() - starttime
        self.bin = np.append(self.bin, features, axis=0)
        return (min(max(totalpred,0.),1.) , endtime)
        
    
    
    def alarmOld(self, vehicle1, vehicle2, MM1, MM2, times):
        """input and output match that of the functions in alarms.py"""
        MM1weights = MM1.weights
        MM2weights = MM2.weights
        
        starttime = time.time()
        v1 = vehicle1.utpoints
        v2 = vehicle2.utpoints
        features = np.empty((len(times),8))

        for j,dtime in enumerate(times):
            v1 = MM1.UTstep(v1, dtime)
            v2 = MM2.UTstep(v2, dtime)
            xy1 = MM1.fullToXY(v1)
            xy2 = MM2.fullToXY(v2)
            x1mean = MM1weights.dot(xy1[:,0])
            y1mean = MM1weights.dot(xy1[:,1])
            cosmean = MM1weights.dot(np.cos(xy1[:,2]))
            sinmean = MM1weights.dot(np.sin(xy1[:,2]))
            th1mean = np.arctan2(sinmean, cosmean)
            x1std = MM1weights.dot((xy1[:,0] - x1mean)**2.)
            y1std = MM1weights.dot((xy1[:,1] - y1mean)**2.)
            th1std = MM1weights.dot(rectify(xy1[:,2] - th1mean)**2.)**.5
            x2mean = MM2weights.dot(xy2[:,0])
            y2mean = MM2weights.dot(xy2[:,1])
            cosmean = MM2weights.dot(np.cos(xy2[:,2]))
            sinmean = MM2weights.dot(np.sin(xy2[:,2]))
            th2mean = np.arctan2(sinmean, cosmean)
            x2std = MM2weights.dot((xy2[:,0] - x2mean)**2.)
            y2std = MM2weights.dot((xy2[:,1] - y2mean)**2.)
            th2std = MM2weights.dot(rectify(xy2[:,2] - th2mean)**2.)**.5
            features[j,:] = [ (x2mean - x1mean) / xy_bound,
                              (y2mean - y1mean) / xy_bound,
                              th1mean / np.pi,
                              th2mean / np.pi,
                              (x1std+x2std)**.5 / std_xy_bound,
                              (y1std+y2std)**.5 / std_xy_bound,
                              th1std / std_angle_bound,
                              th2std / std_angle_bound ]
        pred = self.model.predict(features)
        pred = np.max(pred)
        endtime = time.time() - starttime
        self.bin = np.append(self.bin, features, axis=0)
        return (min(max(pred,0.),1.) , endtime)
      
        
""" uncomment createTruth or trainModels() to easily run from the command line
    comment it again before running the main simulator! """
if __name__=='__main__':
    pass
    #createTruth2()
    #trainModels()