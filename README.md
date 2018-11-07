Code related to "Optimal Alarms for Vehicular Collision Detection", published in IEEE IV Symposium 2017.
The code that actually recreates the results of the paper is available at https://github.com/utexas-ghosh-group/carstop/tree/master/VCD. This code has several modifications:

+ The general polygon-intersection method has been replaced with a faster rectangle-specific method to test vehicle collisions.

+ The package numba was used to achieve significant speedup for low-sample Monte Carlo sampling.

+ Some alternative variations of the unscented transform and regression are included - but don't outperform the methods discussed in the paper.

The simulate_X modules run each simulation.
The regressor_X modules generate training data for each simulation's regressor.
All others are helpers, more documentation is in the files themselves.
