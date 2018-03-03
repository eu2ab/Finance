from numpy import *

## Monte Carlo estimator for European Option
S0 = 100
k = 105
T = 1.0
r = 0.05
sigma = 0.2

I = 100000  # size of sample

z = random.standard_normal(I)
ST = S0 * exp((r - 0.5 * sigma ** 2) * T + sigma * sqrt(T) * z)
hT = maximum(ST - k, 0)
C0 = exp(-r * T) * sum(hT) / I
C0