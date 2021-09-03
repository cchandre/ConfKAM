########################################################################################################################
##                           Dictionary of parameters: https://github.com/cchandre/ConfKAM                            ##
########################################################################################################################

import numpy as xp

#Method = 'region'
Method = 'line_norm'
Nxy = 100
r = 6

omega0 = [(xp.sqrt(5) - 1) / 2, -1]
Omega = [1, 0]
Dv = lambda phi, eps, omega: - omega[0] * eps[0] * xp.sin(phi[0]) - eps[1] * (omega[0] + omega[1]) * xp.sin(phi[0] + phi[1])
CoordRegion = [[0.0, 0.35], [0, 0.12]]
IndxLine = (0, 1)
PolarAngles = [0.0, xp.pi / 2.0]
CoordLine = [0.0, 0.028]
ModesLine = (1, 1)
DirLine = [1, 1]

# sigma = 1.324717957244746
# omega0 = [sigma, sigma ** 2, 1]
# Omega = [1, 1, -1]
# Dv = lambda phi, eps, omega: - omega[0] * eps[0] * xp.sin(phi[0]) - omega[1] * eps[1] * xp.sin(phi[1]) - omega[2] * eps[2] * xp.sin(phi[2])
# CoordRegion = [[0.0, 0.15], [0.0,  0.40], [0.1, 0.1]]
# IndxLine = (0, 1)
# PolarAngles = [0.0, xp.pi / 2.0]
# CoordLine = [0.02, 0.05]
# ModesLine = (1, 1, 0)
# DirLine = [1, 5, 0.1]

AdaptSize = True
Lmin = 2 ** 6
Lmax = 2 ** 12

TolMax = 1e+10
TolMin = 1e-10
Threshold = 1e-12
MaxIter = 100

Type = 'cartesian'
ChoiceInitial = 'continuation'
MethodInitial = 'one_step'

AdaptEps = True
MinEps = 1e-7
MonitorGrad = False

Precision = 64
SaveData = False
PlotResults = False
Parallelization = (True, 4)

########################################################################################################################
##                                                DO NOT EDIT BELOW                                                   ##
########################################################################################################################
Precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(Precision, xp.float64)
dict = {'Method': 'compute_' + Method}
dict.update({
        'Nxy': Nxy,
        'r': r,
		'omega0': xp.asarray(omega0, dtype=Precision),
		'Omega': xp.asarray(Omega, dtype=Precision),
		'Dv': Dv,
		'CoordRegion': xp.asarray(CoordRegion, dtype=Precision),
        'IndxLine': IndxLine,
        'PolarAngles': xp.asarray(PolarAngles, dtype=Precision),
		'CoordLine': CoordLine,
		'ModesLine': xp.asarray(ModesLine),
		'DirLine': xp.asarray(DirLine),
		'AdaptSize': AdaptSize,
		'Lmin': Lmin,
		'Lmax': Lmax,
		'TolMax': TolMax,
		'TolMin': TolMin,
		'Threshold': Threshold,
		'MaxIter': MaxIter,
		'Type': Type,
		'ChoiceInitial': ChoiceInitial,
        'MethodInitial': MethodInitial,
		'AdaptEps': AdaptEps,
		'MinEps': MinEps,
		'MonitorGrad': MonitorGrad,
		'Precision': Precision,
		'SaveData': SaveData,
		'PlotResults': PlotResults,
		'Parallelization': Parallelization})
########################################################################################################################
