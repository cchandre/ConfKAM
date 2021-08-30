########################################################################################################################
##                                   Definition of the parameters for ConfKAM                                         ##
########################################################################################################################
##                                                                                                                    ##
##   Method: 'line_norm', 'region'; choice of method                                                                  ##
##   Nxy: integer; number of points along each line in computations                                                   ##
##                                                                                                                    ##
##   omega0: array of n floats; frequency vector of the invariant torus                                               ##
##   Omega: array of n floats; vector defining the perturbation in actions                                            ##
##   dv: function; derivative of the n-d potential along a line                                                       ##
##   eps_region: array of floats; min and max values of the amplitudes for each mode of the potential (see dv)        ##
##   eps_indx: array of 2 integers; indices of the modes to be varied in region()                                     ##
##             in 'polar', radii are on eps_indx[0] and angles on eps_indx[1]                                         ##
##             parallelization in region() is done along the eps_indx[0] axis                                         ##
##   eps_line: 1d array of floats; min and max values of the amplitudes of the potential used in line_norm()          ##
##   eps_modes: array of 0 and 1; specify which modes are being varied in line_norm() (1 for a varied mode)           ##
##   eps_dir: 1d array of floats; direction of the one-parameter family used in line_norm()                           ##
##                                                                                                                    ##
##   r: integer; order of the Sobolev norm used in line_norm()                                                        ##
##                                                                                                                    ##
##   AdaptL: boolean; if True, changes the dimension of arrays depending on the tail of the FFT of h(psi)             ##
##   Lmin: integer; minimum and default value of the dimension of arrays for h(psi)                                   ##
##   Lmax: integer; maximum value of the dimension of arrays for h(psi) if AdaptL is True                             ##
##                                                                                                                    ##
##   TolMax: float; value of norm for divergence                                                                      ##
##   TolMin: float; value of norm for convergence                                                                     ##
##   Threshold: float; threshold value for truncating Fourier series of h(psi)                                        ##
##   MaxIter: integer; maximum number of iterations for the Newton method                                             ##
##                                                                                                                    ##
##   Type: 'cartesian', 'polar'; type of computation for 2d plots                                                     ##
##   ChoiceInitial: 'fixed', 'continuation'; method for the initial conditions of the Newton method                   ##
##   MethodInitial: 'zero', 'one_step'; method to generate the initial conditions for the Newton iteration            ##
##                                                                                                                    ##
##   AdaptEps: boolean; if True adapt the increment of eps in line_norm()                                             ##
##   MinEps: float; minimum value of the increment of eps if AdaptEps=True                                            ##
##   MonitorGrad: boolean; if True, monitors the gradient of h(psi)                                                   ##
##                                                                                                                    ##
##   Precision: 32, 64 or 128; precision of calculations (default=64)                                                 ##
##   SaveData: boolean; if True, the results are saved in a .mat file                                                 ##
##   PlotResults: boolean; if True, the results are plotted right after the computation                               ##
##   Parallelization: 2d array [boolean, int]; True for parallelization, int is the number of cores to be used        ##
##                                                                                                                    ##
########################################################################################################################
import numpy as xp

#Method = 'region'
Method = 'line_norm'
Nxy = 500
r = 4

omega0 = [(xp.sqrt(5) - 1) / 2, -1]
Omega = [1, 0]
dv = lambda phi, eps, omega: - omega[0] * eps[0] * xp.sin(phi[0]) - eps[1] * (omega[0] + omega[1]) * xp.sin(phi[0] + phi[1])
eps_region = [[0.0, 0.35], [0, 0.12]]
eps_indx = [0, 1]
eps_line = [0.0, 0.028]
eps_modes = [1, 1]
eps_dir = [1, 1]

# sigma = 1.324717957244746
# omega0 = [sigma, sigma ** 2, 1]
# Omega = [1, 1, -1]
# dv = lambda phi, eps, omega: - omega[0] * eps[0] * xp.sin(phi[0]) - omega[1] * eps[1] * xp.sin(phi[1]) - omega[2] * eps[2] * xp.sin(phi[2])
# eps_region = [[0.0, 0.15], [0.0,  0.40], [0.1, 0.1]]
# eps_indx = [0, 1]
# eps_line = [0.0, 0.05]
# eps_modes = [1, 1, 0]
# eps_dir = [1, 5, 0.1]

AdaptL = False
Lmin = 2 ** 7
Lmax = 2 ** 10

TolMax = 1e+30
TolMin = 1e-9
Threshold = 1e-13
MaxIter = 100

Type = 'cartesian'
ChoiceInitial = 'continuation'
MethodInitial = 'one_step'

AdaptEps = True
MinEps = 1e-7
MonitorGrad = False

Precision = 64
SaveData = False
PlotResults = True
Parallelization = [True, 4]

########################################################################################################################
##                                                DO NOT EDIT BELOW                                                   ##
########################################################################################################################
Precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(Precision, xp.float64)
dict = {'Method': Method}
dict.update({
        'Nxy': Nxy,
		'omega0': xp.asarray(omega0, dtype=Precision),
		'Omega': xp.asarray(Omega, dtype=Precision),
		'dv': dv,
		'eps_region': eps_region,
		'eps_line': eps_line,
		'eps_modes': eps_modes,
		'eps_dir': eps_dir,
		'AdaptL': AdaptL,
		'Lmin': Lmin,
		'Lmax': Lmax,
		'eps_indx': eps_indx,
		'r': r,
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
