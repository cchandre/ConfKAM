import numpy as xp
from numpy import linalg as LA
from numpy.fft import fftn, ifftn, fftfreq, fftshift, ifftshift
import convergence as cv
import time
import warnings
warnings.filterwarnings("ignore")

def main():
	dict_params = {
		'n': 2 ** 12,
		'omega0': [0.618033988749895, -1.0],
		'eps_region': [[0.02, 0.35], [0, 0.12]],
		'eps_modes': [1, 1],
		'eps_dir' : [1, 1],
		'Omega': [1.0, 0.0],
		'potential': 'pot1_2d'}
	# dict_params = {
	# 	'n': 2 ** 10,
	# 	'omega0': [0.414213562373095, -1.0],
	# 	'eps_region': [[0, 0.12], [0, 0.225]],
	# 	'Omega': [1.0, 0.0],
	# 	'potential': 'pot1_2d'}
	# dict_params = {
	# 	'n': 2 ** 10,
	# 	'omega0': [0.302775637731995, -1.0],
	# 	'eps_region': [[0, 0.06], [0, 0.2]],
	# 	'Omega': [1.0, 0.0],
	# 	'potential': 'pot1_2d'}
	# dict_params = {
	# 	'n': 2 ** 8,
	# 	'omega0': [1.324717957244746, 1.754877666246693, 1.0],
	# 	'eps_region': [[0.0, 0.15], [0.0,  0.40], [0.1, 0.1]],
	# 	'eps_modes': [1, 1, 0],
	# 	'eps_dir': [1, 5, 0.1],
	# 	'Omega': [1.0, 1.0, -1.0],
	# 	'potential': 'pot1_3d'}
	dict_params.update({
	    'tolmin': 1e-6,
	    'threshold': 1e-11,
		'tolmax': 1e30,
		'maxiter': 50,
		'precision': 64,
		'eps_n': 256,
		'deps': 1e-4,
		'eps_indx': [0, 1],
		'eps_type': 'cartesian',
		'dist_surf': 1e-6,
		'choice_initial': 'continuation',
		'r': 4,
		'save_results': False,
		'plot_results': True})
	dv = {
		'pot1_2d': lambda phi, eps, Omega: Omega[0] * eps[0] * xp.sin(phi[0]) + eps[1] * (Omega[0] + Omega[1]) * xp.sin(phi[0] + phi[1]),
		'pot1_3d': lambda phi, eps, Omega: - Omega[0] * eps[0] * xp.sin(phi[0]) - Omega[1] * eps[1] * xp.sin(phi[1]) - Omega[2] * eps[2] * xp.sin(phi[2])
		}.get(dict_params['potential'], 'pot1_2d')
	case = ConfKAM(dv, dict_params)
	# data = cv.region(case)
	data = cv.line_norm(case)


class ConfKAM:
	def __repr__(self):
		return '{self.__class__.name__}({self.dv, self.DictParams})'.format(self=self)

	def __str__(self):
		return 'KAM in configuration space ({self.__class__.name__}) with omega0 = {self.omega0} and Omega = {self.Omega}'.format(self=self)

	def __init__(self, dv, dict_params):
		for key in dict_params:
			setattr(self, key, dict_params[key])
		self.DictParams = dict_params
		self.precision = {64: xp.float64, 128: xp.float128}.get(self.precision, xp.float64)
		self.dv = dv
		self.dim = len(self.omega0)
		self.omega0 = xp.array(self.omega0, dtype=self.precision)
		self.zero_ = self.dim * (0,)
		ind_nu = self.dim * (fftfreq(self.n, d=1.0/self.precision(self.n)),)
		nu = xp.meshgrid(*ind_nu, indexing='ij')
		self.norm_nu = LA.norm(nu, axis=0)
		self.omega0_nu = xp.einsum('i,i...->...', self.omega0, nu)
		self.Omega = xp.array(self.Omega, dtype=self.precision)
		self.Omega_nu = xp.einsum('i,i...->...', self.Omega, nu)
		self.lk = - self.omega0_nu ** 2
		self.sml_div = 1j * self.omega0_nu
		self.sml_div = xp.divide(1.0, self.sml_div, where=self.sml_div!=0)
		ind_phi = self.dim * (xp.linspace(0.0, 2.0 * xp.pi, self.n, endpoint=False, dtype=self.precision),)
		self.phi = xp.meshgrid(*ind_phi, indexing='ij')
		self.rescale_fft = self.precision(self.n ** self.dim)
		self.threshold_r = self.threshold * self.rescale_fft
		ilk = xp.divide(1.0, self.lk, where=self.lk!=0)
		self.initial_h = lambda eps: [- ifftn(fftn(self.dv(self.phi, eps, self.Omega)) * ilk), 0.0]

	def refine_h(self, h, lam, eps):
		fft_h = fftn(h)
		fft_h[xp.abs(fft_h) <= self.threshold * xp.abs(fft_h).max()] = 0.0
		h_thresh = ifftn(fft_h)
		arg_v = self.phi + xp.tensordot(self.Omega, h_thresh, axes=0)
		fft_l = 1j * self.Omega_nu * fft_h
		fft_l[self.zero_] = self.rescale_fft
		lfunc = ifftn(fft_l)
		epsilon = ifftn(self.lk * fft_h) + self.dv(arg_v, eps, self.Omega) + lam
		fft_leps = fftn(lfunc * epsilon)
		delta = - fft_leps[self.zero_] / fft_l[self.zero_]
		w = ifftn((delta * fft_l + fft_leps) * self.sml_div)
		fft_wll = fftn(w / lfunc ** 2)
		fft_ill = fftn(1.0 / lfunc ** 2)
		w0 = - fft_wll[self.zero_] / fft_ill[self.zero_]
		beta = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj())
		h_ = xp.real(h_thresh + beta * lfunc - xp.mean(beta * lfunc) * lfunc)
		lam_ = xp.real(lam + delta)
		fft_h = fftn(h_)
		fft_h[xp.abs(fft_h) <= self.threshold * xp.abs(fft_h).max()] = 0.0
		h_ = ifftn(fft_h)
		arg_v = self.phi + xp.tensordot(self.Omega, h_, axes=0)
		err = xp.abs(self.lk * fft_h + fftn(self.dv(arg_v, eps, self.Omega) + lam)).sum() / self.rescale_fft
		tail = h_[self.dim * (self.n//4 : 3*self.n//4)]
		return h_, lam_, err

	def norms(self, h, r=0):
		fft_h = fftn(h)
		return xp.sqrt(((1.0 + self.norm_nu ** 2) ** r * (xp.abs(fft_h) / self.rescale_fft) ** 2).sum()), xp.sqrt(xp.abs(ifftn(self.omega0_nu ** r * fft_h) ** 2).sum()), xp.sqrt(xp.abs(ifftn(self.Omega_nu ** r * fft_h) ** 2).sum())

if __name__ == "__main__":
	main()
