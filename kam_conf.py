import numpy as xp
from numpy import linalg as LA
from numpy.fft import rfftn, irfftn, fftn, ifftn, fftfreq, fftshift, ifftshift
import convergence as cv
import gc
import time
import warnings
warnings.filterwarnings("ignore")

def main():
	# dict_params = {
	# 	'n_min': 2 ** 4,
	# 	'n_max': 2 ** 10,
	# 	'omega0': [0.618033988749895, -1.0],
	# 	'Omega': [1.0, 0.0],
	# 	'potential': 'pot1_2d',
	# 	'eps_region': [[0.0, 0.35], [0, 0.12]],
	# 	'eps_line': [0.0, 0.028],
	# 	'eps_modes': [1, 1],
	# 	'eps_dir' : [1, 1]}
	# dict_params = {
	# 	'n_min': 2 ** 4,
	#   'n_max': 2 ** 12,
	# 	'omega0': [0.414213562373095, -1.0],
	# 	'Omega': [1.0, 0.0],
	# 	'potential': 'pot1_2d',
	# 	'eps_region': [[0, 0.12], [0, 0.225]]}
	# dict_params = {
	# 	'n_min': 2 ** 4,
	#   'n_max': 2 ** 12,
	# 	'omega0': [0.302775637731995, -1.0],
	# 	'Omega': [1.0, 0.0],
	# 	'potential': 'pot1_2d',
	# 	'eps_region': [[0, 0.06], [0, 0.2]]}
	dict_params = {
		'n_min': 2 ** 6,
		'n_max': 2 ** 9,
		'omega0': [1.324717957244746, 1.754877666246693, 1.0],
		'Omega': [1.0, 1.0, -1.0],
		'potential': 'pot1_3d',
		'eps_region': [[0.0, 0.15], [0.0,  0.40], [0.1, 0.1]],
		'eps_line': [0, 0.04],
		'eps_modes': [1, 1, 0],
		'eps_dir': [1, 5, 0.1]}
	dict_params.update({
	    'tolmin': 1e-6,
	    'threshold': 1e-9,
		'tolmax': 1e8,
		'maxiter': 30,
		'precision': 64,
		'eps_n': 256,
		'deps': 1e-4,
		'eps_indx': [0, 1],
		'eps_type': 'cartesian',
		'dist_surf': 1e-5,
		'choice_initial': 'fixed',
		'monitor_grad': False,
		'r': 6,
		'parallelization': False,
		'adapt_n': True,
		'adapt_eps': False,
		'save_results': True,
		'plot_results': False})
	dv = {
		'pot1_2d': lambda phi, eps, Omega: - Omega[0] * eps[0] * xp.sin(phi[0]) - eps[1] * (Omega[0] + Omega[1]) * xp.sin(phi[0] + phi[1]),
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
		self.precision = {32: xp.float32, 64: xp.float64, 128: xp.float128}.get(self.precision, xp.float64)
		self.dv = dv
		self.dim = len(self.omega0)
		self.id = xp.reshape(xp.identity(self.dim), 2 * (self.dim, ) + self.dim * (1,))
		self.omega0 = xp.array(self.omega0, dtype=self.precision)
		self.Omega = xp.array(self.Omega, dtype=self.precision)
		self.zero_ = self.dim * (0,)

	def set_var(self, n):
		ind_nu = self.dim * (fftfreq(n, d=1.0/self.precision(n)),)
		ind_phi = self.dim * (xp.linspace(0.0, 2.0 * xp.pi, n, endpoint=False, dtype=self.precision),)
		nu = xp.meshgrid(*ind_nu, indexing='ij')
		self.phi = xp.meshgrid(*ind_phi, indexing='ij')
		self.norm_nu = LA.norm(nu, axis=0)
		self.omega0_nu = xp.einsum('i,i...->...', self.omega0, nu)
		self.Omega_nu = xp.einsum('i,i...->...', self.Omega, nu)
		self.lk = - self.omega0_nu ** 2
		self.sml_div = -1j * xp.divide(1.0, self.omega0_nu, where=self.omega0_nu!=0)
		self.rescale_fft = self.precision(n ** self.dim)
		self.ilk = xp.divide(1.0, self.lk, where=self.lk!=0)
		self.initial_h = lambda eps: self.set_initial_h(eps, order=1)
		self.tail_indx = self.dim * xp.index_exp[n//4:3*n//4+1]
		self.pad = self.dim * ((n//4, n//4),)

	def set_initial_h(self, epsilon, order=1):
		if order == 1:
			return [- ifftn(fftn(self.dv(self.phi, epsilon, self.Omega)) * self.ilk).real, 0.0]
		elif order == 2:
			h0 = - ifftn(fftn(self.dv(self.phi, epsilon, self.Omega)) * self.ilk).real
			h2 = - ifftn(fftn(self.dv(self.phi + xp.tensordot(self.Omega, h0, axes=0), epsilon, self.Omega) - self.dv(self.phi, epsilon, self.Omega)) * self.ilk).real
			lam2 = - xp.mean(self.dv(self.phi + xp.tensordot(self.Omega, h0, axes=0), epsilon, self.Omega) - self.dv(self.phi, epsilon, self.Omega))
			return [h0 + h2, lam2]

	def refine_h(self, h, lam, eps):
		n = h.shape[0]
		self.set_var(n)
		fft_h = fftn(h)
		fft_h[xp.abs(fft_h) <= self.threshold * xp.abs(fft_h).max()] = 0.0
		fft_h[self.zero_] = 0.0
		h_thresh = ifftn(fft_h).real
		arg_v = self.phi + xp.tensordot(self.Omega, h_thresh, axes=0) % (2.0 * xp.pi)
		fft_l = 1j * self.Omega_nu * fft_h
		fft_l[self.zero_] = self.rescale_fft
		lfunc = ifftn(fft_l).real
		epsilon = ifftn(self.lk * fft_h).real + self.dv(arg_v, eps, self.Omega) + lam
		fft_leps = fftn(lfunc * epsilon)
		delta = - fft_leps[self.zero_].real / fft_l[self.zero_].real
		w = ifftn((delta * fft_l + fft_leps) * self.sml_div).real
		del fft_l, fft_leps, epsilon
		gc.collect()
		fft_wll = fftn(w / (lfunc ** 2))
		fft_ill = fftn(1.0 / (lfunc ** 2))
		w0 = - fft_wll[self.zero_].real / fft_ill[self.zero_].real
		beta = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj()).real
		h_ = h_thresh + beta * lfunc - xp.mean(beta * lfunc) * lfunc
		del beta
		gc.collect()
		lam_ = lam + delta
		fft_h_ = fftn(h_)
		tail_norm = xp.abs(fft_h_[self.tail_indx]).max()
		fft_h_[self.zero_] = 0.0
		fft_h_[xp.abs(fft_h_) <= self.threshold * xp.abs(fft_h_).max()] = 0.0
		if (tail_norm >= self.threshold * xp.abs(fft_h_).max()) and (n < self.n_max) and self.adapt_n:
			print('warning: change of value of n (from {} to {})'.format(n, 2 * n))
			self.set_var(2 * n)
			fft_h_ = ifftshift(xp.pad(fftshift(fft_h_), self.pad))
			h = ifftn(ifftshift(xp.pad(fftshift(fft_h), self.pad))).real * (2 ** self.dim)
			h_ = ifftn(fft_h_).real * (2 ** self.dim)
		else:
			h_ = ifftn(fft_h_).real
		arg_v = self.phi + xp.tensordot(self.Omega, h_, axes=0) % (2.0 * xp.pi)
		err = xp.abs(self.lk * fft_h_ + fftn(self.dv(arg_v, eps, self.Omega) + lam_)).sum() / self.rescale_fft
		if self.monitor_grad:
			dh_ = self.id + xp.tensordot(self.Omega, xp.gradient(h_, 2.0 * xp.pi / n), axes=0)
			det_h_ = xp.abs(LA.det(xp.moveaxis(dh_, [0, 1], [-2, -1]))).min()
			if det_h_ <= self.tolmin:
				print('warning: non-invertibility...')
		return h_, lam_, err

	def norms(self, h, r=0):
		self.set_var(h.shape[0])
		fft_h = fftn(h)
		return xp.sqrt(((1.0 + self.norm_nu ** 2) ** r * (xp.abs(fft_h) / self.rescale_fft) ** 2).sum()), xp.sqrt(xp.abs(ifftn(self.omega0_nu ** r * fft_h) ** 2).sum()), xp.sqrt(xp.abs(ifftn(self.Omega_nu ** r * fft_h) ** 2).sum())

if __name__ == "__main__":
	main()
