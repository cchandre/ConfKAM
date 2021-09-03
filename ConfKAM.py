import numpy as xp
from numpy import linalg as LA
from numpy.fft import fftn, ifftn, fftfreq, fftshift, ifftshift
from scipy.optimize import root
import matplotlib.pyplot as plt
from ConfKAM_modules import compute_line_norm, compute_region
from ConfKAM_dict import dict
import gc
import warnings
warnings.filterwarnings("ignore")

def main():
	case = ConfKAM(dict)
	eval(case.Method + '(case)')
	plt.show()

class ConfKAM:
	def __repr__(self):
		return '{self.__class__.__name__}({self.DictParams})'.format(self=self)

	def __str__(self):
		return 'KAM in configuration space ({self.__class__.__name__}) with omega0 = {self.omega0} and Omega = {self.Omega}'.format(self=self)

	def __init__(self, dict):
		for key in dict:
			setattr(self, key, dict[key])
		self.DictParams = dict
		self.dim = len(self.omega0)
		self.zero_ = self.dim * (0,)
		self.id = xp.reshape(xp.identity(self.dim), 2 * (self.dim,) + self.dim * (1,))

	def set_var(self, L):
		ind_nu = self.dim * (fftfreq(L, d=1.0/self.Precision(L)),)
		ind_phi = self.dim * (xp.linspace(0.0, 2.0 * xp.pi, L, endpoint=False, dtype=self.Precision),)
		nu = xp.meshgrid(*ind_nu, indexing='ij')
		self.phi = xp.meshgrid(*ind_phi, indexing='ij')
		self.norm_nu = LA.norm(nu, axis=0)
		self.omega0_nu = xp.einsum('i,i...->...', self.omega0, nu)
		self.Omega_nu = xp.einsum('i,i...->...', self.Omega, nu)
		self.sml_div = - 1j * xp.divide(1.0, self.omega0_nu, where=self.omega0_nu!=0.0)
		self.sml_div[self.zero_] = 0.0
		self.lk = - self.omega0_nu ** 2
		self.ilk = xp.divide(1.0, self.lk, where=self.lk!=0.0)
		self.ilk[self.zero_] = 0.0
		self.rescale_fft = self.Precision(n ** self.dim)
		self.tail_indx = self.dim * xp.index_exp[L//4:3*L//4+1]
		self.pad = self.dim * ((L//4, L//4),)

	def initial_h(self, eps, L, method='one_step'):
		self.set_var(L)
		if method == 'zero':
			return [xp.zeros_like(self.lk), 0.0]
		elif method == 'one_step':
			return [- ifftn(fftn(self.Dv(self.phi, eps, self.Omega)) * self.ilk).real, 0.0]
		else:
			h = - ifftn(fftn(self.Dv(self.phi, eps, self.Omega)) * self.ilk).real
			sol = root(self.conjug_eq, h.flatten(), args=(eps, L), method=method, options={'fatol': 1e-9})
			if sol.success:
				return [sol.x.reshape((L, L)), 0.0]
			else:
				return [h, 0.0]

	def conjug_eq(self, h, eps, L):
		arg_v = (self.phi + xp.tensordot(self.Omega, h.reshape(self.dim * (L,)), axes=0)) % (2.0 * xp.pi)
		return (ifftn(self.lk * fftn(h.reshape(self.dim * (L,)))).real + self.Dv(arg_v, eps, self.Omega)).flatten()

	def refine_h(self, h, lam, eps):
		L = h.shape[0]
		self.set_var(L)
		fft_h = fftn(h)
		fft_h[xp.abs(fft_h) <= self.Threshold * self.rescale_fft] = 0.0
		fft_h[self.zero_] = 0.0
		h_thresh = ifftn(fft_h).real
		arg_v = (self.phi + xp.tensordot(self.Omega, h_thresh, axes=0)) % (2.0 * xp.pi)
		fft_l = 1j * self.Omega_nu * fft_h
		fft_l[self.zero_] = self.rescale_fft
		l = ifftn(fft_l).real
		epsilon = ifftn(self.lk * fft_h).real + self.Dv(arg_v, eps, self.Omega) + lam
		fft_leps = fftn(l * epsilon)
		delta = - fft_leps[self.zero_].real / fft_l[self.zero_].real
		w = ifftn((delta * fft_l + fft_leps) * self.sml_div).real
		del fft_l, fft_leps, epsilon
		gc.collect()
		fft_wll = fftn(w / (l ** 2))
		fft_ill = fftn(1.0 / (l ** 2))
		w0 = - fft_wll[self.zero_].real / fft_ill[self.zero_].real
		beta = ifftn((fft_wll + w0 * fft_ill) * self.sml_div.conj()).real
		h_ = h_thresh + beta * l - xp.mean(beta * l) * l / xp.mean(l)
		del beta
		gc.collect()
		lam_ = lam + delta
		fft_h_ = fftn(h_)
		tail_norm = xp.abs(fft_h_[self.tail_indx]).max()
		fft_h_[self.zero_] = 0.0
		fft_h_[xp.abs(fft_h_) <= self.Threshold * self.rescale_fft] = 0.0
		if self.AdaptL and (tail_norm >= self.TolMin * xp.abs(fft_h_).max()) and (xp.abs(fft_h_).max() <= self.TolMax) and (L < self.Lmax):
			L *= 2
			self.set_var(L)
			h = ifftn(ifftshift(xp.pad(fftshift(fft_h), self.pad))).real * (2 ** self.dim)
			fft_h_ = ifftshift(xp.pad(fftshift(fft_h_), self.pad)) * (2 ** self.dim)
		h_ = ifftn(fft_h_).real
		arg_v = (self.phi + xp.tensordot(self.Omega, h_, axes=0)) % (2.0 * xp.pi)
		err = xp.abs(ifftn(self.lk * fft_h_).real + self.Dv(arg_v, eps, self.Omega) + lam_).max()
		if self.MonitorGrad:
			dh_ = self.id + xp.tensordot(self.Omega, xp.gradient(h_, 2.0 * xp.pi / L), axes=0)
			det_h_ = xp.abs(LA.det(xp.moveaxis(dh_, [0, 1], [-2, -1]))).min()
			if det_h_ <= self.TolMin:
				print('\033[31m        warning: non-invertibility...\033[00m')
		return h_, lam_, err

	def norms(self, h, r=0):
		self.set_var(h.shape[0])
		fft_h = fftn(h)
		return xp.sqrt((self.norm_nu ** (2 * r) * (xp.abs(fft_h) / self.rescale_fft) ** 2).sum()), xp.sqrt((xp.abs(ifftn(self.omega0_nu ** r * fft_h)) ** 2).sum()), xp.sqrt((xp.abs(ifftn(self.Omega_nu ** r * fft_h)) ** 2).sum())

if __name__ == "__main__":
	main()
