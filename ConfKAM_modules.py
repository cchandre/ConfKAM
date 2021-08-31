import numpy as xp
from tqdm import tqdm, trange
import multiprocess
from scipy.io import savemat
import time
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.sans-serif': ['Palatino'],
    'font.size': 24,
    'axes.labelsize': 30,
    'figure.figsize': [8, 8],
    'image.cmap': 'bwr'})

def point(eps, case, h, lam, gethull=False, display=False):
    h_, lam_, err = h.copy(), lam, 1.0
    it_count = 0
    while (case.TolMax >= err >= case.TolMin) and (it_count <= case.MaxIter):
        h_, lam_, err = case.refine_h(h_, lam_, eps)
        it_count += 1
        if display:
            print('\033[90m        iteration={}   err={} \033[00m'.format(it_count, err))
    if err <= case.TolMin:
        it_count = - it_count
    if gethull:
        timestr = time.strftime("%Y%m%d_%H%M")
        save_data('hull', h_, timestr, case)
    return [int(err <= case.TolMin), it_count], h_, lam_

def line_norm(case, display=True):
    print('\033[92m    {} -- line_norm \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    eps_modes = xp.array(case.eps_modes)
    eps_dir = xp.array(case.eps_dir)
    epsilon0 = case.eps_line[0]
    epsvec = epsilon0 * eps_modes * eps_dir + (1 - eps_modes) * eps_dir
    h, lam = case.initial_h(epsvec, case.Lmin, case.MethodInitial)
    deps = (case.eps_line[1] - case.eps_line[0]) / case.Precision(case.Nxy - 1)
    resultnorm = []
    count_fail = 0
    while epsilon0 <= case.eps_line[1] and (count_fail <= case.MaxIter):
        epsilon = epsilon0 + deps
        epsvec = epsilon * eps_modes * eps_dir + (1 - eps_modes) * eps_dir
        if case.ChoiceInitial == 'fixed':
            h, lam = case.initial_h(epsvec, h.shape[0], case.MethodInitial)
        result, h_, lam_ = point(epsvec, case, h, lam, display=False)
        if result[0] == 1:
            count_fail = 0
            resultnorm.append(xp.concatenate((epsilon, case.norms(h_, case.r)), axis=None))
            if display:
                print('\033[90m        epsilon={:.6f}    norm_{:d}={:.3e} \033[00m'.format(epsilon, case.r, case.norms(h_, case.r)[0]))
            save_data('line_norm', xp.array(resultnorm), timestr, case)
        elif case.AdaptEps:
            while (result[0] == 0) and deps >= case.MinEps:
                deps /= 5.0
                epsilon = epsilon0 + deps
                epsvec = epsilon * eps_modes * eps_dir + (1 - eps_modes) * eps_dir
                result, h_, lam_ = point(epsvec, case, h, lam, display=False)
            if result[0] == 1:
                count_fail = 0
                resultnorm.append(xp.concatenate((epsilon, case.norms(h_, case.r)), axis=None))
                if display:
                    print('\033[90m        epsilon={:.6f}    norm_{:d}={:.3e} \033[00m'.format(epsilon, case.r, case.norms(h_, case.r)[0]))
                save_data('line_norm', xp.array(resultnorm), timestr, case)
        if result[0] == 0:
            count_fail += 1
        elif (case.ChoiceInitial == 'continuation'):
            h, lam = h_.copy(), lam_
        epsilon0 = epsilon
    resultnorm = xp.array(resultnorm)
    if case.PlotResults and resultnorm.size != 0:
        fig, ax = plt.subplots(1, 1)
        ax.semilogy(resultnorm[:, 0], resultnorm[:, 1], linewidth=2)
        ax.set_xlabel('$\epsilon$')
        ax.set_ylabel('$\Vert h \Vert_{}$'.format(case.r))
    return resultnorm

def line(epsilon, case, display=False):
    h, lam = case.initial_h(epsilon[0], case.Lmin, case.MethodInitial)
    results = []
    for eps in tqdm(epsilon, disable=not display):
        result, h_, lam_ = point(eps, case, h, lam)
        if (result[0] == 1) and case.ChoiceInitial == 'continuation':
            h, lam = h_.copy(), lam_
        elif case.ChoiceInitial == 'fixed':
            h, lam = case.initial_h(eps, h_.shape[0], case.MethodInitial)
        results.append(result)
    return xp.array(results)[:, 0], xp.array(results)[:, 1]

def region(case):
    print('\033[92m    {} -- region \033[00m'.format(case.__str__()))
    timestr = time.strftime("%Y%m%d_%H%M")
    eps_region = xp.array(case.eps_region, dtype=case.Precision)
    eps_vecs = xp.linspace(eps_region[:, 0], eps_region[:, 1], case.Nxy, dtype=case.Precision)
    if case.Type == 'cartesian':
        eps_list = []
        for it in range(case.Nxy):
            eps_copy = eps_vecs.copy()
            eps_copy[:, case.eps_indx[1]] = eps_vecs[it, case.eps_indx[1]]
            eps_list.append(eps_copy)
    elif case.Type == 'polar':
        thetas = eps_vecs[:, case.eps_indx[1]]
        radii = eps_vecs[:, case.eps_indx[0]]
        eps_list = []
        for it in range(case.Nxy):
            eps_copy = eps_vecs.copy()
            eps_copy[:, case.eps_indx[0]] = radii * xp.cos(thetas[it])
            eps_copy[:, case.eps_indx[1]] = radii * xp.sin(thetas[it])
            eps_list.append(eps_copy)
    convs = []
    iters = []
    if case.Parallelization[0]:
        if case.Parallelization[1] == 'all':
            num_cores = multiprocess.cpu_count()
        else:
            num_cores = min(multiprocess.cpu_count(), case.Parallelization[1])
        pool = multiprocess.Pool(num_cores)
        line_ = lambda it: line(eps_list[it], case)
        for conv, iter in tqdm(pool.imap(line_, iterable=range(case.Nxy)), total=case.Nxy):
            convs.append(conv)
            iters.append(iter)
    else:
        for it in trange(case.Nxy):
            conv, iter = line(eps_list[it], case)
            convs.append(conv)
            iters.append(iter)
    save_data('region', xp.array(convs), timestr, case, info=xp.array(iters))
    if case.PlotResults:
        divnorm = colors.TwoSlopeNorm(vmin=xp.amin(xp.array(iters)), vcenter=0.0, vmax=xp.amax(xp.array(iters)))
        if (case.Type == 'cartesian'):
            fig, ax = plt.subplots(1, 1)
            ax.set_box_aspect(1)
            im = ax.pcolormesh(eps_vecs[:, 0], eps_vecs[:, 1], xp.array(iters), norm=divnorm)
            ax.set_xlabel('$\epsilon_1$')
            ax.set_ylabel('$\epsilon_2$')
            fig.colorbar(im)
        elif (case.Type == 'polar'):
            r, theta = xp.meshgrid(radii, thetas)
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
            ax.contourf(theta, r, xp.array(iters))
    return xp.array(convs)

def save_data(name, data, timestr, case, info=[]):
    if case.SaveData:
        mdic = case.DictParams.copy()
        mdic.update({'data': data, 'info': info})
        date_today = date.today().strftime(" %B %d, %Y\n")
        mdic.update({'date': date_today, 'author': 'cristel.chandre@univ-amu.fr'})
        name_file = type(case).__name__ + '_' + name + '_' + timestr + '.mat'
        savemat(name_file, mdic)
        print('\033[90m        Results saved in {} \033[00m'.format(name_file))
