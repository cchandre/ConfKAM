# KAM in configuration space (ConfKAM) for invariant tori in Hamiltonian systems

- [`ConfKAM_dict.py`](https://github.com/cchandre/ConfKAM/blob/main/ConfKAM_dict.py): to be edited to change the parameters of the ConfKAM computation (see below for a dictionary of parameters)

- [`ConfKAM.py`](https://github.com/cchandre/ConfKAM/blob/main/ConfKAM.py): contains the ConfKAM classes and main functions defining the ConfKAM map

- [`ConfKAM_modules.py`](https://github.com/cchandre/ConfKAM/blob/main/ConfKAM_modules.py): contains the methods to execute the ConfKAM map

Once [`ConfKAM_dict.py`](https://github.com/cchandre/ConfKAM/blob/main/ConfKAM_dict.py) has been edited with the relevant parameters, run the file as 
```sh
python3.8 ConfKAM.py
```

___
##  Parameter dictionary

- *Method*: 'line_norm', 'region'; choice of method                                            
- *Nxy*: integer; number of points along each line in computations                                        
####                                                                                                   
- *omega0*: array of *n* floats; frequency vector of the invariant torus                                
- *Omega*: array of *n* floats; vector defining the perturbation in actions                             
- *dv*: function; derivative of the *n*-d potential along a line                                               
- *eps_region*: array of floats; min and max values of the amplitudes for each mode of the potential (see *dv*) 
- *eps_indx*: array of 2 integers; indices of the modes to be varied in `region()`                    
         in 'polar', radii are on *eps_indx[0]* and angles on *eps_indx[1]*                         
         parallelization in `region()` is done along the *eps_indx[0]* axis                               
- *eps_line*: 1d array of floats; min and max values of the amplitudes of the potential used in `line_norm()`   
- *eps_modes*: array of 0 and 1; specify which modes are being varied in `line_norm()` (1 for a varied mode)     
- *eps_dir*: 1d array of floats; direction of the one-parameter family used in `line_norm()`                 
####                                                                                           
- *r*: integer; order of the Sobolev norm used in `line_norm()`                                          
####                                                                                                           
- *AdaptL*: boolean; if True, changes the dimension of arrays depending on the tail of the FFT of *h*(*psi*)      
- *Lmin*: integer; minimum and default value of the dimension of arrays for *h*(*psi*)                           
- *Lmax*: integer; maximum value of the dimension of arrays for *h*(*psi*) if *AdaptL* is True                   
####                                                                                                         
- *TolMax*: float; value of norm for divergence                                                      
- *TolMin*: float; value of norm for convergence                                                           
- *Threshold*: float; threshold value for truncating Fourier series of *h*(*psi*)                                   
- *MaxIter*: integer; maximum number of iterations for the Newton method                                      
####                                                                                                         
- *Type*: 'cartesian', 'polar'; type of computation for 2d plots                                             
- *ChoiceInitial*: 'fixed', 'continuation'; method for the initial conditions of the Newton method             
####                                                                                                       
- *AdaptEps*: boolean; if True adapt the increment of eps in `line_norm()`                                   
- *MinEps*: float; minimum value of the increment of eps if *AdaptEps*=True                               
- *MonitorGrad*: boolean; if True, monitors the gradient of *h*(*psi*)                                      
####                                                                                 
- *Precision*: 32, 64 or 128; precision of calculations (default=64)                  
- *SaveData*: boolean; if True, the results are saved in a `.mat` file               
- *PlotResults*: boolean; if True, the results are plotted right after the computation              
- *Parallelization*: 2d array [boolean, int]; True for parallelization, int is the number of cores to be used (set int='all' for all of the cores)
####
---
For more information: <cristel.chandre@univ-amu.fr>
