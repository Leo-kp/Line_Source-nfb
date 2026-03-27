from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy import special as sc 
from scipy.optimize import brentq

import inspect
from skimage.restoration import denoise_tv_chambolle

def conv_eng_si(value,
               unit:str
):
  #si/eng
  factor={'psi': 6894.76,
          'psi-1': 1/6894.76,
          'acres': 4046.86,
          'ft': 0.3048,
          'h': 3600,
          'min':60,
          'cP': 0.001,
          'RB/STB':1,
          'STB/D': 0.0000018401,
          'vol_fraction':1,
          'md':9.8692326671601E-16
  }

  unit_c={'psi': 'Pa',
          'psi-1': 'Pa-1',
          'acres': 'm**2',
          'ft': 'm',
          'h': 's',
          'min':'s',
          'cP': 'Pa*s',
          'RB/STB': ' ',
          'STB/D': 'm**3/s',
          'vol_fraction':'vol_fraction',
          'md':'m**2'
  }

  try:
    conversion_factor = factor[unit]
    converted_value = value * conversion_factor
        
    return converted_value,unit_c[unit]
        
  except KeyError:
    print(f"Error: Unit '{unit}' not found in conversion dictionary.")

#-----------------------------------------------------------------------------
def pd_lsrf_nb(t, ddict, k, r):
    '''
    Pd Line-Source Radial-flow with no boundaries (infinite reservoir)
    Supports ONE vectorized parameter at a time (k or r).
    Dimensionless variables (tD, rD) are calculated relative to rw.
    Inputs.
        t: time
        ddict: dictionary with parameters/properties (mu, por, ct, rw)
        k: permeability (scalar or array)
        r: radius (scalar or array)
    Ouputs.
        pd: dimensionless pressure (array)
    '''
    t = np.atleast_1d(t).reshape(-1, 1)
    k = np.atleast_1d(k).flatten()[None, :]
    r = np.atleast_1d(r).flatten()[None, :]

    mu = ddict['mu']
    por = ddict['por']
    c_t = ddict['c_t']
    r_w = ddict['r_w'] 

    t_safe = np.maximum(t, 1e-12) #avoiding zero division
    td = (k * t_safe) / (por * mu * c_t * (r_w**2))
    rd = r / r_w  
    
    pd = -(1 / 2) * sc.expi(-(rd**2) / (4 * td))
    pd = np.where(t <= 0, 0.0, pd)
    
    pd.squeeze() #eliminate phantom 1 dimension/s
    return pd
#-----------------------------------------------------------------------------
def pd_lsrf_fb(t, ddict, k, r, re, N_terms=100):
    """
    Pd Line-Source Radial-flow with flow boundary (bounded circular reservoir)
    Supports ONE vectorized parameter at a time (k, r, or r_e).
    Dimensionless variables (tD, rD) are calculated relative to r_e.
    Inputs.
        t: time
        ddict: dictionary with parameters/properties (mu, por, ct, rw)
        k: permeability (scalar or array)
        r: radius (scalar or array)
        re: external/boundary radius (scalar or array)
    Ouputs.
        pd: dimensionless pressure (array)
    """
    t = np.atleast_1d(t).reshape(-1, 1) 
    k = np.atleast_1d(k).flatten()[None, :]
    r = np.atleast_1d(r).flatten()[None, :]
    re = np.atleast_1d(re).flatten()[None, :]

    mu = ddict['mu']
    por = ddict['por']
    c_t = ddict['c_t']

    t_safe = np.maximum(t, 1e-12)
    td_re = (k * t_safe) / (por * mu * c_t * (re**2))
    rd_re = r / re  
    
    xn = sc.jn_zeros(1, N_terms)
    xn_col = xn[:, np.newaxis, np.newaxis]
    
    j0_num = sc.jn(0, xn_col * rd_re)
    j0_den = (xn_col**2) * (sc.jn(0, xn_col)**2)
    exp_term = np.exp(-(xn_col**2) * td_re)
    series_sum = np.sum((j0_num / j0_den) * exp_term, axis=0)
    
    pd = 0.5 * (4 * td_re + rd_re**2 - 2 * np.log(rd_re) - 1.5 - 4 * series_sum)
    pd = np.where(t <= 0, 0.0, pd)

    pd.squeeze() #eliminate phantom 1 dimension/s
    return pd

#----------------------------------------------------------------------------

def find_roots_am(rw, re, n_roots=1000):
    """
    Roots for Finite-Source Radial-Flow with no flow boundary.
    This function calculates the eigenvalues (roots) used in analytical 
    solutions where the outer boundary has no flow.
    Based on the characteristic equation:
    Y1(a)*J1(a*red)-J1(a)*Y1(a*red)=0
    Inputs:
        rw: Wellbore radius (scalar)
        re: External/boundary radius (scalar or array)
        n_roots: Number of roots to find (default=1000)
    Outputs:
        roots_array: Array of roots formatted for broadcasting
    """
    re_vec = np.atleast_1d(re)
    all_re_roots = []
    
    for val_re in re_vec:
        red = val_re / rw
        roots = []
        f = lambda a: sc.j1(a * red) * sc.y1(a) - sc.j1(a) * sc.y1(a * red)
        
        step = np.pi / (red - 1)
        lower, upper = 1e-10, step
        while len(roots) < n_roots:
            if f(lower) * f(upper) < 0:
                roots.append(brentq(f, lower, upper))
            lower, upper = upper, upper + step
            if upper > 50000: break
        all_re_roots.append(roots)
    
    roots_array = np.array(all_re_roots).T 
    
    if roots_array.ndim == 1: 
        return roots_array[:, np.newaxis, np.newaxis]
    return roots_array[:, np.newaxis, :]
#------------------------------------
def pwd_fsrf_fb(t, ddict, k, r, re, am):
    """
    Pwd (at wellbore) finite-Source Radial-flow with no flow boundary (bounded circular reservoir)
    Supports ONE vectorized parameter at a time (k, r, or r_e).
    Dimensionless variables (tD, rD) are calculated relative to r_w.
    Inputs.
        t: time
        ddict: dictionary with parameters/properties (mu, por, ct, rw)
        k: permeability (scalar or array)
        r: radius (scalar or array) -> dummy input
        re: external/boundary radius (scalar or array)
        am: bessel function roots (array)
    Ouputs.
        pd: dimensionless pressure (array)
    """
    t = np.atleast_1d(t).reshape(-1, 1)          # (N_time, 1)
    k_vec = np.atleast_1d(k).flatten()[None, :]  # (1, N_k)
    re_vec = np.atleast_1d(re).flatten()[None, :] # (1, N_re)
    
    n_cols = max(k_vec.shape[1], re_vec.shape[1])
    
    rw = float(ddict['r_w'])
    mu, por, ct = ddict['mu'], ddict['por'], ddict['c_t']
    
    td = (k_vec * t) / (por * mu * ct * (rw**2))
    red = re_vec / rw  # (1, N_re)
    
    if td.shape[1] < n_cols:
        td = np.tile(td, (1, n_cols))
        
    if red.shape[1] < n_cols:
        red = np.tile(red, (1, n_cols))
        
    red2 = red**2

    am = np.asanyarray(am)
    if am.ndim == 2:
        am = am[:, np.newaxis, :]
    
    if am.shape[2] < n_cols:
        am = np.tile(am, (1, 1, n_cols))

    term_pss = (2 * td / red2) + np.log(red) - 0.75
    series_part = np.zeros_like(td)
    pss_mask = (td / red2) < 0.3 
    
    if np.any(pss_mask):
        row_idx, col_idx = np.where(pss_mask)
        
        td_active = td[row_idx, col_idx] 
        am_active = am[:, 0, col_idx]    
        red_active = red[0, col_idx]     
        
        j1_am_red = sc.j1(am_active * red_active)
        j1_am_rw = sc.j1(am_active)
        
        num = np.exp(-(am_active**2) * td_active) * (j1_am_red**2)
        den = (am_active**2) * (j1_am_red**2 - j1_am_rw**2)
        
        series_part[pss_mask] = 2.0 * np.sum(num / den, axis=0)

    pwd_raw = term_pss + series_part
    pwd = np.where(t <= 0, 0.0, pwd_raw - pwd_raw[0, :])
    pwd.squeeze() #eliminate phantom 1 dimension/s

    return pwd


#--------------------------------------------------------------------------
def find_roots_am_fs_cp(rw, re, n_roots=1000):
    """
    Roots for Finite-Source Radial-Flow with Constant Pressure Boundary.
    This function calculates the eigenvalues (roots) used in analytical 
    solutions where the outer boundary is maintained at a constant pressure.
    Based on the characteristic equation:
    J1(a) * Y0(a * red) - Y1(a) * J0(a * red) = 0
    Inputs:
        rw: Wellbore radius (scalar)
        re: External/boundary radius (scalar or array)
        n_roots: Number of roots to find (default=1000)
    Outputs:
        roots_array: Array of roots formatted for broadcasting
    """
    re_vec = np.atleast_1d(re)
    all_re_roots = []
    
    for val_re in re_vec:
        red = val_re / rw
        roots = []
        f = lambda a: sc.j1(a) * sc.y0(a * red) - sc.y1(a) * sc.j0(a * red)
        
        step = np.pi / (red - 1)
        lower, upper = 1e-10, step
        while len(roots) < n_roots:
            if f(lower) * f(upper) < 0:
                roots.append(brentq(f, lower, upper))
            lower, upper = upper, upper + step
            if upper > 100000: break
        all_re_roots.append(roots)
    
    roots_array = np.array(all_re_roots).T 
    
    if roots_array.ndim == 1: 
        return roots_array[:, np.newaxis, np.newaxis]
    return roots_array[:, np.newaxis, :]

#------------------------------------
def pd_fsrf_pb(t, ddict, k, r, re, am):
    """
    Pd finite-Source Radial-flow with constant-pressure boundary (bounded circular reservoir)
    Supports ONE vectorized parameter at a time (k, r, or r_e).
    Dimensionless variables (tD, rD) are calculated relative to r_w.
    Inputs.
        t: time
        ddict: dictionary with parameters/properties (mu, por, ct, rw)
        k: permeability (scalar or array)
        r: radius (scalar or array) -> dummy input
        re: external/boundary radius (scalar or array)
        am: bessel function roots (array)
    Ouputs.
        pd: dimensionless pressure (array)
    """
    t = np.atleast_1d(t).reshape(-1, 1)          # (N_time, 1)
    k_vec = np.atleast_1d(k).flatten()[None, :]  # (1, N_k)
    re_vec = np.atleast_1d(re).flatten()[None, :] # (1, N_re)
    r_vec = np.atleast_1d(r).flatten()[None, :]   # (1, N_r)
    
    n_cols = max(k_vec.shape[1], re_vec.shape[1], r_vec.shape[1])
    rw = float(ddict['r_w'])
    mu, por, ct = ddict['mu'], ddict['por'], ddict['c_t']
    
    td = (k_vec * t) / (por * mu * ct * (rw**2))
    red = re_vec / rw 
    rd = r_vec / rw
    
    if td.shape[1] < n_cols: td = np.tile(td, (1, n_cols))
    if red.shape[1] < n_cols: red = np.tile(red, (1, n_cols))
    if rd.shape[1] < n_cols: rd = np.tile(rd, (1, n_cols))

    am = np.asanyarray(am)
    if am.ndim == 2: am = am[:, np.newaxis, :]
    if am.shape[2] < n_cols: am = np.tile(am, (1, 1, n_cols))

    term_ss = np.log(red / rd)

    am_3d = am
    td_3d = td[np.newaxis, :, :]
    red_3d = red[np.newaxis, :, :]
    rd_3d = rd[np.newaxis, :, :]
    
    u0 = (sc.j0(am_3d * rd_3d) * sc.y1(am_3d) - 
          sc.y0(am_3d * rd_3d) * sc.j1(am_3d))
    
    j0_red = sc.j0(am_3d * red_3d)
    j1_rw = sc.j1(am_3d)
    
    num = np.exp(-(am_3d**2) * td_3d) * (j0_red**2) * u0
    den = (am_3d**2) * (j1_rw**2 - j0_red**2)
    
    series_part = -np.pi * np.sum(num / den, axis=0)

    pd_raw = term_ss - series_part
    
    pd = np.where(t <= 0, 0.0, pd_raw - pd_raw[0, :])
    
    return pd.squeeze()


#---------------------------------------------------------------------------
def step_rate_r(func, delta_t, tp, q_array, rd_dict, k_val, r_val, *args):
    """
    2D Optimized Step Rate for radial-flow family. 
    Supports ONE vectorized parameter (k, r, or an arg).
    Inputs.
        func: function-call to evaluate the steps 
        delta_t: time value of test
        tp: T time of step protocol change
        q_array: flowrate of test in every step T
        rd_dict: dictionary with parameters/properties (pi,mu,b,h)
        k_val: permeability value (scalar or array)
        r_val: radii of evaluation (scalar or array)
        *arg: ordered argument of function kernel used
    Ouput:
        p_ws: pressure of the step rate in consistent units. 
    """
    pi, mu, B = rd_dict['p_i'], rd_dict['mu'], rd_dict.get('B', 1.0)
    h = rd_dict.get('h', 1.0) #if not h or unsuccesful retrive, then h=1.0
    k_arr = np.atleast_1d(k_val).flatten()[None, :] 
    r_arr = np.atleast_1d(r_val).flatten()[None, :] 
   
    dt_matrix = delta_t[:, None] - tp[None, :]    # delta_t (N, 1), tp (1, M) -> dt_matrix (N, M)
    mask = dt_matrix > 1e-12
    dt_safe = np.where(mask, dt_matrix, 0.0)
    
    pwd_raw = func(dt_safe.ravel(), rd_dict, k_arr, r_arr, *args)
    num_times, num_events = dt_matrix.shape
    num_scenarios = pwd_raw.size // (num_times * num_events) #automatic escenarios
    
    pwd_3d = pwd_raw.reshape(num_times, num_events, num_scenarios)
    dq = np.diff(q_array, prepend=0)
    
    summation = np.einsum('j,ijk->ik', dq, pwd_3d) # Superposition via Einstein Summation:
    
    C_const = (mu * B) / (2 * np.pi * k_arr * h)
    p_ws = pi - (C_const * summation)
    
    p_ws = np.where(delta_t[:, None] <= 1e-12, pi, p_ws)
    
    offset = p_ws[0, :] - pi
    p_ws = p_ws - offset
    
    return p_ws.squeeze()

#----------------------------------------------------------------------------

def scale_and_smooth(series, w=0.1):
    """
    Smothing a signal with scaling and function
    arg. 
    series: data series
    w: weight of the smoothing
    Re.
    Smoothed curve in original scale
    """
    q_min = series.min()
    q_max = series.max()
    q_range = q_max - q_min
    
    if q_range == 0: return series # Avoid division by zero
    
    scaled = (series - q_min) / q_range
    
    smoothed = denoise_tv_chambolle(scaled.values, weight=w)
    
    return (smoothed * q_range) + q_min

#----------------------------------------------------------------------------

def pickings(series, window=20, sensitivity=5):
    """
    Picking based on steps by a moving average method
    arg. 
    series: data series
    window: moving std window
    sensitivity: sensitivity to noise
    Re.
    Smoothed curve in original scale
    """
    series_vals = series.values
    series_idx = series.index
    
    rows=[]
   
    rolling_std = series.rolling(window=window).std().fillna(0).values #function of moving standard deviation
    
    last_val = series_vals[0]
    i = window
    
    while i < len(series_vals) - 1:
        current_val = series_vals[i]
        local_noise = rolling_std[i]

        dynamic_threshold = (local_noise * sensitivity) + 0.05 #0.05 for not triggering in silence
        
        if abs(current_val - last_val) > dynamic_threshold:
            rows.append([series_idx[i], current_val]) 
            last_val = current_val
            i += window
        else:
            i += 1

    return   np.array(rows)#transpose the shape


#----------------------------------------------------------------------------