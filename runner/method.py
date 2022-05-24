# Copyright 2022 Luping Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import copy
import torch as th
import numpy as np
from functools import partial
import pdb


def choose_method(name):
    if name == 'DDIM':
        return gen_order_1
    elif name == 'FON':
        return gen_fon
    elif name == 'PRK':
        return Psedo_rk
    elif name == 'S-PNDM':
        return gen_order_2
    elif name == 'F-PNDM':
        return correct_order_4
    elif name.startswith("PNDM"):
        return partial(Psedo_AB, order=int(name[4]))
    elif name.startswith("CAB"):
        return partial(Correct_AB1, order=int(name[3]))
    elif name.startswith("CCAB"):
        return partial(Correct_ABM, order=int(name[4]))
        #return partial(Correct_AB2, order=int(name[4]),n=float(name[6:]))
        #return partial(Correct_AB,  order=int(name[4]),n=float(name[6:]))
        #return gen_order_4
    elif name.startswith("NEWT"):
        #return partial(Newton_costom, order=int(name[4]))
        return partial(Newton, order=int(name[4]))
    elif name == "CRK":
        return Correct_rk
    elif name.startswith("CMIX"):
        return partial(Custom_MIX, order=int(name[4]))
    elif name.startswith("DEIS"):
        return partial(Expo_Inte, order=int(name[4]))
    elif name == 'PF':
        return gen_pflow
    else:
        return None

def choose_gamma_method(name):
    if name == 'DDIM':
        return pre_1order, cor_1order
    else:
        return pre_1order, cor_1order

def choose_line_method(name):
    if name == 'DDIM':
        return opt_1order, Correct_rk_eps
    else:
        return opt_1order, Correct_rk_eps

def pre_1order(img, g, g_next, model, gamma, ets):
    st = th.sqrt(g**2 + 1)
    sn = th.sqrt(g_next**2 + 1)
    x_bar = img * st  
    t = get_t_from_g(g, gamma, size=img.shape[0])
    noise = model(img, t)
    ets.append(noise)
    img_next = (x_bar +  (g_next - g) * noise)/sn
    return img_next, ets[-1]

def get_eps(img, g, model, gamma):
    t = get_t_from_g(g, gamma, size=img.shape[0])
    noise = model(img, t)
    return noise

def rkf45(img, g, g_next, model, gamma, tau = 1e-5):
    #runge kutta fehlberg
    dg = (g_next - g)
    g0 = g
    g1 = g + dg/4
    g2 = g + dg*3/8
    g3 = g + dg*12/13
    g4 = g + dg
    g5 = g + dg/2
    s0 = th.sqrt(g0**2 + 1)
    s1 = th.sqrt(g1**2 + 1)
    s2 = th.sqrt(g2**2 + 1)
    s3 = th.sqrt(g3**2 + 1)
    s4 = th.sqrt(g4**2 + 1)
    s5 = th.sqrt(g5**2 + 1)
    t0 = get_t_from_g(g0, gamma, size=img.shape[0])
    t1 = get_t_from_g(g1, gamma, size=img.shape[0])
    t2 = get_t_from_g(g2, gamma, size=img.shape[0])
    t3 = get_t_from_g(g3, gamma, size=img.shape[0])
    t4 = get_t_from_g(g4, gamma, size=img.shape[0])
    t5 = get_t_from_g(g5, gamma, size=img.shape[0])
    x_bar = img * s0

    e1 = model(img, t0)
    x1 = (x_bar + dg*(e1/4)) / s1

    e2 = model(x1, t1)
    x2 = (x_bar + dg*(e1*3/32      + e2*9/32)) / s2

    e3 = model(x2, t2)
    x3 = (x_bar + dg*(e1*1932/2197 - e2*7200/2197 + e3*7296/2197)) / s3

    e4 = model(x3, t3)
    x4 = (x_bar + dg*(e1*439/216 - e2*8     + e3*3680/513  - e4*845/4104)) / s4

    e5 = model(x4, t4)
    x5 = (x_bar + dg*(-e1*8/27 + e2*2        - e3*3544/2565 + e4*1859/4104  - e5*11/40)) / s5

    e6 = model(x5, t5)
    y = (x_bar + dg*(e1*25/216 + e3*1408/2565   + e4*2197/4101 - e5*1/5)) / s4
    z = (x_bar + dg*(e1*16/135 + e3*6656/12825  + e4*28561/56430 - e5*9/50 + e6*2/55)) / s4

    #print(th.mean((y-z)**2))
    if th.mean((y-z)**2)<tau:
        return z
    else:
        gm = (g+g_next)/2
        x = rkf45(img, g, gm, model, gamma, tau = tau)
        return  rkf45(x, gm, g_next, model, gamma, tau = tau)

def rkf45_(img, g, g_next, model, gamma, tau = 1):
    dg = (g_next - g)
    g0 = g
    g1 = g + dg/2
    g2 = g + dg*1/2
    g3 = g + dg
    s0 = th.sqrt(g0**2 + 1)
    s1 = th.sqrt(g1**2 + 1)
    s2 = th.sqrt(g2**2 + 1)
    s3 = th.sqrt(g3**2 + 1)
    t0 = get_t_from_g(g0, gamma, size=img.shape[0])
    t1 = get_t_from_g(g1, gamma, size=img.shape[0])
    t2 = get_t_from_g(g2, gamma, size=img.shape[0])
    t3 = get_t_from_g(g3, gamma, size=img.shape[0])
    x_bar = img * s0

    e1 = model(img, t0)
    x1 = (x_bar + dg*(e1/2)) / s1

    e2 = model(x1, t1)
    x2 = (x_bar + dg*(e2/2)) / s2

    e3 = model(x2, t2)
    x3 = (x_bar + dg*(e3)) / s3

    e4 = model(x3, t3)
    x = (x_bar + dg*(e1*1/6  + e2*2/6 + e3*2/6 + e4*1/6)) / s3

    return x

def cor_1order(img, g, g_next, model, gamma, ets):
    st = th.sqrt(g**2 + 1)
    sn = th.sqrt(g_next**2 + 1)
    x_bar = img * st  
    eps = (ets[-2] + ets[-1])/2
    img_next = (x_bar +  (g_next - g) * eps)/sn
    return eps

def opt_1order(img, g, g_next, model, gamma, ets=None):
    st = th.sqrt(g**2 + 1)
    sn = th.sqrt(g_next**2 + 1)
    x_bar = img * st  
    t = get_t_from_g(g, gamma, size=img.shape[0])
    ets = model(img, t)
    #ets.append(noise)
    #img_next = transfer(img, t, t_next, noise, alphas_cump)
    img_next = (x_bar +  (g_next - g) * ets)/sn
    return img_next, ets

def Correct_rk_eps(x, g, g_next, model, gamma, ets = None):
    #gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    #at = alphas_cump[t[0].long()+1]
    #ad = alphas_cump[t_next[0].long()+1]
    #gt = th.sqrt((1-at)/at)
    #gd = th.sqrt((1-ad)/ad)
    gm = (g+g_next)/2
    sqam = 1/th.sqrt(gm**2+1)
    dg = (g_next-g)
    st = th.sqrt(g**2 + 1)
    sn = th.sqrt(g_next**2 + 1)
    x_bar = x * st
    
    t = get_t_from_g(g, gamma, size=x.shape[0])
    tm = get_t_from_g(gm, gamma, size=x.shape[0])
    t_next = get_t_from_g(g_next, gamma, size=x.shape[0])

    e_1 = model(x, t)
    #ets.append(e_1)
    x_2 = (x_bar + dg/2 * e_1) * sqam

    e_2 = model(x_2, tm)
    x_3 = (x_bar + dg/2 * e_2) * sqam

    e_3 = model(x_3, tm)
    x_4 = (x_bar + dg * e_3) / sn #th.sqrt(ad)

    e_4 = model(x_4, t_next)
    noise = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    #img_next = transfer(x, t, t_next, noise, alphas_cump)
    img_next = (x_bar +  dg * noise)/sn
    return img_next, noise

def get_t_from_g(g, gamma, size=None):
    tt = th.argmin(abs(gamma-g))   
    if tt > len(gamma) - 2 : tc = tt
    else:
        aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
        bb = (gamma[tt+1] - gamma[tt-1])/2
        cc = gamma[tt] - g
        tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    if size is None:
        return tc
    else:
        return tc * th.ones(size, device=tc.device)

def gen_pflow(img, t, t_next, model, betas, total_step):
    n = img.shape[0]
    beta_0, beta_1 = betas[0], betas[-1]

    t_start = th.ones(n, device=img.device) * t
    beta_t = (beta_0 + t_start * (beta_1 - beta_0)) * total_step

    log_mean_coeff = (-0.25 * t_start ** 2 * (beta_1 - beta_0) - 0.5 * t_start * beta_0) * total_step
    std = th.sqrt(1. - th.exp(2. * log_mean_coeff))

    # drift, diffusion -> f(x,t), g(t)
    drift, diffusion = -0.5 * beta_t.view(-1, 1, 1, 1) * img, th.sqrt(beta_t)
    score = - model(img, t_start * (total_step - 1)) / std.view(-1, 1, 1, 1)  # score -> noise
    drift = drift - diffusion.view(-1, 1, 1, 1) ** 2 * score * 0.5  # drift -> dx/dt

    return drift


def gen_fon(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t + t_next) / 2.0, t_next]
    if len(ets) > 2:
        noise = model(img, t)
        img_next = transfer(img, t, t-1, noise, alphas_cump)
        delta1 = img_next - img
        ets.append(delta1)
        delta = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = model(img, t_list[0])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_1 = img_ - img
        ets.append(delta_1)

        img_2 = img + delta_1 * (t - t_next).view(-1, 1, 1, 1) / 2.0
        noise = model(img_2, t_list[1])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_2 = img_ - img

        img_3 = img + delta_2 * (t - t_next).view(-1, 1, 1, 1) / 2.0
        noise = model(img_3, t_list[1])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_3 = img_ - img

        img_4 = img + delta_3 * (t - t_next).view(-1, 1, 1, 1)
        noise = model(img_4, t_list[2])
        img_ = transfer(img, t, t - 1, noise, alphas_cump)
        delta_4 = img_ - img
        delta = (1 / 6.0) * (delta_1 + 2*delta_2 + 2*delta_3 + delta_4)

    img_next = img + delta * (t - t_next).view(-1, 1, 1, 1)
    return img_next


def gen_order_4(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t+t_next)/2, t_next]
    if len(ets) > 2:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = runge_kutta(img, t_list, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next

def Psedo_AB(img, t, t_next, model, alphas_cump, ets, order = 4):
    ets.append(model(img, t))
    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next

def Psedo_AB2(img, t, t_next, model, alphas_cump, ets, order = 4):
    # with t square
    t2 = t**2/1000
    t2_next = t_next**2/1000
    
    ets.append(model(img, t2))
    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    img_next = transfer(img, t2, t2_next, noise, alphas_cump)
    return img_next

def Psedo_rk(x, t, t_next, model, alphas_cump, ets):
    t_mid = (t+t_next)/2
    e_1 = model(x, t)
    ets.append(e_1)
    x_2 = transfer(x, t, t_mid, e_1, alphas_cump)

    e_2 = model(x_2, t_mid)
    x_3 = transfer(x, t, t_mid, e_2, alphas_cump)

    e_3 = model(x_3, t_mid)
    x_4 = transfer(x, t, t_next, e_3, alphas_cump)

    e_4 = model(x_4, t_next)
    noise = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    img_next = transfer(x, t, t_next, noise, alphas_cump)
    return img_next

def Psedo_ABM(img, t, t_next, model, alphas_cump, ets, order = 4):
    ets.append(model(img, t))
    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    img_temp = transfer(img, t, t_next, noise, alphas_cump)
    ets0 = model(img_temp, t_next)
    if cur==0:
        noise = ets0
    elif cur == 1:
        noise = (ets0 +  ets[-1] ) / 2
    elif cur == 2:
        noise = (5 * ets0 +  8 * ets[-1] -  1 * ets[-2]) / 12
    else:
        noise = (9 * ets0 +  19 * ets[-1] -  5 * ets[-2] +  1 * ets[-2]) / 24

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next

def Correct_AB1(img, t, t_next, model, alphas_cump, ets, order = 4):
    ## Normal and standard dg/dt = 1
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    gamma_max = gamma[980] 
    del_gam = gamma_max / 980 
    #gamma_max = gamma[-1]
    #del_gam = gamma_max / len(gamma) 
    g0 = t[0] * del_gam
    g_1 = t_next[0] *del_gam
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g_1**2 + 1)
    x_bar = img * st  

    tt = th.argmin(abs(gamma-g0))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g0
    tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    tc = tc * th.ones_like(t)
    
    ets.append(model(img, tc))
    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    img_next = (x_bar + (g_1-g0) * noise)/sn
    return img_next

def Correct_AB2(img, t, t_next, model, alphas_cump, ets, order = 4, n =2):
    ## custom dg/dt = t^(n-1)
    L = 980 + 20
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    gamma_max = gamma[980] 
    Ln = np.power(L,n)
    tau_max = th.pow(Ln/(Ln-np.power(20,n))*n*gamma_max,1/n)
    #tau_max = th.pow(n*gamma_max,1/n)
    del_tau = tau_max / L #960
    #print(t[0],t_next[0])
    #pdb.set_trace()
    #np.savetxt("0.txt",gamma.cpu().numpy())

    t0 =  (t[0]+20) * del_tau
    t_1 = (t_next[0]+20) * del_tau
    g0 = th.pow(t0,n)/n   - th.pow(del_tau*20,n)/n
    g_1 = th.pow(t_1,n)/n - th.pow(del_tau*20,n)/n
    #pdb.set_trace()
    #print(t0,t_1,del_tau*20)
    #print(g0,g_1,gamma_max)
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g_1**2 + 1)
    x_bar = img * st  

    tt = th.argmin(abs(gamma-g0))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g0
    tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    tc = tc * th.ones_like(t)
    #print(g_1 - g0)
    
    ets.append(model(img, tc) * th.pow(t0,n-1))
    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    img_next = (x_bar + (t_1-t0) * noise)/sn
    return img_next

def Correct_AB_backup(img, t, t_next, model, alphas_cump, ets, order = 4, n =2):
    ## Normal and standard dg/dt = 1
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    #pdb.set_trace()
    g0 = gamma[t[0].cpu().numpy()]
    g_1 = gamma[t_next[0].cpu().numpy()]
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g_1**2 + 1)
    x_bar = img * st  
    
    ets.append(model(img, t))
    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    img_next = (x_bar +  (g_1 - g0) * noise)/sn
    return img_next

def Correct_AB(img, t, t_next, model, alphas_cump, ets, order = 4, n =2):
    ## implicit dg/dt
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    g0 = gamma[(t[0]+1).cpu().numpy()]
    g_1 = gamma[(t_next[0]+1).cpu().numpy()]
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g_1**2 + 1)
    x_bar = img * st  

    if False:
        ets.append(model(img, t))
    elif order == 1:
        ets.append(model(img, t) * (g_1 - g0)/(t_next[0] - t[0]))
    else:
        del_g = (gamma[(t[0]+1).cpu().numpy()] - gamma[(t[0]-1).cpu().numpy()])/2
        del_g1 = (g_1 - g0)/(t_next[0] - t[0])
        #del_g = th.max(del_g,del_g1)
        ets.append(model(img, t) * del_g )
        #print(del_g, del_g1 )

    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    #img_next = (x_bar +  (g_1 - g0) * noise/del_g)/sn
    img_next = (x_bar +  (t_next[0] - t[0]) * noise) /  sn
    return img_next

def Correct_ABM1(img, t, t_next, model, alphas_cump, ets, order = 4):
    ## Adams-Bashforth-Moulton Predictor-Corrector
    ## Normal and standard dg/dt = 1
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    gamma_max = gamma[980] 
    del_gam = gamma_max / 980 
    #gamma_max = gamma[-1]
    #del_gam = gamma_max / len(gamma) 
    g0 = t[0] * del_gam
    g_1 = t_next[0] *del_gam
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g_1**2 + 1)
    x_bar = img * st  

    tt = th.argmin(abs(gamma-g0))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g0
    tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    tc = tc * th.ones_like(t)
    
    ets.append(model(img, tc))
    cur = min(order,len(ets))
    ## Predictor
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    img_temp = (x_bar + (g_1-g0) * noise)/sn
    ## Corrector 
    tt = th.argmin(abs(gamma-g_1))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g_1
    #tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    #tc = tc * th.ones_like(t)

    e0 = model(img_temp, tc)

    if cur==0:
        noise = e0
    elif cur == 1:
        noise = (e0 + ets[-1] ) / 2
    elif cur == 2:
        noise = (5 * e0 + 8 * ets[-1] - 1 * ets[-2]) / 12
    else:
        noise = (9 * e0 +  19 * ets[-1] - 5 * ets[-2] + 1 * ets[-3]) / 24

    img_next = (x_bar + (g_1-g0) * noise)/sn
    return img_next

def Correct_ABM2(img, t, t_next, model, alphas_cump, ets, order = 4, n =2):
    ## Adams-Bashforth-Moulton Predictor-Corrector
    ## custom dg/dt = t^(n-1)
    L = 980 + 20
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    gamma_max = gamma[980] 
    Ln = np.power(L,n)
    tau_max = th.pow(Ln/(Ln-np.power(20,n))*n*gamma_max,1/n)
    #tau_max = th.pow(n*gamma_max,1/n)
    del_tau = tau_max / L #960
    #print(t[0],t_next[0])
    #pdb.set_trace()
    #np.savetxt("0.txt",gamma.cpu().numpy())

    t0 =  (t[0]+20) * del_tau
    t_1 = (t_next[0]+20) * del_tau
    g0 = th.pow(t0,n)/n   - th.pow(del_tau*20,n)/n
    g_1 = th.pow(t_1,n)/n - th.pow(del_tau*20,n)/n
    #pdb.set_trace()
    #print(t0,t_1,del_tau*20)
    #print(g0,g_1,gamma_max)
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g_1**2 + 1)
    x_bar = img * st  

    tt = th.argmin(abs(gamma-g0))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g0
    tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    tc = tc * th.ones_like(t)
    #print(g0,t0,tc[0])
    
    ets.append(model(img, tc) * th.pow(t0,n-1))
    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    #print(t0)
    #if t0.long()>1.:
    if True:
        img_temp = (x_bar +  (t_1-t0) * noise) /  sn

        tt = th.argmin(abs(gamma-g_1))
        aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
        bb = (gamma[tt+1] - gamma[tt-1])/2
        cc = gamma[tt] - g_1
        tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
        tc = tc * th.ones_like(t)

        ets0 = model(img_temp, tc)  * th.pow(t_1,n-1)
        if cur==0:
            noise = ets0
        elif cur == 1:
            noise = (ets0 +  ets[-1] ) / 2
        elif cur == 2:
            noise = (5 * ets0 +  8 * ets[-1] -  1 * ets[-2]) / 12
        elif cur == 3:
            noise = (9 * ets0 +  19 * ets[-1] -  5 * ets[-2] +  1 * ets[-2]) / 24
        else:
            noise = (251 * ets0 +  646 * ets[-1] -  264 * ets[-2] +  106 * ets[-2] - 19 * ets[-3]) / 720

    img_next = (x_bar + (t_1-t0) * noise)/sn

    return img_next

def Correct_ABM(img, t, t_next, model, alphas_cump, ets, order = 4, n =2):
    ## Adams-Bashforth-Moulton Predictor-Corrector
    ## implicit dg/dt
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    g0 = gamma[(t[0]+1).cpu().numpy()]
    g_1 = gamma[(t_next[0]+1).cpu().numpy()]
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g_1**2 + 1)
    x_bar = img * st  

    if True:
        ets.append(model(img, t))
    elif order == 1:
        ets.append(model(img, t) * (g_1 - g0)/(t_next[0] - t[0]))
    else:
        del_g = (gamma[(t[0]+1).cpu().numpy()] - gamma[(t[0]-1).cpu().numpy()])/2
        del_g1 = (g_1 - g0)/(t_next[0] - t[0])
        #del_g = th.max(del_g,del_g1)
        ets.append(model(img, t) * del_g )
        #print(del_g, del_g1 )

    cur = min(order,len(ets))
    if cur==1:
        noise = ets[-1]
    elif cur == 2:
        noise = (3 * ets[-1] - ets[-2] ) / 2
    elif cur == 3:
        noise = (23 * ets[-1] - 16 * ets[-2] + 5 * ets[-3]) / 12
    else:
        noise = (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4]) / 24

    #img_next = (x_bar +  (g_1 - g0) * noise/del_g)/sn
    #img_next = (x_bar +  (t_next[0] - t[0]) * noise) /  sn
    if True:
        img_temp = transfer(img, t, t_next, noise, alphas_cump)
        #img_temp = (x_bar +  (t_next[0] - t[0]) * noise) /  sn

        del_g = (gamma[(t_next[0]+1).cpu().numpy()] - gamma[(t_next[0]-1).cpu().numpy()])/2
        ets0 = model(img_temp, t_next) * del_g
        if cur==0:
            noise2 = ets0
        elif cur == 1:
            noise2 = (ets0 +  ets[-1] ) / 2
        elif cur == 2:
            noise2 = (5 * ets0 +  8 * ets[-1] -  1 * ets[-2]) / 12
        elif cur == 3:
            noise2 = (9 * ets0 +  19 * ets[-1] -  5 * ets[-2] +  1 * ets[-2]) / 24
        else:
            noise2 = (251 * ets0 +  646 * ets[-1] -  264 * ets[-2] +  106 * ets[-2] - 19 * ets[-3]) / 720
        print(th.norm(noise-noise2))

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    #img_next = (x_bar +  (t_next[0] - t[0]) * noise) /  sn
    return img_next

def Newton(img, t, t_next, model, alphas_cump, ets, order = 4, n =2):
    ## Newton polynomial
    e0 = model(img, t)
    #pdb.set_trace()
    at = alphas_cump[t[0].long()+1]
    ad = alphas_cump[t_next[0].long()+1]
    #at = alphas_cump[t[0].cpu().numpy()]
    #ad = alphas_cump[t_next[0].cpu().numpy()]
    g0 = th.sqrt((1-at)/at)
    g = th.sqrt((1-ad)/ad)

    ets.append({
        "eps":e0,
        "gamma":g0
    })
    #ets.append(e0)

    cur = min(order,len(ets))
    if cur>=1:
        noise = ets[-1]["eps"]
    if cur >= 2:
        e1 = ets[-2]["eps"]
        g1 = ets[-2]["gamma"]
        e0p1 = (e0-e1)/(g0-g1)
        ets[-1]["eps1"] = e0p1
        noise = noise + e0p1 * (g - g0) / 2
    if cur >= 3:
        g2 = ets[-3]["gamma"]
        e1p1 = ets[-2]["eps1"]
        e0p2 = (e0p1-e1p1) / (g0-g2) 
        ets[-1]["eps2"] = e0p2
        noise = noise + e0p2 * (2 * (g - g0)**2  - 3 *(g - g0)*(g1 - g0)) / 6
    if cur >= 4: #untested
        g3 = ets[-4]["gamma"]
        e1p2 = ets[-2]["eps2"]
        e0p3 = (e0p2-e1p2) / (g0-g3) 
        noise = noise + e0p3 * (g-g0) * (4*(g1-g)*(g2-g) + 2*(g1-g0)*(g2-g0) - (g-g0)**2) / 12

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next

def Newton2(img, t, t_next, model, alphas_cump, ets, order = 4, n =2):
    ## Newton polynomial with t square
    t2 = t**2/1000
    t2_next = t_next**2/1000

    e0 = model(img, t2)
    at = alphas_cump[t2[0].cpu().numpy()]
    ad = alphas_cump[t2_next[0].cpu().numpy()]
    g0 = th.sqrt((1-at)/at)
    g = th.sqrt((1-ad)/ad)

    ets.append({
        "eps":e0,
        "gamma":g0
    })
    #ets.append(e0)

    cur = min(order,len(ets))
    if cur>=1:
        noise = ets[-1]["eps"]
    if cur >= 2:
        e1 = ets[-2]["eps"]
        g1 = ets[-2]["gamma"]
        e0p1 = (e0-e1)/(g0-g1)
        ets[-1]["eps1"] = e0p1
        noise = noise + e0p1 * (g - g0) / 2
    if cur >= 3:
        g2 = ets[-3]["gamma"]
        e1p1 = ets[-2]["eps1"]
        e0p2 = (e0p1-e1p1) / (g0-g2) 
        ets[-1]["eps2"] = e0p2
        noise = noise + e0p2 * (2 * (g - g0)**2  - 3 *(g - g0)*(g1 - g0)) / 6
    if cur >= 4: #untested
        g3 = ets[-4]["gamma"]
        e1p2 = ets[-2]["eps2"]
        e0p3 = (e0p2-e1p2) / (g0-g3) 
        noise = noise + e0p3 * (g-g0) * (4*(g1-g)*(g2-g) + 2*(g1-g0)*(g2-g0) - (g-g0)**2) / 12

    img_next = transfer(img, t2, t2_next, noise, alphas_cump)
    return img_next

def Newton_costom(img, t, t_next, model, alphas_cump, ets, order = 4, n =4):
    ## Newton polynomial with custom dg/dt = t^(n-1)
    L = 980 + 20
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    gamma_max = gamma[980] 
    Ln = np.power(L,n)
    tau_max = th.pow(Ln/(Ln-np.power(20,n))*n*gamma_max,1/n)
    del_tau = tau_max / L #960

    t0 =  (t[0]+20) * del_tau
    t_1 = (t_next[0]+20) * del_tau
    g0 = th.pow(t0,n)/n   - th.pow(del_tau*20,n)/n
    g = th.pow(t_1,n)/n - th.pow(del_tau*20,n)/n
    #pdb.set_trace()
    #print(t0,t_1,del_tau*20)
    #print(g0,g_1,gamma_max)
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g**2 + 1)
    x_bar = img * st  

    tt = th.argmin(abs(gamma-g0))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g0
    tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    tc = tc * th.ones_like(t)

    tt = th.argmin(abs(gamma-g))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g
    tc_next = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    tc_next = tc_next * th.ones_like(t)

    #e0 = model(img, t2)
    #at = alphas_cump[t2[0].cpu().numpy()]
    #ad = alphas_cump[t2_next[0].cpu().numpy()]
    #g0 = th.sqrt((1-at)/at)
    #g = th.sqrt((1-ad)/ad)
    e0 = model(img, tc)

    ets.append({
        "eps":e0,
        "gamma":g0
    })
    #ets.append(e0)

    cur = min(order,len(ets))
    if cur>=1:
        noise = ets[-1]["eps"]
    if cur >= 2:
        e1 = ets[-2]["eps"]
        g1 = ets[-2]["gamma"]
        e0p1 = (e0-e1)/(g0-g1)
        ets[-1]["eps1"] = e0p1
        noise = noise + e0p1 * (g - g0) / 2
    if cur >= 3:
        g2 = ets[-3]["gamma"]
        e1p1 = ets[-2]["eps1"]
        e0p2 = (e0p1-e1p1) / (g0-g2) 
        ets[-1]["eps2"] = e0p2
        noise = noise + e0p2 * (2 * (g - g0)**2  - 3 *(g - g0)*(g1 - g0)) / 6
    if cur >= 4: #untested
        g3 = ets[-4]["gamma"]
        e1p2 = ets[-2]["eps2"]
        e0p3 = (e0p2-e1p2) / (g0-g3) 
        noise = noise + e0p3 * (g-g0) * (4*(g1-g)*(g2-g) + 2*(g1-g0)*(g2-g0) - (g-g0)**2) / 12

    img_next = transfer(img, tc, tc_next, noise, alphas_cump)
    return img_next

def Custom_MIX(img, t, t_next, model, alphas_cump, ets, order = 4, n =4):
    ## custom dg/dt = t^(n-1)
    L = 980 + 20
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    gamma_max = gamma[980] 
    Ln = np.power(L,n)
    tau_max = th.pow(Ln/(Ln-np.power(20,n))*n*gamma_max,1/n)
    del_tau = tau_max / L #960

    t0 =  (t[0]+20) * del_tau
    t_1 = (t_next[0]+20) * del_tau
    g0 = th.pow(t0,n)/n   - th.pow(del_tau*20,n)/n
    g = th.pow(t_1,n)/n - th.pow(del_tau*20,n)/n

    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g**2 + 1)
    x_bar = img * st  

    tt = th.argmin(abs(gamma-g0))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g0
    tc = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    tc = tc * th.ones_like(t)

    tt = th.argmin(abs(gamma-g))
    aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
    bb = (gamma[tt+1] - gamma[tt-1])/2
    cc = gamma[tt] - g
    tc_next = tt + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa
    tc_next = tc_next * th.ones_like(t)

    e0 = model(img, tc)

    ets.append({
        "eps":e0,
        "gamma":g0,
        "d_g": th.pow(t0,n-1)
    })
    #ets.append(e0)
    #print(g0 - g)
    #print((g0 - g)/gamma_max)

    cur = min(order,len(ets))
    if (g0 - g).cpu().numpy()<1 and t[0]<998:
        
        if cur>=1:
            d0 =  ets[-1]["d_g"]
            noise = e0 * d0
        if cur >= 2:
            e1 = ets[-2]["eps"]
            g1 = ets[-2]["gamma"]
            e0p1 = (e0-e1)/(g0-g1)
            ets[-1]["eps1"] = e0p1
            d1 = ets[-2]["d_g"]
            noise = (3 * e0*d0 - e1*d1 ) / 2
        if cur >= 3:
            e2 = ets[-3]["eps"]
            g2 = ets[-3]["gamma"]
            e1p1 = ets[-2]["eps1"]
            e0p2 = (e0p1-e1p1) / (g0-g2) 
            ets[-1]["eps2"] = e0p2
            d2 = ets[-3]["d_g"]
            noise = (23 * e0*d0 - 16 * e1*d1 + 5 * e2*d2) / 12
        if cur >=4:
            e3 = ets[-4]["eps"]
            g3 = ets[-4]["gamma"]
            e1p2 = ets[-2]["eps2"]
            e0p3 = (e0p2-e1p2) / (g0-g3) 
            d3 = ets[-4]["d_g"]
            noise = (55 * e0*d0 - 59 * e1*d1 + 37 * e2*d2 - 9 * e3*d3) / 24

        img_next = (x_bar + (t_1-t0) * noise)/sn

    else: #NEWTON
        if cur>=1:
            noise = ets[-1]["eps"]
        if cur >= 2:
            e1 = ets[-2]["eps"]
            g1 = ets[-2]["gamma"]
            e0p1 = (e0-e1)/(g0-g1)
            ets[-1]["eps1"] = e0p1
            noise = noise + e0p1 * (g - g0) / 2
        if cur >= 3:
            g2 = ets[-3]["gamma"]
            e1p1 = ets[-2]["eps1"]
            e0p2 = (e0p1-e1p1) / (g0-g2) 
            ets[-1]["eps2"] = e0p2
            noise = noise + e0p2 * (2 * (g - g0)**2  - 3 *(g - g0)*(g1 - g0)) / 6
        if cur >= 4: #untested
            g3 = ets[-4]["gamma"]
            e1p2 = ets[-2]["eps2"]
            e0p3 = (e0p2-e1p2) / (g0-g3) 
            noise = noise + e0p3 * (g-g0) * (4*(g1-g)*(g2-g) + 2*(g1-g0)*(g2-g0) - (g-g0)**2) / 12

        img_next = transfer(img, tc, tc_next, noise, alphas_cump)
    return img_next

def Correct_rk(x, t, t_next, model, alphas_cump, ets):
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    at = alphas_cump[t[0].long()+1]
    ad = alphas_cump[t_next[0].long()+1]
    gt = th.sqrt((1-at)/at)
    gd = th.sqrt((1-ad)/ad)
    gm = (gt+gd)/2
    sqam = 1/th.sqrt(gm**2+1)
    dg = (gd-gt)
    x_bar = x/th.sqrt(at)
    
    if t[0] == t_next[0]:
        tm = t
    else:
        tt = th.argmin(abs(gamma-gm))
        aa = (gamma[tt+1] + gamma[tt-1])/2 - gamma[tt]
        bb = (gamma[tt+1] - gamma[tt-1])/2
        cc = gamma[tt] - gm
        tm = tt - 1 + (-bb + th.sqrt(bb**2-4*aa*cc))/2/aa 
        tm = tm * th.ones_like(t)

    e_1 = model(x, t)
    ets.append(e_1)
    x_2 = (x_bar + dg/2 * e_1) * sqam

    e_2 = model(x_2, tm)
    x_3 = (x_bar + dg/2 * e_2) * sqam

    e_3 = model(x_3, tm)
    x_4 = (x_bar + dg * e_3) * th.sqrt(ad)

    e_4 = model(x_4, t_next)
    noise = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    img_next = transfer(x, t, t_next, noise, alphas_cump)
    return img_next

def Expo_Inte(img, t, t_next, model, alphas_cump, ets, order = 4, n =2):
    ## Exponential Integrator  linear t
    e0 = model(img, t)
    #pdb.set_trace()
    at = alphas_cump[t[0].long()+1]
    ad = alphas_cump[t_next[0].long()+1]
    #at = alphas_cump[t[0].cpu().numpy()]
    #ad = alphas_cump[t_next[0].cpu().numpy()]
    gamma = th.sqrt((1-alphas_cump)/alphas_cump)
    g0 = th.sqrt((1-at)/at)
    g = th.sqrt((1-ad)/ad)
    st = th.sqrt(g0**2 + 1)
    sn = th.sqrt(g**2 + 1)
    x_bar = img * st  

    ets.append({
        "eps":e0,
        "gamma":g0,
        "time":t[0],
    })
    #ets.append(e0)

    cur = min(order,len(ets))
    if cur>=1:
        noise = (g - g0) * ets[-1]["eps"]
    if cur >= 2:
        e1 = ets[-2]["eps"]
        t1 = ets[-2]["time"]
        e0p1 = (e0-e1)/(t[0]-t1)
        ets[-1]["eps1"] = e0p1
        int_g = th.sum(gamma[int(t_next[0])+1:int(t[0])+1])
        noise = noise + e0p1 * ((t_next[0]-t[0])*g - int_g)
        print(((t[0],t_next[0])))
    if cur >= 3 and False:
        g2 = ets[-3]["gamma"]
        e1p1 = ets[-2]["eps1"]
        e0p2 = (e0p1-e1p1) / (g0-g2) 
        ets[-1]["eps2"] = e0p2
        noise = noise + e0p2 * (2 * (g - g0)**2  - 3 *(g - g0)*(g1 - g0)) / 6
    if cur >= 4 and False: #untested
        g3 = ets[-4]["gamma"]
        e1p2 = ets[-2]["eps2"]
        e0p3 = (e0p2-e1p2) / (g0-g3) 
        noise = noise + e0p3 * (g-g0) * (4*(g1-g)*(g2-g) + 2*(g1-g0)*(g2-g0) - (g-g0)**2) / 12

    #img_next = transfer(img, t, t_next, noise, alphas_cump)
    img_next = (x_bar +   noise )/sn
    return img_next

def correct_order_4(img, t, t_next, model, alphas_cump, ets):
    t_list = [t, (t+t_next)/2, t_next]
    if len(ets) > 2:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
    else:
        noise = correct_rk(img, t_list, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next
    
def correct_rk(x, t_list, model, alphas_cump, ets):
    at = alphas_cump[t_list[0].long() + 1].view(-1, 1, 1, 1)
    ad = alphas_cump[t_list[2].long() + 1].view(-1, 1, 1, 1)
    gt = th.sqrt((1-at)/at)
    gd = th.sqrt((1-ad)/ad)
    gm = (gt+gd)/2
    sqam = 1/th.sqrt(gm**2+1)
    dg = (gd-gt)
    x_bar = x/th.sqrt(at)
    # TODO fixed time

    e_1 = model(x, t_list[0])
    ets.append(e_1)
    x_2 = (x_bar + dg/2 * e_1) * sqam
    #x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(x_2, t_list[1])
    x_3 = (x_bar + dg/2 * e_2) * sqam
    #x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(x_3, t_list[1])
    x_4 = (x_bar + dg * e_3) * th.sqrt(ad)
    #x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(x_4, t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et

def runge_kutta(x, t_list, model, alphas_cump, ets):
    e_1 = model(x, t_list[0])
    ets.append(e_1)
    x_2 = transfer(x, t_list[0], t_list[1], e_1, alphas_cump)

    e_2 = model(x_2, t_list[1])
    x_3 = transfer(x, t_list[0], t_list[1], e_2, alphas_cump)

    e_3 = model(x_3, t_list[1])
    x_4 = transfer(x, t_list[0], t_list[2], e_3, alphas_cump)

    e_4 = model(x_4, t_list[2])
    et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)

    return et


def gen_order_2(img, t, t_next, model, alphas_cump, ets):
    if len(ets) > 0:
        noise_ = model(img, t)
        ets.append(noise_)
        noise = 0.5 * (3 * ets[-1] - ets[-2])
    else:
        noise = improved_eular(img, t, t_next, model, alphas_cump, ets)

    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def improved_eular(x, t, t_next, model, alphas_cump, ets):
    e_1 = model(x, t)
    ets.append(e_1)
    x_2 = transfer(x, t, t_next, e_1, alphas_cump)

    e_2 = model(x_2, t_next)
    et = (e_1 + e_2) / 2
    # x_next = transfer(x, t, t_next, et, alphas_cump)

    return et


def gen_order_1(img, t, t_next, model, alphas_cump, ets):
    noise = model(img, t)
    ets.append(noise)
    img_next = transfer(img, t, t_next, noise, alphas_cump)
    return img_next


def transfer(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long() + 1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long() + 1].view(-1, 1, 1, 1)

    x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - \
                                1 / (at.sqrt() * (((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

    x_next = x + x_delta
    return x_next


def transfer_dev(x, t, t_next, et, alphas_cump):
    at = alphas_cump[t.long()+1].view(-1, 1, 1, 1)
    at_next = alphas_cump[t_next.long()+1].view(-1, 1, 1, 1)

    x_start = th.sqrt(1.0 / at) * x - th.sqrt(1.0 / at - 1) * et
    x_start = x_start.clamp(-1.0, 1.0)

    x_next = x_start * th.sqrt(at_next) + th.sqrt(1 - at_next) * et

    return x_next
