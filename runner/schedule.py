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
import math
import torch as th
import torch.nn as nn
import numpy as np
from fastprogress.fastprogress import master_bar, progress_bar

import runner.method as mtd
import pdb


def get_schedule(args, config):
    if config['type'] == "quad":
        betas = (np.linspace(config['beta_start'] ** 0.5, config['beta_end'] ** 0.5, config['diffusion_step'], dtype=np.float64) ** 2)
    elif config['type'] == "linear":
        betas = np.linspace(config['beta_start'], config['beta_end'], config['diffusion_step'], dtype=np.float64)
    elif config['type'] == 'cosine':
        betas = betas_for_alpha_bar(config['diffusion_step'], lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2)
    else:
        betas = None

    betas = th.from_numpy(betas).float()
    alphas = 1.0 - betas
    alphas_cump = alphas.cumprod(dim=0)

    return betas, alphas_cump


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class Schedule(object):
    def __init__(self, args, config):
        device = th.device(args.device)
        betas, alphas_cump = get_schedule(args, config)
        self.device = device

        self.betas, self.alphas_cump = betas.to(device), alphas_cump.to(device)
        self.alphas_cump_pre = th.cat([th.ones(1).to(device), self.alphas_cump[:-1]], dim=0)
        self.total_step = config['diffusion_step']

        self.method = mtd.choose_method(args.method)  # add pflow
        self.method_name = args.method
        self.ets = None
    
    def update_spd(self, spd):
        self.gamma = th.sqrt((1-self.alphas_cump)/self.alphas_cump)
        gamma_rs = self.gamma[None,None,:,None]
        seq = th.linspace(1,-1, spd+1, device=self.device)#[1:] #* (len(self.gamma) - 1 )
        #seq = (1-(1-th.linspace(1,0, spd, device=self.device))**2)
        #seq = (th.linspace(1,0, spd, device=self.device)**2)*(1.6) -1
        #pdb.set_trace()
        grid = th.stack((th.ones_like(seq),seq),-1)[None,None]
        gamma_seq = th.nn.functional.grid_sample(gamma_rs,grid, align_corners=True)[0,0,0]
        self.gamma_seq = []
        for i in range(gamma_seq.shape[0]): 
            self.gamma_seq.append(gamma_seq[i])
        #self.gamma_seq[1] = self.gamma_seq[1] + 5
        #self.gamma_seq = [gamma_seq[i] for i in range(gamma_seq.shape[0])]

    def diffusion(self, img, t_end, t_start=0, noise=None):
        if noise is None:
            noise = th.randn_like(img)
        alpha = self.alphas_cump.index_select(0, t_end).view(-1, 1, 1, 1)
        img_n = img * alpha.sqrt() + noise * (1 - alpha).sqrt()

        return img_n, noise

    def denoising(self, img_n, t_end, t_start, model, first_step=False, pflow=False):
        if pflow:
            drift = self.method(img_n, t_start, t_end, model, self.betas, self.total_step)

            return drift
        else:
            if first_step:
                self.ets = []
            img_next = self.method(img_n, t_start, t_end, model, self.alphas_cump, self.ets)

            return img_next

    @th.no_grad()
    def run_optim(self, size, model):
        img_start = th.randn(size, device=self.device)
        predictor, corrector = mtd.choose_gamma_method(self.method_name)
        mb = master_bar(range(5))
        for i0 in mb:
            bar = progress_bar(range(len(self.gamma_seq)-1),parent=mb)
            ets, ets_used, ets_correct = [], [], []
            nnorm = []
            img_n = img_start
            for i in bar:
                g0 = self.gamma_seq[i]
                g_1 = self.gamma_seq[i+1]
                img_n, e_used = predictor(img_n, g0, g_1, model, self.gamma, ets)
                ets_used.append(e_used)
                if i > 0:
                    g0 = self.gamma_seq[i-1]
                    g_1 = self.gamma_seq[i]
                    e_correct = corrector(img_n, g0, g_1, model, self.gamma, ets)
                    ets_correct.append(e_correct)

            if True:
                g0 = self.gamma_seq[i+1]
                g_1 = g0 * 0
                img_n, e_used = predictor(img_n, g0, g_1, model, self.gamma, ets)
                e_correct = corrector(img_n, g0, g_1, model, self.gamma, ets)
                ets_correct.append(e_correct)
            diff = [th.norm(e0-e1) for e0, e1 in zip(ets_correct,ets_used)]

            g_diff = (self.gamma_seq[:-1] - self.gamma_seq[1:])
            diff = th.stack(diff) #/self.gamma_seq[:-1]
            mu = th.mean(diff)
            sd = th.std(diff)
            z = (diff-mu)/sd
            print(mu)
            #print(diff)
            #pdb.set_trace()
            #log_g = th.log(g_diff/sum(g_diff))
            #ratio = th.exp(log_g - 0.1*z)
            #pdb.set_trace()
            #ratio = g_diff - z
            #ratio = ratio/sum(ratio)
            ratio = g_diff/sum(g_diff)
            v = diff[1:] - diff[0]
            m = v/th.norm(v)
            new_ratio = (ratio[1:] - 1e-4*(m)) 
            new_ratio = th.cat([1-th.sum(new_ratio,0,keepdim=True),new_ratio])* self.gamma_seq[0]
            new_seq = th.sum(new_ratio) - th.cumsum(new_ratio,0) + self.gamma_seq[-1]
            #new_seq = (1 - th.cumsum(ratio,0)) * self.gamma_seq[0] 
            #pdb.set_trace()
            self.gamma_seq = th.cat([self.gamma_seq[0:10],self.gamma_seq[10:]*0.9],0)
            #print(self.gamma_seq[:-1] - self.gamma_seq[1:])
            #pdb.set_trace()
        return img_n

    @th.no_grad()
    def denoising_from_schedual(self, img_n, model):
        predictor, corrector = mtd.choose_line_method(self.method_name)
        bar = progress_bar(range(len(self.gamma_seq)-1))
        for i in bar:
            g0 = self.gamma_seq[i]
            g_1 = self.gamma_seq[i+1]
            img_n, _ = predictor(img_n, g0, g_1, model, self.gamma)

        return img_n

    @th.no_grad()
    def _line_search(self, size, model): #Adam method
        img_start = th.randn(size, device=self.device)
        predictor, corrector = mtd.choose_line_method(self.method_name)
        j = 0
        for i in range(1):
            img = img_start
            g0 = self.gamma_seq[0]
            g_1 = self.gamma_seq[1] +5
            g_2 = self.gamma_seq[2]
            x, eps0 = corrector(img, g0, g_1, model, self.gamma)
            x, eps1 = corrector(x, g_1, g_2, model, self.gamma)
            target = (g_1-g0) * eps0 + (g_2-g_1) * eps1
            with th.enable_grad():
                g_1 = g_1.detach().requires_grad_()
                opt = th.optim.Adam([g_1], lr=1.0)
                for j in range(250):
                    x, e0 = predictor(img, g0, g_1, model, self.gamma)
                    x, e1 = predictor(x, g_1, g_2, model, self.gamma)
                
                    current = (g_1-g0) * e0 + (g_2-g_1) * e1
                    loss = th.sum((current - target)**2)
                    opt.zero_grad(set_to_none=True)
                    loss.backward(retain_graph=False)
                    opt.step()
                    print(loss,g_1)

            self.gamma_seq[1] = g_1

    @th.no_grad()
    def _line_search(self, size, model): #golden section
        phi = 0.6180
        img_start = th.randn(size, device=self.device)
        predictor, corrector = mtd.choose_line_method(self.method_name)
        for i0 in range(3):
            for i in range(len(self.gamma_seq)-2):
                img = img_start
                g0 = self.gamma_seq[i]
                g_1 = self.gamma_seq[i+1]
                g_2 = self.gamma_seq[i+2]
                x, eps0 = corrector(img, g0, g_1, model, self.gamma)
                x, eps1 = corrector(x, g_1, g_2, model, self.gamma)
                target = (g_1-g0) * eps0 + (g_2-g_1) * eps1

                g_lower = g0
                g_upper = g_2

                def cal_obj(g, target=target, img=img):
                    x, e0 = predictor(img, g0, g, model, self.gamma)
                    x, e1 = predictor(x, g, g_2, model, self.gamma)
                    current = (g-g0) * e0 + (g_2-g) * e1
                    return th.sum((current - target)**2)

                hsize = g0 - g_2
                g_m1 = g_2 + phi*hsize
                mid1 = cal_obj(g_m1)

                g_m2 = g0 - phi*hsize
                mid2 = cal_obj(g_m2)

                for j in range(10):
                    if mid1.long() < mid2.long():
                        g_upper = g_m2
                        g_m2 = g_m1
                        mid2 = mid1
                        hsize = hsize * phi
                        g_m1 = g_upper + phi * hsize 
                        mid1 = cal_obj(g_m1)
                    else:
                        g_lower = g_m1
                        g_m1 = g_m2
                        mid1 = mid2
                        hsize = hsize * phi
                        g_m2 = g_lower - phi * hsize 
                        mid2 = cal_obj(g_m2)
                
                if mid1.long() < mid2.long():
                    self.gamma_seq[i+1] = g_m1
                else:
                    self.gamma_seq[i+1] = g_m2

    @th.no_grad()
    def _line_search(self, size, model): #quadratic section
        #img_start = th.randn(size, device=self.device)
        predictor, corrector = mtd.choose_line_method(self.method_name)

        def next_point(x1,x2,x3, f1,f2,f3):
            lhs = 2*((x2-x3)*f1 + (x3-x1)*f2 + (x1-x2)*f3)
            rhs = (x2**2-x3**2)*f1 + (x3**2-x1**2)*f2 + (x1**2-x2**2)*f3
            return rhs/lhs

        for i0 in range(5):
            img = th.randn(size, device=self.device)
            for i in range(len(self.gamma_seq)-2):
                #img = image_step
                g0 = self.gamma_seq[i]
                g_1 = self.gamma_seq[i+1]
                g_2 = self.gamma_seq[i+2]
                x, eps0 = corrector(img, g0, g_1, model, self.gamma)
                x, eps1 = corrector(x, g_1, g_2, model, self.gamma)
                target = (g_1-g0) * eps0 + (g_2-g_1) * eps1

                def cal_obj(g, target=target, img=img, g0=g0, g_2=g_2):
                    x, e0 = predictor(img, g0, g, model, self.gamma)
                    x, e1 = predictor(x, g, g_2, model, self.gamma)
                    current = (g-g0) * e0 + (g_2-g) * e1
                    return th.sum((current - target)**2), e0


                f_1, e0 = cal_obj(g_1)
                f_0 = th.sum(((g_2-g0) * e0 - target)**2)
                f_2 = f_0
                for i1 in range(6):
                    g = next_point(g0,g_1,g_2, f_0,f_1,f_2)
                    f, _ = cal_obj(g)
                    if g < g_1 and f < f_1:
                        f_0 = f_1
                        g0 = g_1
                        f_1 = f
                        g_1 = g
                    elif  g < g_1 and f > f_1:
                        f_2 = f
                        g_2 = g
                    elif g > g_1 and f < f_1:
                        f_2 = f_1
                        g_2 = g_1
                        f_1 = f
                        g_1 = g
                    else:
                        f_0 = f 
                        g0 = g 
                    print(g0,g_1,g_2)
                
                self.gamma_seq[i+1] = g_1
                x, eps0 = corrector(img, g0, g_1, model, self.gamma)
                x, eps1 = corrector(x, g_1, g_2, model, self.gamma)
                img = x

    @th.no_grad()
    def line_search(self, size, model): #quadratic section RKF 45
        #img_start = th.randn(size, device=self.device)
        predictor, corrector = mtd.choose_line_method(self.method_name)
        corrector2 = mtd.rkf45

        def next_point(x1,x2,x3, f1,f2,f3):
            lhs = 2*((x2-x3)*f1 + (x3-x1)*f2 + (x1-x2)*f3)
            rhs = (x2**2-x3**2)*f1 + (x3**2-x1**2)*f2 + (x1**2-x2**2)*f3
            return rhs/lhs

        for i0 in range(5):
            img = th.randn(size, device=self.device)
            for i in range(len(self.gamma_seq)-2):
                #img = image_step
                g0 = self.gamma_seq[i]
                g_1 = self.gamma_seq[i+1]
                g_2 = self.gamma_seq[i+2]
                #x, eps0 = corrector(img, g0, g_1, model, self.gamma)
                #x, eps1 = corrector(x, g_1, g_2, model, self.gamma)
                #target0 = (g_1-g0) * eps0 + (g_2-g_1) * eps1

                t_img1 = corrector2(img, g0, g_1, model, self.gamma) 
                t_img2 = corrector2(t_img1, g_1, g_2, model, self.gamma) 
                target = t_img2 * th.sqrt(g_2**2 + 1) - img * th.sqrt(g0**2 + 1)

                def cal_obj(g, target=target, img=img, g0=g0, g_2=g_2):
                    x, e0 = predictor(img, g0, g, model, self.gamma)
                    x, e1 = predictor(x, g, g_2, model, self.gamma)
                    current = (g-g0) * e0 + (g_2-g) * e1
                    return th.sum((current - target)**2), e0


                f_1, e0 = cal_obj(g_1)
                f_0 = th.sum(((g_2-g0) * e0 - target)**2)
                f_2 = f_0
                for i1 in range(6):
                    g = next_point(g0,g_1,g_2, f_0,f_1,f_2)
                    f, _ = cal_obj(g)
                    if g < g_1 and f < f_1:
                        f_0 = f_1
                        g0 = g_1
                        f_1 = f
                        g_1 = g
                    elif  g < g_1 and f > f_1:
                        f_2 = f
                        g_2 = g
                    elif g > g_1 and f < f_1:
                        f_2 = f_1
                        g_2 = g_1
                        f_1 = f
                        g_1 = g
                    else:
                        f_0 = f 
                        g0 = g 
                    print(g0,g_1,g_2)
                
                self.gamma_seq[i+1] = g_1
                img = corrector2(img, g0, g_1, model, self.gamma)
        
        return img

    @th.no_grad()
    def _line_search(self, size, model): #quadratic section Euler2+RK4
        eps = mtd.get_eps
        def next_point(x1,x2,x3, f1,f2,f3):
            lhs = 2*((x2-x3)*f1 + (x3-x1)*f2 + (x1-x2)*f3)
            rhs = (x2**2-x3**2)*f1 + (x3**2-x1**2)*f2 + (x1**2-x2**2)*f3
            return rhs/lhs

        for i0 in range(5):
            img = th.randn(size, device=self.device)
            for i in range(len(self.gamma_seq)-2):
                #img = image_step
                g0 = self.gamma_seq[i]
                g_1 = self.gamma_seq[i+1]
                g_2 = self.gamma_seq[i+2]

                def cal_obj(g, img=img, g0=g0, g_2=g_2):
                    s0 = th.sqrt(g0**2 + 1)
                    s1 = th.sqrt(g**2 + 1)
                    s2 = th.sqrt(g_2**2 + 1)
                    x_bar = img * s0 
                    # 2 steps Euler's method
                    e1 = eps(img, g0, model, self.gamma)
                    x1 = (x_bar +  (g - g0) * e1)/s1
                    e2 = eps(x1, g, model, self.gamma)
                    x2 = (x1*s1 +  (g_2 - g) * e2)/s2

                    # 1 step RK's method
                    e3 = eps((x_bar +  (g - g0) * e2)/s1, g, model, self.gamma)
                    x3 = (x_bar +  (g_2 - g0) * e2)/s2
                    e4 = eps(x3, g_2, model, self.gamma)
                    r = (g - g0)/(g_2 - g0)
                    eprime = ((1 - 1/(2*r) - 1/(24*r**2) + 1/(24*r**3)) * e1
                             +(1/(2*r) - 1/(6*r**2)) * e2
                             +(1/(6*r**2) - 1/(24*r**3)) * e3
                             +(1/(24*r**2)) * e4)
                    x4 = (x_bar +  (g_2 - g0) * eprime)/s2
                    #pdb.set_trace()
                    return th.sum((x2 - x4)**2), x4
                    #return th.sum(((g - g0) * e1 + (g_2 - g) * e2 - (g_2 - g0) * eprime)**2), x4

                #f, _ = cal_obj((g0+g_2)/2)

                f_1, xmin = cal_obj(g_1)
                g0 = g0*0.9 + g_1 * 0.1 
                g_2 = g_2*0.9 + g_1 * 0.1
                f_0, _ = cal_obj(g0)
                f_2, _ = cal_obj(g_2)
                #print(f_0,f_1,f_2)
                #pdb.set_trace()
                for i1 in range(8):
                    g = next_point(g0,g_1,g_2, f_0,f_1,f_2)
                    f, x = cal_obj(g)
                    if g < g_1 and f < f_1:
                        f_0 = f_1
                        g0 = g_1
                        f_1 = f
                        g_1 = g
                        xmin = x
                    elif  g < g_1 and f > f_1:
                        f_2 = f
                        g_2 = g
                    elif g > g_1 and f < f_1:
                        f_2 = f_1
                        g_2 = g_1
                        f_1 = f
                        g_1 = g
                        xmin = x
                    else:
                        f_0 = f 
                        g0 = g 
                    print(g0,g_1,g_2)
                    #print(f_0,f_1,f_2)
                    #pdb.set_trace()
                
                self.gamma_seq[i+1] = g_1
                #x, eps0 = corrector(img, g0, g_1, model, self.gamma)
                #x, eps1 = corrector(x, g_1, g_2, model, self.gamma)
                img = xmin

        return 0

    @th.no_grad()
    def schedule_search(self, noise, model): #schedule_search
        predictor, corrector = mtd.choose_line_method(self.method_name)
        corrector = mtd.rkf45

        if False:
            large_seq = th.linspace(1,-1, 250+1, device=self.device)[1:]
            gamma_rs = self.gamma[None,None,:,None]
            grid_large = th.stack((th.ones_like(large_seq),large_seq),-1)[None,None]
            gamma_seq = th.nn.functional.grid_sample(gamma_rs,grid_large, align_corners=True)[0,0,0]
            large_gamma_seq = []
            for i in range(gamma_seq.shape[0]): 
                large_gamma_seq.append(gamma_seq[i])

            bar = progress_bar(range(len(large_gamma_seq)-1))
            img_n = noise
            for i in bar:
                g0 = large_gamma_seq[i]
                g_1 = large_gamma_seq[i+1]
                img_n = corrector(img_n, g0, g_1, model, self.gamma)
            pdb.set_trace()
            th.save(img_n,f"pretrained/results/test/temp.pt")
        
        target = th.load(f"pretrained/results/test/temp.pt")
        seq1 = th.linspace(1,-1, len(self.gamma_seq), device=self.device)
        seq2 = (th.linspace(1,0, len(self.gamma_seq), device=self.device)**2)*(2) -1
        
        gamma_rs = self.gamma[None,None,:,None]
        N = 3
        min_mse = None
        mb = master_bar(range(2*N+1))
        for a in mb:
            ii = a/N - 1#- 0.5
            seq = (1-ii) * seq1 + (ii) * seq2
            grid = th.stack((th.ones_like(seq),seq),-1)[None,None]
            gamma_seq = th.nn.functional.grid_sample(gamma_rs,grid, align_corners=True)[0,0,0]
            
            bar = progress_bar(range(len(self.gamma_seq)-1), parent=mb)
            img_n = noise
            for i in bar:
                g0 = gamma_seq[i]
                g_1 = gamma_seq[i+1]
                img_n, _ = predictor(img_n, g0, g_1, model, self.gamma)

            mse = th.mean((img_n - target)**2)
            print(10 * th.log10(2/mse))
            if min_mse is None or mse<min_mse: 
                min_seq = gamma_seq
                min_mse = mse


        self.gamma_seq = []
        for i in range(min_seq.shape[0]): 
            self.gamma_seq.append(min_seq[i])
