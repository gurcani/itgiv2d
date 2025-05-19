#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:35:36 2025

@author: ogurcan
"""

import numpy as np
import cupy as xp
import gc
import sys
import os
from mlsarray.mlsarray import mlsarray,slicelist,init_kspace_grid,rfft2
from mlsarray.gensolver import gensolver,save_data
import h5py as h5

filename='out.h5'
Npx,Npy=1024,1024
t0,t1=0.0,300.0
wecontinue=False
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=100,100
dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]

lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
ksqr=kx**2+ky**2
Nk=kx.size
rtol,atol=1e-9,1e-12

w=10.0
Phik=1e-6*xp.exp(-lkx**2/2/w**2-lky**2/2/w**2)*xp.exp(1j*2*np.pi*xp.random.rand(lkx.size).reshape(lkx.shape))
Tk=1e-6*xp.exp(-lkx**2/2/w**2-lky**2/2/w**2)*xp.exp(1j*2*np.pi*xp.random.rand(lkx.size).reshape(lkx.shape))
Phik[slbar]=0
Tk[slbar]=0
zk=np.hstack((Phik,Tk))
del lkx,lky; gc.collect()

chi=0.1
a=9.0/40.0
b=67.0/160.0
kapt=3.5
nuH,nuL=5e-4,0.0

if(wecontinue):
    fl=h5.File(filename,'r+',libver='latest')
    fl.swmr_mode = True
    zk=fl['last/zk'][()]
    t=fl['last/t'][()]
else:
    if os.path.exists(filename):
        os.remove(filename)
    fl=h5.File(filename,'w',libver='latest')
    fl.swmr_mode = True
    t=t0
    save_data(fl,'data',ext_flag=False,kx=kx.get(),ky=ky.get())
    save_data(fl,'params',ext_flag=False,kapt=kapt,chi=chi,a=a,b=b,Lx=Lx,Ly=Ly)
    save_data(fl,'last',ext_flag=False,zk=zk.get(),t=t)

def irft(uk):
    utmp=mlsarray(Npx,Npy)
    utmp[sl]=uk
    utmp[-1:-int(Nx/2):-1,0]=utmp[1:int(Nx/2),0].conj()
    utmp.irfft2()
    return utmp.view(dtype=float)[:,:-2]

def rft(u):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return xp.hstack(uk[sl])

def save_last(t,y):
    zk=y.view(dtype=complex)
    save_data(fl,'last',ext_flag=False,zk=zk.get(),t=t)

def save_real_fields(t,y):
    zk=y.view(dtype=complex)
    Phik=zk[0:Nk]
    Om=irft(-ksqr*Phik)
    Tk=zk[Nk:]
    T=irft(Tk)
    save_data(fl,'fields',ext_flag=True,Om=Om.get(),T=T.get(),t=t)

def save_fluxes(t,y):
    zk=y.view(dtype=complex)
    Phik=zk[0:Nk]
    Tk=zk[Nk:]
    dyphi=irft(1j*ky*Phik)
    Om=irft(-ksqr*Phik)
    T=irft(Tk)
    Q=np.mean(-dyphi*T,1)
    Pi=np.mean(-dyphi*Om,1)
    save_data(fl,'fluxes',ext_flag=True,Q=Q.get(),Pi=Pi.get(),t=t)

def save_zonal(t,y):
    zk=y.view(dtype=complex)
    Phik=zk[0:Nk]
    Tk=zk[Nk:]
    vy=irft(-1j*kx*Phik)
    Om=irft(-ksqr*Phik)
    T=irft(Tk)
    save_data(fl,'fields/zonal/',ext_flag=True,vbar=xp.mean(vy,1).get(),ombar=xp.mean(Om,1).get(),Tbar=xp.mean(T,1).get(),t=t)

def fshow(t,y):
    zk=y.view(dtype=complex)
    Phik=zk[0:Nk]
    Tk=zk[Nk:]
    dyphi=irft(1j*ky*Phik)
    T=irft(Tk)
    Q=np.mean(-dyphi*T)
    print('Q=',Q.get())

def rhs(t,y):
    zk=y.view(dtype=complex)
    dzkdt=xp.zeros_like(zk)
    Phik,Tk=zk[:Nk],zk[Nk:]
    dPhikdt,dTkdt=dzkdt[:Nk],dzkdt[Nk:]
    
    dxphi=irft(1j*kx*Phik)
    dyphi=irft(1j*ky*Phik)
    dxT=irft(1j*kx*Tk)
    dyT=irft(1j*ky*Tk)
    sigk=xp.sign(ky)
    W=irft((sigk+ksqr)*Phik)
    
    dPhikdt[:]=1j*ky*((1+kapt*ksqr)*Phik+Tk)-chi*ksqr**2*(a*Phik-b*Tk)*sigk
    dPhikdt[:]+=(1j*kx*rft(dyphi*W)-1j*ky*rft(dxphi*W))
#    dPhikdt[:]+= kx**2*rft(dxphi*dyT)-ky**2*rft(dyphi*dxT)+kx*ky*rft(dyphi*dyT-dxphi*dxT)
    dPhikdt[:]/=(sigk+ksqr)
    dPhikdt[:]+=-sigk*(nuH*ksqr**2*Phik+nuL/ksqr**2*Phik)

    dTkdt[:]=-1j*ky*kapt*Phik-chi*ksqr*Tk*sigk
    dTkdt[:]+=rft(dyphi*dxT-dxphi*dyT)
    dTkdt[:]+=-sigk*(nuH*ksqr**2*Tk+nuL/ksqr**2*Tk)
    
    return dzkdt.view(dtype=float)

fsave=[save_last, save_fluxes, save_zonal, save_real_fields]
dtsave=[0.1,1.0,1.0,1.0]
dtstep,dtshow=0.1,0.1
r=gensolver('cupy_ivp.DOP853',rhs,t,zk.view(dtype=float),t1,fsave=fsave,fshow=fshow,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,rtol=rtol,atol=atol)
r.run()
fl.close()
