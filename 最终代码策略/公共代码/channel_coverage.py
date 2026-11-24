"""全向未发现频道的覆盖账本。近似面积只排序；结束/删站由连续核验决定。"""
import math
import numpy as np
from adaptive_solver import disk_coverage_radius


class ChannelCoverage:
    def __init__(self,channels=range(1,21),radius=999.99):
        self.negative={k:[] for k in channels};self.radius=radius;self.cache={}
        a=np.arange(96)*2*math.pi/96;r=1800*np.sqrt((np.arange(24)+.5)/24)
        self.quadrature=(r[:,None,None]*np.c_[np.cos(a),np.sin(a)][None,:,:]).reshape(-1,2)
        self.remaining={k:np.ones(len(self.quadrature),dtype=bool) for k in channels}

    def add_negative(self,k,point):
        p=np.asarray(point,dtype=float).copy()
        if any(np.linalg.norm(p-q)<1e-7 for q in self.negative[k]):return
        self.negative[k].append(p)
        self.remaining[k]&=np.linalg.norm(self.quadrature-p,axis=1)>self.radius

    def certificate(self,k,future=(),extra=()):
        points=self.negative[k]+list(future)+list(extra)
        if not points:return False,math.inf,None
        key=tuple(sorted(set((float(p[0]),float(p[1])) for p in points)))
        if key not in self.cache:
            value,witness=disk_coverage_radius(np.array(key))
            self.cache[key]=(value,witness.tolist())
        value,witness=self.cache[key]
        return value<=self.radius,value,witness

    def estimated_gain(self,k,point):
        mask=self.remaining[k]
        if not mask.any():return 0.
        return float(np.mean(np.linalg.norm(self.quadrature[mask]-point,axis=1)<=self.radius))
