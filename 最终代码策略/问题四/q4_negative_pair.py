"""连续几何收缩：正站与成对阴性站的遮挡锥。只复用已有观测。"""
from itertools import combinations
import numpy as np
from geometry import clip,minimum_enclosing_circle
from q4_scan_spine import SpineQ4


def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def negative_pair_cut(poly,anchor,p,q):
    edge=q-p;length=float(np.linalg.norm(edge))
    if length<1e-4:return None
    normal=np.array([-edge[1],edge[0]])/length;bound=float(normal@p)
    if normal@anchor>bound:normal=-normal;bound=-bound
    if bound-normal@anchor<1e-4:return None
    if float((poly@normal-bound).max())<1e-3:return None
    far=clip(poly,-normal,-bound)
    if not len(far):return None
    v,w=p-anchor,q-anchor
    if cross(v,w)<0:v,w=w,v
    if cross(v,w)<1e-5:return None
    rel=far-anchor
    if np.min(cross(v,rel))< -1e-8 or np.min(cross(rel,w))< -1e-8:return None
    distance=max(float(np.linalg.norm(far-p,axis=1).max()),float(np.linalg.norm(far-q,axis=1).max()))
    if distance>999.9:return None
    new=clip(poly,normal,bound+1e-6)
    if not len(new):raise RuntimeError('正站与成对阴性站约束矛盾')
    return new,dict(anchor=anchor.tolist(),negative_pair=[p.tolist(),q.tolist()],normal=normal.tolist(),bound=bound,max_distance=distance)


class NegativePairQ4(SpineQ4):
    def __init__(self,backend,question=4,variant='negative_base',**kwargs):
        super().__init__(backend,question,variant='spine300' if variant=='negative_spine' else 'baseline_listen8',**kwargs)
        self.variant=variant;self.pair_events=[]
        if variant=='negative_geometry':self.options={}
        if variant=='negative_short':self.options=dict(measures=2)

    def measure(self,point,k,station_id=None):
        response=super().measure(point,k,station_id);b=self.states[k]
        if b.status!='found' or len(self.negative[k])<2:return response
        # 每次收缩仍保留原正观测多边形外包；有限遍历不依赖离散假设权重。
        for anchor,bearing in b.observations:
            for p,q in combinations(self.negative[k],2):
                result=negative_pair_cut(b.poly,anchor,p,q)
                if result is None:continue
                new,proof=result;before=minimum_enclosing_circle(b.poly)[1]
                b.poly=new;self.cache.pop(k,None)
                if self.audit_callback:self.audit_callback(k,b.poly)
                event=dict(channel=k,radius_before=before,radius_after=minimum_enclosing_circle(new)[1],proof=proof)
                self.pair_events.append(event);self.record(dict(negative_pair=event,polygon=new.tolist()))
        return response

    def run(self):
        r=super().run();r['pair_events']=self.pair_events;return r
