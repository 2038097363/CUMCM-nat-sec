"""独立Q4研究候选。有限假设只用于决策，正反馈多边形与21站保证保持独立。"""
import math
from collections import defaultdict
import numpy as np
from optical_solver import OpticalV2
from geometry import minimum_enclosing_circle, nearest_clear_point, observation_candidates, unit
from solve import open_route


VARIANTS = {
    'base': {},
    'state_near': dict(selection='state_near'),
    'state_value': dict(selection='state_value'),
    'state_info': dict(selection='state_info'),
    'state_two': dict(selection='state_value', measures=2),
    'state_shared': dict(selection='state_value', reuse=2),
    'state_step': dict(selection='state_value', step=True),
    'state_step_shared': dict(selection='state_value', step=True, reuse=2),
    'short_two': dict(measures=2),
    'state_one': dict(selection='state_value', measures=1),
    'state_three': dict(selection='state_value', measures=3),
    'state_two_shared': dict(selection='state_value', measures=2, reuse=2),
    'state_near_two': dict(selection='state_near', measures=2),
    'state_info_two': dict(selection='state_info', measures=2),
    'state_two_dense': dict(selection='state_value', measures=2, dense=True),
    'state_two_low': dict(selection='state_value', measures=2, radius_low=True),
    'direct_optical': dict(measures=0),
    'dense_one': dict(selection='state_value', measures=1, dense=True),
    'dense_three': dict(selection='state_value', measures=3, dense=True),
    'dense_shared': dict(selection='state_value', measures=2, dense=True, reuse=2),
    'dense_information': dict(selection='state_info', measures=2, dense=True, mutual=True),
    'dense_expected': dict(selection='state_value', measures=2, dense=True, expected=True),
    'dense_adaptive': dict(selection='state_value', measures=5, dense=True, expected=True, adaptive=True),
}


def spatial_samples(poly,dense=False):
    """扇形三角剖分面积求积，无随机真值输入；不用于剪枝。"""
    pts, mass = [], []
    for b,c in zip(poly[1:-1],poly[2:]):
        a=poly[0];u=b-a;v=c-a;area=abs(u[0]*v[1]-u[1]*v[0])/2
        if area<1e-8:continue
        nodes=((1/6,1/6),(2/3,1/6),(1/6,2/3))
        if dense:
            nodes=[]
            for i in range(3):
                for j in range(3-i):
                    nodes.append(((i+1/3)/3,(j+1/3)/3))
                    if i+j<2:nodes.append(((i+2/3)/3,(j+2/3)/3))
        for u,v in nodes:
            p=a+u*(b-a)+v*(c-a)
            if np.linalg.norm(p)<=1800+1e-6:pts.append(p);mass.append(area/len(nodes))
    if not pts:return np.array([poly.mean(axis=0)]),np.ones(1)
    return np.array(pts),np.array(mass)


class ValueQ4(OpticalV2):
    def __init__(self,backend,question=4,variant='base',**kwargs):
        if question!=4:raise ValueError('本候选只处理第四问')
        super().__init__(backend,4,mode='reuse',scan_weight=.25,optical_mode='rectangle',**kwargs)
        self.options=dict(VARIANTS[variant]);self.variant=variant
        self.negative={k:[] for k in self.states};self.cache={}
        self.phase='other';self.phase_costs=defaultdict(lambda:defaultdict(float))
        self.local_measures=defaultdict(int);self.probes=defaultdict(int)
        self.extra_count=0;self.in_extra=False;self.decisions=[]

    def act(self,path,point=None,channel=None):
        before=dict(self.costs)
        out=super().act(path,point,channel)
        if point is not None:
            for k,v in self.costs.items():self.phase_costs[self.phase][k]+=v-before[k]
            self.history[-1]['phase']=self.phase
        return out

    def measure(self,point,k,station_id=None):
        result=super().measure(point,k,station_id)
        if result=='no_signal':self.negative[k].append(np.asarray(point).copy())
        self.cache.pop(k,None)
        return result

    def clear(self,point,k):
        result=super().clear(point,k);self.cache.pop(k,None);return result

    def scan(self,point):
        old=self.phase;self.phase='search'
        try:return super().scan(point)
        finally:self.phase=old

    def optical_points(self,*args,**kwargs):
        self.phase='optical';return super().optical_points(*args,**kwargs)

    def hypotheses(self,k):
        if k in self.cache:return self.cache[k]
        b=self.states[k];points,area=spatial_samples(b.poly,self.options.get('dense',False))
        positive=np.asarray([p for p,_ in b.observations]);negative=np.asarray(self.negative[k])
        directions=np.vstack(([0.,0.],np.c_[np.cos(np.arange(32)*np.pi/16),np.sin(np.arange(32)*np.pi/16)]))
        loc=[];heads=[];rads=[];weights=[]
        for p,a in zip(points,area):
            if any(np.linalg.norm(p-f)<=20 for f in b.failed_clears):continue
            lo=max(1000.,np.max(np.linalg.norm(positive-p,axis=1)))
            if lo>1500:continue
            ok=np.ones(33,dtype=bool)
            ok[1:]=np.all((positive-p)@directions[1:].T>=-1e-7,axis=0)
            hi=np.full(33,1500.)
            if len(negative):
                dist=np.linalg.norm(negative-p,axis=1)
                front=(negative-p)@directions.T>=-1e-7
                hi=np.minimum(hi,np.min(np.where(front,dist[:,None],np.inf),axis=0))
            for j in np.flatnonzero(ok & (hi>lo+1e-8)):
                # 全向先验一半，定向先验一半；不是类型判定。
                w=a*(hi[j]-lo)*(0.5 if j==0 else 0.5/32)
                fractions=(.000001,) if self.options.get('radius_low') else (.25,.75)
                for f in fractions:
                    loc.append(p);heads.append(directions[j]);rads.append(lo+f*(hi[j]-lo));weights.append(w/len(fractions))
        if not weights:
            c,_=minimum_enclosing_circle(b.poly)
            result=(np.array([c]),np.array([[0.,0.]]),np.array([1500.]),np.array([1.]))
        else:
            weights=np.array(weights);result=(np.array(loc),np.array(heads),np.array(rads),weights/weights.sum())
        self.cache[k]=result;return result

    def information(self,k,point):
        g,d,r,w=self.hypotheses(k);v=point-g;dist=np.linalg.norm(v,axis=1)
        sig=(dist<=r)&(np.sum(v*d,axis=1)>=-1e-7)
        angles=np.degrees(np.arctan2(-v[:,1],-v[:,0]))%360
        # 方位分箱只作为规划近似；三个误差支撑点抑制网格过度乐观。
        ids=np.concatenate([np.where(sig,np.where(dist<=5,181,np.floor(((angles+e)%360)/2.5).astype(int)),180) for e in (-1.,0.,1.)])
        ww=np.tile(w/3,3);gg=np.tile(g,(3,1));mass=np.bincount(ids,weights=ww,minlength=182)
        means=np.zeros((182,2))
        for j in range(2):means[:,j]=np.bincount(ids,weights=ww*gg[:,j],minlength=182)/np.maximum(mass,1e-30)
        residual=float(np.sum(ww*np.linalg.norm(gg-means[ids],axis=1)))
        future=float(np.sum(mass*np.linalg.norm(means-point,axis=1)))
        nz=mass[mass>0];entropy=float(-np.sum(nz*np.log(nz)))
        if self.options.get('mutual'):
            outcomes=ids.reshape(3,-1)
            same=(outcomes[0]==outcomes[1])&(outcomes[1]==outcomes[2])
            distinct=(outcomes[0]!=outcomes[1])&(outcomes[1]!=outcomes[2])&(outcomes[0]!=outcomes[2])
            conditional=np.where(same,0.,np.where(distinct,np.log(3),np.log(3)-2/3*np.log(2)))
            entropy=max(0.,entropy-float(w@conditional))
        return residual,future,entropy,float(w[sig].sum())

    def optical_route(self,k,planning=False):
        b=self.states[k];first,a=b.observations[0];old=self.phase
        points=self.optical_points(b.poly,first,a,b.failed_clears)
        if planning:self.phase=old
        routes=(points,list(reversed(points)),open_route(points,self.position))
        def length(route):return sum(np.linalg.norm(y-x) for x,y in zip([self.position]+list(route),route))
        best=min(routes,key=length)
        if not self.options.get('expected'):return best,None
        g,d,r,w=self.hypotheses(k)
        def expected(route):
            remaining=np.ones(len(w),dtype=bool);value=2.;prev=self.position
            for p in route:
                value+=float(w[remaining].sum())*(np.linalg.norm(p-prev)/5+3)
                remaining&=np.linalg.norm(g-p,axis=1)>20
                prev=p
            return float(value)
        allowed=[route for route in routes if length(route)<=1.2*length(best)+1e-6]
        route=min(allowed,key=expected)
        return route,expected(route)

    def select(self,k):
        b=self.states[k];g,d,r,w=self.hypotheses(k);estimate=np.sum(w[:,None]*g,axis=0)
        old=observation_candidates(b.poly,self.position,b.observations,4)
        last,angle=b.observations[-1];u=unit(angle);v=np.array([-u[1],u[0]])
        length=max(20.,float((estimate-last)@u))
        choices=old+[last+fraction*length*u+side*v for fraction in (.6,.9,1.05) for side in (-90.,-35.,35.,90.)]
        choices=[p for p in choices if all(np.linalg.norm(p-a)>2 for a in b.attempted)]
        best=None
        for p in choices:
            move=np.linalg.norm(p-self.position)/5;res,future,ent,prob=self.information(k,p)
            if self.options.get('selection')=='state_near':score=move+5+future/5+.15*res
            elif self.options.get('selection')=='state_info':score=-(ent+.01*max(0,200-res))/(move+6+future/5)
            else:score=move+6+future/5+.4*res
            if best is None or score<best[0]:best=(score,p,res,prob)
        return best,estimate

    def reuse(self,primary):
        limit=self.options.get('reuse',0)
        if not limit or self.in_extra or self.extra_count>=48:return
        self.in_extra=True;old=self.phase;self.phase='shared'
        try:
            offers=[]
            for k,b in self.states.items():
                if k==primary or b.status!='found' or len(b.observations)>=6:continue
                if any(np.linalg.norm(self.position-p)<50 for p in b.attempted):continue
                c,r=minimum_enclosing_circle(b.poly)
                if r<=self.config.clear_margin or np.linalg.norm(c-self.position)-r>1500:continue
                res,future,ent,prob=self.information(k,self.position)
                g,d,rr,w=self.hypotheses(k);prior=float(np.sum(w*np.linalg.norm(g-np.sum(w[:,None]*g,axis=0),axis=1)))
                gain=prior-res
                if gain>30 and prob>.35:offers.append((gain,k))
            for _,k in sorted(offers,reverse=True)[:limit]:
                self.extra_count+=1;self.measure(self.position.copy(),k)
        finally:self.in_extra=False;self.phase=old

    def localize_and_clear(self,k):
        old=self.phase;self.phase='localize'
        try:
            if not self.options:return super().localize_and_clear(k)
            if 'selection' not in self.options:
                oldlimit=self.config.max_measure_attempts;self.config.max_measure_attempts=self.options.get('measures',5)
                try:return super().localize_and_clear(k)
                finally:self.config.max_measure_attempts=oldlimit
            b=self.states[k];limit=self.options.get('measures',5)
            while b.status=='found' and self.local_measures[k]<limit:
                center,radius=minimum_enclosing_circle(b.poly)
                if radius<=self.config.clear_margin:
                    q=nearest_clear_point(b.poly,self.position,self.config.clear_margin)
                    if not self.clear(q,k):raise RuntimeError('保证清除失败')
                    self.reuse(k);return
                choice,estimate=self.select(k)
                if self.options.get('adaptive') and choice is not None:
                    _,optcost=self.optical_route(k,planning=True)
                    if optcost is not None and optcost<=choice[0]:break
                g,d,r,w=self.hypotheses(k)
                chance=float(w[np.linalg.norm(g-estimate,axis=1)<=20].sum())
                if self.probes[k]<2 and (radius<=100 or chance>=.45) and all(np.linalg.norm(estimate-p)>3 for p in b.failed_clears):
                    self.probes[k]+=1
                    if self.clear(estimate,k):self.reuse(k);return
                    choice,estimate=self.select(k)
                if choice is None:break
                self.local_measures[k]+=1
                self.decisions.append(dict(channel=k,score=float(choice[0]),signal_weight=float(choice[3]),radius=float(radius)))
                self.measure(choice[1],k);self.reuse(k)
                if self.options.get('step'):return
            if b.status!='found':return
            center,radius=minimum_enclosing_circle(b.poly)
            if radius<=self.config.clear_margin:
                if not self.clear(nearest_clear_point(b.poly,self.position,self.config.clear_margin),k):raise RuntimeError('保证清除失败')
                return
            route,_=self.optical_route(k)
            for point in route:
                if self.clear(point,k):self.reuse(k);return
            raise RuntimeError('光学覆盖结束仍未清除')
        finally:self.phase=old

    def run(self):
        r=super().run();r.update(variant=self.variant,phase_costs={k:dict(v) for k,v in self.phase_costs.items()},extra_measures=self.extra_count)
        return r
