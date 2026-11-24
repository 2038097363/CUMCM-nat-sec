"""问题四可见性证书驱动的状态价值调度算法（VC-SVS）。"""
import numpy as np
from q4_blind_safe import BlindSafeQ4
from q4_soft_dispatch import SoftQ4
from q4_task_value import ValueQ4
from geometry import nearest_clear_point
from scipy.optimize import minimize
from geometry import minimum_enclosing_circle,observation_candidates

MODES={
    'no_probe':dict(probe=False),
    'local_risk':dict(local='risk',probe=True),
    'risk_no_probe':dict(local='risk',probe=False),
    'r4_after_miss':dict(local='r4_after_miss',probe=True),
    'r4_local':dict(local='r4_local',probe=True),
    'compact_guard':dict(compact_guard=True,probe=True),
    'exit_clear':dict(exit_clear=True,probe=True),
    'compact_exit':dict(compact_guard=True,exit_clear=True,probe=True),
    'lookahead':dict(local='lookahead',probe=True),
    'lookahead_no_probe':dict(local='lookahead',probe=False),
}


class VCSVQ4(BlindSafeQ4):
    def __init__(self,backend,question=4,variant='local_risk',**kwargs):
        if variant not in MODES:raise ValueError(variant)
        super().__init__(backend,question,**kwargs);self.variant=variant;self.r12=MODES[variant]
        self.local_events=[];self.last_local_miss=False;self.local_channel=None
        self.search_guard=False;self.exit_anchor=None;self.guaranteed_target=None

    def classify_origin(self):
        super().classify_origin()
        self.search_guard=bool(self.r12.get('compact_guard') and self.compact_search)

    def choose(self,tasks,todo,pending):
        self.exit_anchor=None
        if self.search_guard:
            if todo:return self.baseline_choice(todo,pending)
            self.search_guard=False
        if self.r12.get('exit_clear') and todo:
            from solve import open_route
            self.exit_anchor=open_route(todo,self.position)[0].copy()
        if not self.r12['probe']:return SoftQ4.choose(self,tasks,todo,pending)
        return super().choose(tasks,todo,pending)

    def measure(self,point,k,station_id=None):
        result=ValueQ4.measure(self,point,k,station_id) if self.search_guard else super().measure(point,k,station_id)
        if self.local_channel==k:self.last_local_miss=result=='no_signal'
        return result

    def localize_and_clear(self,k):
        self.local_channel=k;self.last_local_miss=False
        old=dict(self.options)
        try:
            # 独立消融：保留完整几何与调度，仅对局部补测次数作受限消融。
            if self.r12.get('local')=='r4_local' or self.search_guard:self.options={}
            return super().localize_and_clear(k)
        finally:self.options=old;self.local_channel=None;self.last_local_miss=False

    def clear(self,point,k):
        # 对连续可行域全部顶点20米约束下，优化当前点→清除点→预计下一扫描站。
        # 只在原调用已是保证清除时改变落点；启发式下一站不作全任务最优保证。
        if self.r12.get('exit_clear') and self.exit_anchor is not None and not self.search_guard:
            poly=self.states[k].poly
            if len(poly) and np.max(np.linalg.norm(poly-point,axis=1))<=19.90001:
                start=self.position.copy();anchor=self.exit_anchor.copy()
                def cost(p):return float(np.linalg.norm(p-start)+np.linalg.norm(p-anchor))
                opt=minimize(cost,np.asarray(point),method='SLSQP',constraints=[dict(type='ineq',fun=lambda p:19.9**2-np.sum((poly-p)**2,axis=1))],options=dict(maxiter=60,ftol=1e-7))
                if opt.success and np.max(np.linalg.norm(poly-opt.x,axis=1))<=19.900001 and cost(opt.x)<cost(point)-.01:
                    event=dict(channel=k,mode='exit_clear',old_point=np.asarray(point).tolist(),new_point=opt.x.tolist(),predicted_path_saving_s=(cost(point)-cost(opt.x))/5)
                    self.local_events.append(event);self.record(dict(vcsv_local=event));point=opt.x
        return super().clear(point,k)

    def select(self,k):
        original,estimate=super().select(k);mode=self.r12.get('local')
        if original is not None and mode=='lookahead':return self.lookahead_select(k,original,estimate)
        if original is None or mode not in ('risk','r4_after_miss'):return original,estimate
        b=self.states[k]
        old=[p for p in observation_candidates(b.poly,self.position,b.observations,4)
             if all(np.linalg.norm(p-a)>2 for a in b.attempted)]
        if not old:return original,estimate
        nearest=min(old,key=lambda p:np.linalg.norm(p-self.position))
        if mode=='r4_after_miss':
            if not self.last_local_miss:return original,estimate
            res,future,ent,prob=self.information(k,nearest)
            chosen=(float(np.linalg.norm(nearest-self.position)/5+6+future/5+.4*res),nearest,res,prob)
        else:
            # 负反馈未必能收缩连续域，因此不把离散负反馈分支的置信度当作几何收益。
            # 用一次横跨当前不确定区域的时间作为无信号后重选位置的成本代理。
            _,radius=minimum_enclosing_circle(b.poly);retry_cost=min(1000.,2*radius)/5
            candidates=[original[1]]+old
            scored=[]
            for p in candidates:
                res,future,ent,prob=self.information(k,p)
                score=float(np.linalg.norm(p-self.position)/5+6+future/5+.4*res+(1-prob)*retry_cost)
                scored.append((score,p,res,prob))
            chosen=min(scored,key=lambda c:c[0])
        if np.linalg.norm(chosen[1]-original[1])>1e-6:
            event=dict(channel=k,mode=mode,after_miss=self.last_local_miss,
                old_point=original[1].tolist(),new_point=chosen[1].tolist(),old_probability=original[3],new_probability=chosen[3])
            self.local_events.append(event);self.record(dict(vcsv_local=event))
        return chosen,estimate

    def lookahead_select(self,k,original,estimate):
        b=self.states[k];g,d,r,w=self.hypotheses(k)
        old=observation_candidates(b.poly,self.position,b.observations,4)
        last,angle=b.observations[-1]
        u=np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle))]);v=np.array([-u[1],u[0]])
        candidates=[original[1]]+old+[estimate+forward*u+side*v for forward in (-60.,0.) for side in (-60.,-25.,25.,60.)]
        candidates=[p for p in candidates if all(np.linalg.norm(p-a)>2 for a in b.attempted)]
        # 用三个位于误差界内的支撑点评估一次测量后的离散分支；仅用于规划。
        gg=np.tile(g,(3,1));ww=np.tile(w/3,3);prepared=[]
        for p in candidates:
            delta=p-g;dist=np.linalg.norm(delta,axis=1)
            signal=(dist<=r)&(np.sum(delta*d,axis=1)>=-1e-7)
            bearing=np.degrees(np.arctan2(-delta[:,1],-delta[:,0]))%360
            ids=np.concatenate([np.where(signal,np.where(dist<=5,181,np.floor(((bearing+err)%360)/2.).astype(int)),180) for err in (-1.,0.,1.)])
            prepared.append((p,ids,float(w[signal].sum())))
        def terminal(position,weights):
            mass=float(weights.sum())
            if mass<=1e-12:return 0.
            center=weights@gg/mass
            residual=np.linalg.norm(gg-center,axis=1)
            # 有限候选上的两步终端费用：到代表点、成功清除及剩余位置不确定性的搜索代理。
            return float(mass*(np.linalg.norm(position-center)/5+5)+weights@np.maximum(0.,residual-20)*.4)
        initial=terminal(self.position,ww)
        # 先选单步费用最小的4个候选，第二步只在这些候选及分支代表点内比较。
        ranked=[]
        for p,ids,prob in prepared:
            score=np.linalg.norm(p-self.position)/5+6
            for label in np.unique(ids):score+=terminal(p,ww*(ids==label))
            ranked.append((float(score),p,ids,prob))
        short=sorted(ranked,key=lambda x:x[0])[:4];best=None
        for _,p,ids,prob in ranked:
            score=float(np.linalg.norm(p-self.position)/5+6)
            for label in np.unique(ids):
                weights=ww*(ids==label);mass=float(weights.sum())
                stop=terminal(p,weights);continuations=[stop]
                # 两次固定误差支撑仅做积分近似，不意味着重复测量获得独立噪声。
                for _,q,jds,_ in short:
                    if np.linalg.norm(q-p)<=2:continue
                    cost=mass*(np.linalg.norm(q-p)/5+6)
                    for next_label in np.unique(jds[weights>0]):cost+=terminal(q,weights*(jds==next_label))
                    continuations.append(cost)
                score+=min(continuations)
            if best is None or score<best[0]:best=(score,p,0.,prob)
        if np.linalg.norm(best[1]-original[1])>1e-6:
            event=dict(channel=k,mode='lookahead',old_point=original[1].tolist(),new_point=best[1].tolist(),score=best[0],terminal_proxy=initial)
            self.local_events.append(event);self.record(dict(vcsv_local=event))
        return best,estimate

    def run(self):
        r=super().run();r['vcsv_local_events']=self.local_events;return r
