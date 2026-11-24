"""第十轮候选：动态连续调度评分，兼顾发现上限与沿途清除。"""
import time
import numpy as np
from q4_portfolio import PortfolioQ4
from q4_scan_spine import insert_services
from q4_route_joint import tour_order
from adaptive_solver import reduced_stations
from solve import open_route
from geometry import minimum_enclosing_circle,observation_candidates
from q4_value_audit import coverage_audit


MODES={'soft_geo':(.5,.85,False),'soft_cap':(.5,.85,True),'soft_wide':(.35,.8,True)}


def smoothstep(x):
    x=float(np.clip(x,0.,1.));return x*x*(3.-2.*x)


class SoftQ4(PortfolioQ4):
    def __init__(self,backend,question=4,variant='soft_cap',**kwargs):
        super().__init__(backend,question,variant='portfolio90',**kwargs)
        self.variant=variant;self.soft_options=MODES[variant];self.decision_events=[]

    def caution(self,todo):
        if not todo:return 0.,0.,0
        n=sum(b.status in ('found','cleared') for b in self.states.values())
        vectors=[];weights=[]
        for b in self.states.values():
            if b.status not in ('found','cleared') or not b.observations:continue
            c,r=minimum_enclosing_circle(b.poly);distance=float(np.linalg.norm(c))
            if distance<1e-6:continue
            vectors.append(c/distance);weights.append(1./(1.+r/300.))
        resultant=float(np.linalg.norm(np.average(vectors,axis=0,weights=weights))) if vectors else 0.
        low,high,cap_enabled=self.soft_options
        concentration=smoothstep((resultant-low)/(high-low))*min(1.,len(vectors)/4.)
        cap=(n/16.)**8 if cap_enabled else 0.
        weight=1.-(1.-concentration)*(1.-cap)
        return weight,resultant,n

    def immediate(self,task):
        kind,index,point=task
        if kind=='scan':
            unknown=sum(b.status=='undiscovered' for b in self.states.values())
            return (np.linalg.norm(point-self.position)/5+6*unknown)*.25
        b=self.states[index];c,r=minimum_enclosing_circle(b.poly)
        points=observation_candidates(b.poly,self.position,b.observations,4)
        distance=min([np.linalg.norm(p-self.position) for p in points]+[np.linalg.norm(c-self.position)])
        return distance/5+5

    def route_cost(self,first,tasks):
        kind,k,p=first;kind='target' if kind=='localize' else kind
        remaining=[t for t in tasks if (t[0],t[1])!=(kind,k)]
        points=np.vstack([self.position,p]+[t[2] for t in remaining])
        return float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()/5)

    def choose(self,tasks,todo,pending):
        baseline=self.baseline_choice(todo,pending);proposal=tasks[0]
        a,resultant,n=self.caution(todo)
        choices=[baseline,proposal]
        travel=[self.route_cost(t,tasks) for t in choices];best_travel=min(travel)
        scores=[a*self.immediate(t)+(1-a)*(d-best_travel) for t,d in zip(choices,travel)]
        index=min(range(2),key=lambda i:(scores[i],i))
        event=dict(time_s=self.virtual_time,caution_weight=a,resultant=resultant,discovered=n,remaining_stations=len(todo),
            baseline=[baseline[0],baseline[1]],proposal=[proposal[0],proposal[1]],scores=scores,route_costs=travel,chosen=index)
        self.decision_events.append(event);self.record(dict(soft_decision=event))
        return choices[index]

    def run(self):
        if not self.spine:
            r=super().run();r['listen_events']=self.listen_events
            r['coverage_audit']=coverage_audit(self)
            if r['status']=='completed' and not r['coverage_audit']['valid']:r['status']='failed: 实际覆盖未通过'
            return r
        todo=list(reduced_stations(4)[1:]);entered=False;status='completed'
        try:
            response=self.act('/enter');entered=True
            self.deadline=time.monotonic()+response.get('remaining_real_duration_s',1200)
            self.scan(np.zeros(2));self.classify_origin()
            for step in range(101):
                pending=[k for k,b in self.states.items() if b.status=='found']
                if sum(b.status in ('found','cleared') for b in self.states.values())==16:todo=[]
                if not todo and not pending:break
                if step==100:raise RuntimeError('有限任务上限')
                scans=[]
                if todo:
                    # 与round4相同的当前站开放路线，服务插入不能颠倒这些扫描点。
                    for p in open_route(todo,self.position):
                        i=min(range(len(todo)),key=lambda j:np.linalg.norm(todo[j]-p))
                        scans.append(('scan',i,todo[i]))
                services=[]
                for k in pending:
                    c,r=minimum_enclosing_circle(self.states[k].poly)
                    if not todo or r<=self.threshold:services.append(('target',k,c))
                if scans:tasks=insert_services(scans,services,self.position)
                else:
                    order=tour_order([t[2] for t in services],self.position,8)
                    tasks=[services[i] for i in order]
                kind,index,point=self.choose(tasks,todo,pending)
                self.task_events.append(dict(time_s=self.virtual_time,kind=kind,index=index,pending=len(pending),remaining_stations=len(todo)))
                if kind=='scan':self.scan(todo.pop(index))
                else:self.localize_and_clear(index)
            certificate=coverage_audit(self)
            if not certificate['valid']:raise RuntimeError('实际逐频道覆盖未通过')
            for b in self.states.values():
                if b.status=='undiscovered':b.status='absent_certified'
            if any(b.status not in ('cleared','absent_certified') for b in self.states.values()):raise RuntimeError('存在未清目标')
        except Exception as exc:
            status=f'failed: {type(exc).__name__}: {exc}';self.record(dict(failure=status))
        finally:
            if entered:
                try:self.act('/exit')
                except Exception as exc:status=f'failed: exit: {exc}'
            if self.log_file:self.log_file.close()
        n=sum(b.status=='cleared' for b in self.states.values())
        return dict(question=4,variant=self.variant,status=status,cleared=n,virtual_time_s=self.virtual_time,
            average_time_s=self.virtual_time/n if n else None,real_time_s=time.monotonic()-self.started,
            costs=self.costs,actions=self.actions,states={str(k):b.status for k,b in self.states.items()},
            completed_scan_count=len(self.completed_scans),station_count=21,task_events=self.task_events,
            listen_events=self.listen_events,pair_events=self.pair_events,portfolio=self.portfolio,decision_events=self.decision_events,phase_costs={k:dict(v) for k,v in self.phase_costs.items()},coverage_audit=coverage_audit(self))
