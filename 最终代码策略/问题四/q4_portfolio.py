"""首扫方向集中度驱动的调度组合；不读取源数、类型或位置真值。"""
import time
import numpy as np
from q4_negative_pair import NegativePairQ4
from q4_scan_spine import insert_services
from q4_route_joint import tour_order
from adaptive_solver import reduced_stations
from solve import open_route
from geometry import minimum_enclosing_circle,observation_candidates
from q4_value_audit import coverage_audit


class PortfolioQ4(NegativePairQ4):
    def __init__(self,backend,question=4,variant='portfolio90',**kwargs):
        super().__init__(backend,question,variant='negative_spine',**kwargs)
        self.variant=variant;self.sector_limit=float(variant.replace('portfolio',''))
        self.compact_search=False;self.portfolio={}

    def classify_origin(self):
        angles=sorted(b.observations[0][1]%360 for b in self.states.values() if b.observations)
        if len(angles)>=3:
            gaps=np.diff(angles+[angles[0]+360.]);span=float(360-max(gaps))
        else:span=360.
        self.compact_search=len(angles)>=3 and span<=self.sector_limit
        self.portfolio=dict(positive_channels=len(angles),angular_span=span,threshold=self.sector_limit,compact_search=self.compact_search)
        self.record(dict(portfolio=self.portfolio))

    def baseline_choice(self,todo,pending):
        offers=[]
        if todo:
            p=open_route(todo,self.position)[0]
            j=min(range(len(todo)),key=lambda i:np.linalg.norm(todo[i]-p))
            unknown=sum(b.status=='undiscovered' for b in self.states.values())
            offers.append(((np.linalg.norm(p-self.position)/5+6*unknown)*.25,'scan',j,p))
        for k in pending:
            b=self.states[k];c,r=minimum_enclosing_circle(b.poly)
            points=observation_candidates(b.poly,self.position,b.observations,4)
            distance=min([np.linalg.norm(p-self.position) for p in points]+[np.linalg.norm(c-self.position)])
            offers.append((distance/5+5,'localize',k,c))
        _,kind,index,point=min(offers,key=lambda t:t[:3])
        return kind,index,point

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
                if self.compact_search:
                    kind,index,point=self.baseline_choice(todo,pending)
                else:kind,index,point=tasks[0]
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
            listen_events=self.listen_events,pair_events=self.pair_events,portfolio=self.portfolio,phase_costs={k:dict(v) for k,v in self.phase_costs.items()},coverage_audit=coverage_audit(self))
