"""Q4独立研究：保留原调度的补测消融、扫描顺序约束的沿途插入。"""
import time
import numpy as np
from q4_route_joint import RouteQ4, tour_order
from q4_task_value import ValueQ4
from adaptive_solver import reduced_stations
from solve import open_route
from geometry import minimum_enclosing_circle
from q4_value_audit import coverage_audit


def insert_services(scans, services, start):
    """保持扫描相对次序；按最小增加路程逐个插入服务。仅为启发式。"""
    route=list(scans);remaining=list(services)
    while remaining:
        offers=[]
        for j,task in enumerate(remaining):
            for i in range(len(route)+1):
                a=start if i==0 else route[i-1][2]
                delta=np.linalg.norm(task[2]-a)
                if i<len(route):
                    b=route[i][2];delta+=np.linalg.norm(task[2]-b)-np.linalg.norm(a-b)
                offers.append((float(delta),j,i))
        _,j,i=min(offers);route.insert(i,remaining.pop(j))
    return route


class SpineQ4(RouteQ4):
    def __init__(self,backend,question=4,variant='baseline_listen8',**kwargs):
        super().__init__(backend,question,variant='weight_05',**kwargs)
        self.variant=variant;self.scan_weight=.25
        self.spine=variant.startswith('spine')
        self.threshold=100. if variant=='spine100' else 300.
        stride=1 if variant.endswith('listen1') else 4 if variant.endswith('listen4') else 8
        self.route_options=dict(weight=.25,listen_stride=stride)
        if variant=='round4_listen8':self.options={}

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
            self.scan(np.zeros(2))
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
                kind,index,point=tasks[0]
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
            listen_events=self.listen_events,phase_costs={k:dict(v) for k,v in self.phase_costs.items()},coverage_audit=coverage_audit(self))
