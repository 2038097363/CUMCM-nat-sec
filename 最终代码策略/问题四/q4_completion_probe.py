"""仅剩至多一个未发现源时，允许一次有路程上限的外站探查。"""
import time
import numpy as np
from q4_soft_dispatch import SoftQ4
from q4_scan_spine import insert_services
from q4_route_joint import tour_order
from adaptive_solver import reduced_stations
from solve import open_route
from geometry import minimum_enclosing_circle
from q4_value_audit import coverage_audit


class CompletionProbeQ4(SoftQ4):
    def __init__(self,backend,question=4,variant='probe3000',**kwargs):
        super().__init__(backend,question,variant='soft_cap',**kwargs)
        self.variant=variant;self.detour_limit=float(variant.replace('probe',''))
        self.probe_used=False;self.resume_station=None;self.probe_events=[]

    def choose(self,tasks,todo,pending):
        if not todo:self.resume_station=None
        if self.resume_station is not None:
            p=self.resume_station;self.resume_station=None
            j=min(range(len(todo)),key=lambda i:np.linalg.norm(todo[i]-p))
            if np.linalg.norm(todo[j]-p)>1e-6:raise RuntimeError('探查后的原计划站丢失')
            return 'scan',j,todo[j]
        original=super().choose(tasks,todo,pending)
        a,resultant,n=self.caution(todo)
        if self.probe_used or n!=15 or resultant<.8 or original[0]!='scan' or np.linalg.norm(original[2])>1800:return original
        circles=[minimum_enclosing_circle(b.poly) for b in self.states.values() if b.status in ('found','cleared') and b.observations]
        if not circles:return original
        center=np.average([c for c,r in circles],axis=0,weights=[1./(1.+r/300.) for c,r in circles])
        candidates=[]
        for j,p in enumerate(todo):
            if np.linalg.norm(p)<=1800 or np.linalg.norm(p-center)>1500:continue
            extra=float(np.linalg.norm(self.position-p)+np.linalg.norm(p-original[2])-np.linalg.norm(self.position-original[2]))
            if extra<=self.detour_limit:candidates.append((float(np.linalg.norm(p-self.position)),extra,j,p))
        if not candidates:return original
        _,extra,j,p=min(candidates,key=lambda x:x[:3])
        self.probe_used=True;self.resume_station=original[2].copy()
        event=dict(start_s=self.virtual_time,position=p.tolist(),resume=original[2].tolist(),movement_extra_bound_s=extra/5,unknown_channels=5)
        self.probe_events.append(event);self.record(dict(completion_probe_plan=event))
        return 'probe',j,p

    def execute_probe(self,point):
        old=self.mode;self.mode='reduced_only'
        try:self.scan(point)
        finally:self.mode=old
        event=self.probe_events[-1];event.update(end_s=self.virtual_time,discovered=sum(b.status in ('found','cleared') for b in self.states.values()))
        # 未命中则下一步直接回原计划站；原todo保留探查站，维持原巡检骨架。
        self.record(dict(completion_probe_result=dict(event)))

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
                if kind=='probe':self.execute_probe(point)
                elif kind=='scan':self.scan(todo.pop(index))
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
            listen_events=self.listen_events,pair_events=self.pair_events,portfolio=self.portfolio,decision_events=self.decision_events,probe_events=self.probe_events,phase_costs={k:dict(v) for k,v in self.phase_costs.items()},coverage_audit=coverage_audit(self))
