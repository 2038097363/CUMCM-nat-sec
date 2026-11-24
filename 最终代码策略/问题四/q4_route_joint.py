"""Q4第九轮研究：扫描与服务目标共同进入开放路线。保证层沿用21站。"""
import time
import numpy as np
from q4_task_value import ValueQ4
from adaptive_solver import reduced_stations
from geometry import minimum_enclosing_circle,nearest_clear_point
from q4_value_audit import coverage_audit


def tour_order(points,start,starts=1):
    """距离矩阵、多起点最近邻+开放2-opt，首位置固定，不闭环。"""
    if not len(points):return []
    pts=np.vstack((start,points));dist=np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2)
    best=None
    for first in sorted(range(1,len(pts)),key=lambda i:dist[0,i])[:starts]:
        left=set(range(1,len(pts)));left.remove(first);route=[0,first]
        while left:
            j=min(left,key=lambda j:(dist[route[-1],j],j));route.append(j);left.remove(j)
        for _ in range(20):
            changed=False
            for i in range(1,len(route)-1):
                for j in range(i+1,len(route)):
                    a,b,c=route[i-1],route[i],route[j]
                    change=dist[a,c]-dist[a,b]
                    if j+1<len(route):d=route[j+1];change+=dist[b,d]-dist[c,d]
                    if change< -1e-7:route[i:j+1]=reversed(route[i:j+1]);changed=True
            if not changed:break
        length=sum(dist[a,b] for a,b in zip(route,route[1:]))
        if best is None or length<best[0]:best=(length,route[1:])
    return [i-1 for i in best[1]]


MODES={
    'tour_all':dict(ready=False,starts=1),
    'tour_ready':dict(ready=True,starts=1),
    'tour_all_multi':dict(ready=False,starts=4),
    'tour_ready_multi':dict(ready=True,starts=4),
    'tour_small_multi':dict(ready=100.,starts=4),
    'tour_all_eight':dict(ready=False,starts=8),
    'tour_all_listen4':dict(ready=False,starts=8,listen_stride=4),
    'tour_all_listen8':dict(ready=False,starts=8,listen_stride=8),
    'tour_small_listen4':dict(ready=100.,starts=4,listen_stride=4),
    'tour_small_listen8':dict(ready=100.,starts=4,listen_stride=8),
    'tour_200_listen8':dict(ready=200.,starts=8,listen_stride=8),
    'tour_300_listen8':dict(ready=300.,starts=8,listen_stride=8),
    'tour_two_positive':dict(ready=False,starts=8,listen_stride=8,two_positive=True),
    'horizon2':dict(ready=False,starts=8,listen_stride=8,horizon=2),
    'horizon4':dict(ready=False,starts=8,listen_stride=8,horizon=4),
    'horizon8':dict(ready=False,starts=8,listen_stride=8,horizon=8),
    'horizon4_small':dict(ready=100.,starts=8,listen_stride=8,horizon=4),
    'scan_first':dict(ready=True,starts=4,scan_first=True),
    'weight_05':dict(weight=.5),
    'weight_10':dict(weight=1.),
    'weight_20':dict(weight=2.),
}


class RouteQ4(ValueQ4):
    def __init__(self,backend,question=4,variant='tour_ready_multi',**kwargs):
        super().__init__(backend,question,variant='state_two_dense',**kwargs)
        self.variant=variant;self.route_options=dict(MODES[variant]);self.task_events=[]
        self.optical_active=False;self.listening=False;self.optical_replan=False
        self.listen_count=0;self.optical_failures=0;self.listen_events=[]
        if 'weight' in self.route_options:self.scan_weight=self.route_options['weight']

    def localize_and_clear(self,k):
        self.listen_count=0;self.optical_failures=0;self.optical_active=False
        try:return super().localize_and_clear(k)
        finally:self.optical_active=False

    def clear(self,point,k):
        success=super().clear(point,k)
        stride=self.route_options.get('listen_stride')
        if success or not stride or not self.optical_active or self.listening:return success
        self.optical_failures+=1
        if (self.optical_failures-1)%stride or self.listen_count>=8:return False
        if any(np.linalg.norm(point-p)<2 for p in self.states[k].attempted):return False
        self.listen_count+=1;self.listening=True
        try:
            before=minimum_enclosing_circle(self.states[k].poly)[1]
            response=self.measure(point,k);after=minimum_enclosing_circle(self.states[k].poly)[1]
            if self.states[k].status!='cleared' and after<=19.9:
                q=nearest_clear_point(self.states[k].poly,self.position,19.9)
                if not self.clear(q,k):raise RuntimeError('补测后保证清除失败')
            self.optical_replan=after<.8*before
            event=dict(channel=k,failure_index=self.optical_failures,response=response,radius_before=before,radius_after=after,replan=self.optical_replan)
            self.listen_events.append(event);self.record(dict(optical_listen=event))
            return self.states[k].status=='cleared'
        finally:self.listening=False

    def optical_route(self,k,planning=False):
        route,cost=super().optical_route(k,planning)
        if planning or not self.route_options.get('listen_stride'):return route,cost
        def iterate():
            self.optical_active=True;self.optical_replan=False;points=route;index=0
            while index<len(points):
                yield points[index];index+=1
                if self.optical_replan:
                    points,_=super(RouteQ4,self).optical_route(k)
                    index=0;self.optical_replan=False
        return iterate(),cost

    def run(self):
        if 'weight' in self.route_options:return super().run()
        entered=False;status='completed';todo=list(reduced_stations(4)[1:]);steps=0
        try:
            response=self.act('/enter');entered=True;self.deadline=time.monotonic()+response.get('remaining_real_duration_s',1200)
            self.scan(np.zeros(2))
            while todo or any(b.status=='found' for b in self.states.values()):
                steps+=1
                if steps>100:raise RuntimeError('任务循环超出有限上限')
                if sum(b.status in ('found','cleared') for b in self.states.values())==16:todo=[]
                indices=range(len(todo))
                if self.route_options.get('horizon') and todo:
                    indices=tour_order(todo,self.position,4)[:self.route_options['horizon']]
                tasks=[('scan',i,todo[i]) for i in indices]
                pending=[]
                for k,b in self.states.items():
                    if b.status!='found':continue
                    c,r=minimum_enclosing_circle(b.poly);pending.append(('target',k,c))
                    threshold=self.route_options['ready']
                    if threshold is True:threshold=19.9
                    eligible=not todo or not threshold or r<=threshold
                    if todo and self.route_options.get('two_positive'):eligible=len(b.observations)>=2 or r<=19.9
                    if eligible and not (todo and self.route_options.get('scan_first')):tasks.append(('target',k,c))
                if not tasks:tasks=pending
                if not tasks:break
                order=tour_order([t[2] for t in tasks],self.position,self.route_options['starts'])
                kind,index,point=tasks[order[0]]
                self.task_events.append(dict(time_s=self.virtual_time,kind=kind,index=index,pending=len(pending),remaining_stations=len(todo)))
                if kind=='scan':self.scan(todo.pop(index))
                else:self.localize_and_clear(index)
            if todo and sum(b.status=='cleared' for b in self.states.values())!=16:raise RuntimeError('覆盖尚未完成')
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
        certificate=coverage_audit(self)
        if status=='completed' and not certificate['valid']:status='failed: 实际逐频道覆盖未通过'
        return dict(question=4,variant=self.variant,status=status,cleared=n,virtual_time_s=self.virtual_time,average_time_s=self.virtual_time/n if n else None,
            real_time_s=time.monotonic()-self.started,costs=self.costs,actions=self.actions,states={str(k):b.status for k,b in self.states.items()},
            station_count=21,completed_scan_count=len(self.completed_scans),phase_costs={k:dict(v) for k,v in self.phase_costs.items()},task_events=self.task_events,listen_events=self.listen_events,
            coverage_audit=certificate,route_options=self.route_options,local_measure_limit=2,optical_listen_limit=8)
