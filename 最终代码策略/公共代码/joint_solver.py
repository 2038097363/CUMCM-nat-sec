"""第三轮：按频道覆盖证书、沿途补扫、原子动作调度、成本驱动试探。

复用旧接口与几何代码。Q4自动禁用无信号圆排除/删站，仅可比较公共成本模块。
"""
from dataclasses import asdict
import math,time
import numpy as np
from adaptive_solver import AdaptiveRobot,reduced_stations
from solve import coverage_stations,open_route
from geometry import minimum_enclosing_circle,nearest_clear_point,observation_candidates
from channel_coverage import ChannelCoverage
from action_cost import clear_score,measurement_tail,detour_probe_terms


class JointRobot(AdaptiveRobot):
    def __init__(self,*args,channel_cover=True,cost_actions=True,ring_radius=1150.,
                 gain_scans=False,max_extra_per_channel=3,cost_style='worst',**kwargs):
        super().__init__(*args,mode='reuse_neg',**kwargs)
        self.channel_cover=bool(channel_cover and self.question==3)
        self.cost_actions=cost_actions;self.ring_radius=ring_radius
        self.gain_scans=bool(gain_scans and self.channel_cover)
        self.max_extra_per_channel=max_extra_per_channel
        self.cost_style=cost_style;self.probe_followup={};self.cost_details={}
        self.ledger=ChannelCoverage();self.todo=[];self.station_ids=[]
        self.targeted={k:0 for k in self.states};self.probes={k:0 for k in self.states}
        self.extras={k:0 for k in self.states};self.cost_cache={};self.certificates=[]
        self.report_mode=('channel' if self.channel_cover else 'fixed')+('_cost' if cost_actions else '_threshold')
        if self.gain_scans:self.report_mode+='_gain'
        if cost_actions and cost_style=='detour':self.report_mode=self.report_mode.replace('_cost','_detour')

    def measure(self,point,k,station_id=None):
        result=super().measure(point,k,station_id)
        if self.question==3 and result=='no_signal':self.ledger.add_negative(k,point)
        return result

    def unknown(self):return [k for k,b in self.states.items() if b.status=='undiscovered']

    def certify_absent(self):
        if not self.channel_cover:return
        for k in self.unknown():
            valid,r,w=self.ledger.certificate(k)
            if valid:
                self.states[k].status='absent_certified'
                self.certificates.append(dict(channel=k,type='absent',radius=r,witness=w,
                                              evidence_event_count=len(self.history),
                                              negative_points=[p.tolist() for p in self.ledger.negative[k]]))

    def cancel_by_count(self):
        if sum(b.status in ('found','cleared') for b in self.states.values())==16:
            self.todo=[]
            for k in self.unknown():self.states[k].status='absent_certified'
            return True
        return False

    def retire_at_current(self):
        """逐频道核验，只有补测缺少证据的频道后才删除该预设站。"""
        if not self.channel_cover or not self.todo or self.cancel_by_count():return
        self.certify_absent()
        for j in sorted(range(len(self.todo)),key=lambda i:np.linalg.norm(self.todo[i]-self.position),reverse=True):
            future=[p for i,p in enumerate(self.todo) if i!=j];needed=[];possible=True
            for k in self.unknown():
                if self.ledger.certificate(k,future)[0]:continue
                if not self.ledger.certificate(k,future,[self.position])[0]:possible=False;break
                needed.append(k)
            if not possible:continue
            removed=self.todo[j].copy();position=self.position.copy()
            for k in sorted(needed,key=lambda k:(k!=self.channel,k)):
                self.measure(position,k);self.extras[k]+=1
            checks=[]
            for k in self.unknown():
                valid,r,w=self.ledger.certificate(k,future)
                if not valid:raise RuntimeError('删站时频道覆盖证书不完整')
                checks.append(dict(channel=k,radius=r,witness=w,
                                   negative_points=[p.tolist() for p in self.ledger.negative[k]]))
            self.todo.pop(j)
            self.pruning.append(dict(removed=removed.tolist(),replacement=position.tolist(),
                                     evidence_event_count=len(self.history),
                                     measured_channels=needed,future=[p.tolist() for p in future],checks=checks))
            self.certify_absent();return

    def stationary_service(self):
        if self.cancel_by_count():return
        # 如果定位动作恰好到达预设站，立即复用本次停点完成该站扫描。
        near=[j for j,p in enumerate(self.todo) if np.linalg.norm(p-self.position)<1e-6]
        if near:self.scan(self.todo.pop(near[0]))
        if self.channel_cover:
            self.retire_at_current()
            if self.gain_scans and self.todo:
                for k in self.unknown():
                    if self.extras[k]>=self.max_extra_per_channel:continue
                    # 固定等面积求积只决定是否值得测；每频道最多3次这种探索补扫。
                    gain=self.ledger.estimated_gain(k,self.position)
                    if gain>=.35 and all(np.linalg.norm(self.position-p)>100 for p in self.states[k].attempted):
                        self.measure(self.position.copy(),k);self.extras[k]+=1
                self.retire_at_current()

    def candidates(self,k):
        b=self.states[k]
        points=observation_candidates(b.poly,self.position,b.observations,self.question)
        points+=list(self.todo)
        points=[p for p in points if all(np.linalg.norm(p-a)>2 for a in b.attempted)]
        if self.question==3:
            safe=[p for p in points if np.max(np.linalg.norm(b.poly-p,axis=1))<=1000]
            if safe:points=safe
        return points

    def local_action(self,k):
        b=self.states[k];center,r=minimum_enclosing_circle(b.poly)
        if r<=self.config.clear_margin:
            p=nearest_clear_point(b.poly,self.position,self.config.clear_margin)
            return (np.linalg.norm(p-self.position)/5+5,'guaranteed',k,p)
        if self.targeted[k]>=self.config.max_measure_attempts:return (0.,'fallback',k,center)
        points=self.candidates(k)
        if not points:return (0.,'fallback',k,center)
        can_probe=self.probes[k]<2 and all(np.linalg.norm(center-f)>3 for f in b.failed_clears)
        if not self.cost_actions:
            if r<=100 and can_probe:return (np.linalg.norm(center-self.position)/5+5,'probe',k,center)
            p=min(points,key=lambda p:np.linalg.norm(p-self.position))
            return (np.linalg.norm(p-self.position)/5+6,'measure',k,p)
        if self.cost_style=='detour':
            p=min(points,key=lambda p:np.linalg.norm(p-self.position))
            if can_probe:
                terms=detour_probe_terms(self.position,center,p,self.channel!=k)
                if terms['allowed']:
                    self.probe_followup[k]=p.copy();self.cost_details[k]=terms
                    return (terms['success_s'],'probe_then_measure',k,center)
            return (np.linalg.norm(p-self.position)/5+5+int(self.channel!=k),'measure',k,p)
        choices=[];polykey=b.poly.tobytes()
        for p in points:
            key=(k,polykey,tuple(p))
            if key not in self.cost_cache:
                safe=self.question==3 and np.max(np.linalg.norm(b.poly-p,axis=1))<=1000
                self.cost_cache[key]=measurement_tail(b.poly,p,self.config.delta,safe)
            score=np.linalg.norm(p-self.position)/5+5+int(self.channel!=k)+self.cost_cache[key]
            choices.append((score,'measure',k,p))
        if can_probe:choices.append((clear_score(b.poly,self.position,center),'probe',k,center))
        return min(choices,key=lambda item:item[0])

    def fallback(self,k):
        b=self.states[k];s,a=b.observations[0]
        points=self.optical_points(b.poly,s,a,b.failed_clears)
        def length(route):
            chain=[self.position]+list(route)
            return sum(np.linalg.norm(x-y) for x,y in zip(chain[:-1],chain[1:]))
        route=min([points,list(reversed(points)),open_route(points,self.position)],key=length)
        for point in route:
            if self.clear(point,k):return
        raise RuntimeError('光学后备覆盖结束仍未清除')

    def run(self):
        status='completed';entered=False
        sites=coverage_stations(3,q3_radius=self.ring_radius) if self.question==3 else reduced_stations(4)
        self.todo=[p.copy() for p in sites if np.linalg.norm(p)>1e-8]
        try:
            response=self.act('/enter');entered=True
            self.deadline=time.monotonic()+response.get('remaining_real_duration_s',1200)
            self.scan(np.zeros(2));iterations=0
            while self.todo or any(b.status=='found' for b in self.states.values()):
                iterations+=1
                if iterations>512:raise RuntimeError('动作进展异常，禁止无界循环')
                self.cancel_by_count();self.certify_absent()
                if self.channel_cover and not self.unknown():self.todo=[]
                tasks=[]
                if self.todo:
                    p=open_route(self.todo,self.position)[0]
                    tasks.append(((np.linalg.norm(p-self.position)/5+6*len(self.unknown()))*self.scan_weight,'scan',0,p))
                tasks += [self.local_action(k) for k,b in self.states.items() if b.status=='found']
                if not tasks:break
                score,kind,k,p=min(tasks,key=lambda item:item[0])
                self.record(dict(decision=kind,channel=k,point=p.tolist(),score=float(score)))
                if kind=='scan':
                    j=min(range(len(self.todo)),key=lambda i:np.linalg.norm(self.todo[i]-p))
                    self.scan(self.todo.pop(j))
                elif kind=='measure':self.targeted[k]+=1;self.measure(p,k)
                elif kind=='probe':self.probes[k]+=1;self.clear(p,k)
                elif kind=='probe_then_measure':
                    self.probes[k]+=1
                    self.record(dict(probe_cost_comparison=self.cost_details[k],channel=k))
                    if not self.clear(p,k):
                        self.targeted[k]+=1;self.measure(self.probe_followup[k],k)
                elif kind=='guaranteed':
                    if not self.clear(p,k):raise RuntimeError('保证清除失败')
                else:self.fallback(k)
                if kind!='scan':self.stationary_service()
            if self.channel_cover:
                self.certify_absent()
                if self.unknown():raise RuntimeError('结束时仍有未排查频道')
            elif not self.todo:
                for k in self.unknown():self.states[k].status='absent_certified'
            if any(b.status not in ('cleared','absent_certified') for b in self.states.values()):raise RuntimeError('结束状态不完整')
        except Exception as exc:
            status=f'failed: {type(exc).__name__}: {exc}';self.record(dict(failure=status))
        finally:
            if entered:
                try:self.act('/exit')
                except Exception as exc:status=f'failed: exit: {exc}'
            if self.log_file:self.log_file.close()
        n=sum(b.status=='cleared' for b in self.states.values())
        return dict(question=self.question,policy=self.report_mode,status=status,cleared=n,
                    virtual_time_s=self.virtual_time,average_time_s=self.virtual_time/n if n else None,
                    real_time_s=time.monotonic()-self.started,costs=self.costs,actions=self.actions,
                    states={str(k):b.status for k,b in self.states.items()},configuration=asdict(self.config),
                    channel_cover=self.channel_cover,cost_actions=self.cost_actions,ring_radius=self.ring_radius,
                    cost_style=self.cost_style,
                    scan_weight=self.scan_weight,extra_measurements=sum(self.extras.values()),
                    completed_scan_count=len(self.completed_scans),pruning=self.pruning,
                    coverage_certificates=self.certificates)
