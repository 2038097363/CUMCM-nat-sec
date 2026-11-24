"""问题三连续覆盖区域联合调度算法（CCR-JS）。

连续覆盖区域的底层7个依赖模块位于src/q3_ccr_js_lib，哈希随结果记录。
Q4仅试验路径排序，不采用全向负信号排除或半径区间先验。
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
import io,time,hashlib
import numpy as np
from q3_ccr_js_adapter import PeerOffline,LIB
from adaptive_solver import disk_coverage_radius,reduced_stations
from geometry import optical_cover,minimum_enclosing_circle,observation_candidates
from dataclasses import asdict
from solve import open_route
from optical_solver import OpticalV2
from algorithms_p3 import improved_path


def multi_route(points,start):
    return [np.asarray(p) for p in improved_path([tuple(p) for p in points],tuple(start),'multi')]


class CCRJSQ3(PeerOffline):
    def __init__(self,backend,question=3,log_path=None,audit_callback=None,stop_on_discovery=True):
        if question!=3:raise ValueError('CCRJSQ3仅用于第三问')
        super().__init__(backend,audit_callback)
        self.stop_on_discovery=stop_on_discovery
        self.coverage_audit=None;self.early_stop_events=[];self.backup_events=[]
        if log_path is not None:
            p=Path(log_path);p.parent.mkdir(parents=True,exist_ok=True)
            self.log.close();self.log=p.open('x',encoding='utf-8')

    @property
    def states(self):
        certified=bool(self.coverage_audit and self.coverage_audit['valid'])
        return {k:SimpleNamespace(status='cleared' if k in self.cleared else 'found' if k in self.polygons
                else 'absent_certified' if certified else 'undiscovered') for k in range(1,21)}

    def refresh_pending(self,pending):
        if self.stop_on_discovery and self.remaining and len(set(self.polygons)|self.cleared)==16:
            self.early_stop_events.append(dict(time_s=self.virtual_time,remaining_stations=len(self.remaining)))
            self.remaining.clear()
        return super().refresh_pending(pending)

    def resolve(self,k):
        try:return super().resolve(k)
        except RuntimeError as exc:
            if 'fallback exceeds its action budget' not in str(exc):raise
        # 接管原包的大轴对齐包围框预算失败，使用本项目沿首测轴的覆盖网格。
        b=np.asarray(self.polygons[k]);first,bearing=self.good[k][0]
        points=optical_cover(b,np.asarray(first),bearing)
        def length(route):
            chain=[np.asarray(self.position)]+list(route)
            return sum(np.linalg.norm(q-p) for p,q in zip(chain[:-1],chain[1:]))
        route=min([points,list(reversed(points)),open_route(points,self.position)],key=length)
        self.backup_events.append(dict(channel=k,points=len(route)))
        for p in route:
            if self.clear(p,k):return
        raise RuntimeError('CCR-JS后备覆盖完成仍未清除')

    def certify_completion(self):
        checks=[];cache={}
        if len(self.cleared)<16:
            for k in range(1,21):
                if k in self.cleared:continue
                key=tuple(sorted(set(self.negative_evidence[k])))
                if key not in cache:cache[key]=disk_coverage_radius(key)[0] if key else float('inf')
                checks.append(dict(channel=k,radius=cache[key],valid=k not in self.polygons and cache[key]<=999.99))
        return dict(valid=all(c['valid'] for c in checks),channels=checks,count_bound=len(self.cleared)==16)

    def post(self,path,extra=None):
        if path=='/exit':
            # 即使核验失败也先退出会话；run随后报告失败，不能伪装完成。
            self.coverage_audit=self.certify_completion()
        return super().post(path,extra)

    def run(self):
        with redirect_stdout(io.StringIO()):raw=super().run()
        valid=bool(self.coverage_audit and self.coverage_audit['valid'])
        status='completed' if raw['error'] is None and valid else f"failed: {raw['error'] or '覆盖或清除未完成'}"
        return dict(question=3,policy='ccr_js',status=status,cleared=len(self.cleared),
            virtual_time_s=self.virtual_time,average_time_s=raw['average_clear_time_s'],real_time_s=raw['client_elapsed_s'],
            costs=self.costs,actions=self.actions,coverage_audit=self.coverage_audit,scan_points=self.scan_points,
            early_stop_events=self.early_stop_events,backup_events=self.backup_events,
            reference_source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in LIB.glob('*.py')},
            configuration=dict(region=self.region_config,localization=self.p3_config,stop_on_discovery=self.stop_on_discovery),
            source_attribution='用户提供同学problem3_best_v4；本项目新增接口适配、16源发现早停、完成核验和大网格后备')


class CCRJSQ4Optical(OpticalV2):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,mode='reuse',scan_weight=.25,optical_mode='rectangle',**kwargs)
        self.route_events=[]

    def optical_points(self,*args,**kwargs):
        points=super().optical_points(*args,**kwargs)
        def length(route):
            chain=[self.position]+list(route)
            return float(sum(np.linalg.norm(b-a) for a,b in zip(chain[:-1],chain[1:])))
        original=min([points,list(reversed(points)),open_route(points,self.position)],key=length)
        candidate=multi_route(points,self.position)
        if length(candidate)<length(original)-1e-6:
            self.route_events.append(dict(original_full_distance=length(original),new_full_distance=length(candidate)))
            return candidate
        return points

    def run(self):
        r=super().run();r['route_events']=self.route_events;return r


class CCRJSQ4Scan(OpticalV2):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,mode="reuse",scan_weight=.25,optical_mode="rectangle",**kwargs)

    def run(self):
        status='completed';entered=False
        all_stations=reduced_stations(self.question)
        todo=[s.copy() for s in all_stations if np.linalg.norm(s)>1e-8]
        try:
            enter=self.act('/enter');entered=True
            self.deadline=time.monotonic()+enter.get('remaining_real_duration_s',1200)
            self.scan(np.zeros(2))
            while todo or any(b.status=='found' for b in self.states.values()):
                # 已发现数量达到题给上限即可关闭发现任务，不必等到全部清除。
                if sum(b.status in ('found','cleared') for b in self.states.values())==16:
                    todo=[]
                    for b in self.states.values():
                        if b.status=='undiscovered':b.status='absent_certified'
                pending=[k for k,b in self.states.items() if b.status=='found']
                if sum(b.status=='cleared' for b in self.states.values())==16:break
                if self.mode=='reduced_only' and pending:
                    k=min(pending,key=lambda k:np.linalg.norm(self.states[k].poly.mean(axis=0)-self.position))
                    self.localize_and_clear(k);continue
                tasks=[]
                if todo:
                    path=multi_route(todo,self.position);p=path[0]
                    j=min(range(len(todo)),key=lambda j:np.linalg.norm(todo[j]-p))
                    unknown=sum(b.status=='undiscovered' for b in self.states.values())
                    tasks.append(((np.linalg.norm(p-self.position)/5+6*unknown)*self.scan_weight,'scan',j))
                for k in pending:
                    b=self.states[k];c,r=minimum_enclosing_circle(b.poly)
                    targets=observation_candidates(b.poly,self.position,b.observations,self.question)
                    travel=min([np.linalg.norm(p-self.position) for p in targets]+[np.linalg.norm(c-self.position)])
                    # 仅用于调度排序的下游成本近似，不能作为全任务最优性证书。
                    tasks.append((travel/5+5,'localize',k))
                _,kind,index=min(tasks)
                if kind=='scan':self.scan(todo.pop(index))
                else:
                    self.localize_and_clear(index)
                    self.opportunistic_swap(todo)
            if sum(b.status=='cleared' for b in self.states.values())<16:
                if todo:raise RuntimeError('覆盖路线未完成')
                for b in self.states.values():
                    if b.status=='undiscovered':b.status='absent_certified'
            if (sum(b.status=='cleared' for b in self.states.values())<16 and
                any(b.status not in ('cleared','absent_certified') for b in self.states.values())):
                raise RuntimeError('完成状态不完整')
        except Exception as exc:
            status=f'failed: {type(exc).__name__}: {exc}';self.record({'failure':status})
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
                    scan_weight=self.scan_weight,station_count=len(all_stations),
                    completed_scan_count=len(self.completed_scans),pruning=self.pruning)
