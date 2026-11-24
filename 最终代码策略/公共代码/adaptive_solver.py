"""第二轮结构优化：覆盖重规划、扫描与定位联合调度、跨频道复用观测。

保持旧solve.py作冻结基线。本文件仍处于核验阶段，不自动连接官方模拟器。
"""
from dataclasses import asdict
from itertools import combinations
import math
import time
import numpy as np
from scipy.spatial import Delaunay,QhullError,ConvexHull
from geometry import minimum_enclosing_circle,observation_candidates,circle3,clip,unit
from solve import Robot,PolicyConfig,coverage_stations,open_route
from directional_design_audit import stations
from optical_hex import hex_cover


def disk_coverage_radius(sites,area_radius=1800.):
    """连续圆域上最近站距离的最大值；检查Voronoi顶点和圆周分段极值。"""
    st=np.unique(np.round(np.asarray(sites),9),axis=0)
    candidates=[np.zeros(2),np.array([area_radius,0.])]
    if len(st)>=3:
        try:triples=[st[tri] for tri in Delaunay(st).simplices]
        except QhullError:triples=list(combinations(st,3))
        for a,b,c in triples:
            ab=b-a;ac=c-a
            if abs(ab[0]*ac[1]-ab[1]*ac[0])<1e-8:continue
            center,_=circle3(a,b,c)
            if np.linalg.norm(center)<=area_radius+1e-8:candidates.append(center)
    for a,b in combinations(st,2):
        normal=b-a;length=np.linalg.norm(normal)
        if length<1e-8:continue
        offset=(b@b-a@a)/(2*length)
        if abs(offset)<=area_radius:
            n=normal/length;side=np.array([-n[1],n[0]])
            height=math.sqrt(max(0,area_radius**2-offset**2))
            candidates.extend([offset*n+height*side,offset*n-height*side])
    for s in st:
        if np.linalg.norm(s)>1e-8:
            candidates.extend([area_radius*s/np.linalg.norm(s),-area_radius*s/np.linalg.norm(s)])
    points=np.array(candidates)
    nearest=np.min(np.linalg.norm(points[:,None,:]-st[None,:,:],axis=2),axis=1)
    i=int(np.argmax(nearest))
    return float(nearest[i]),points[i]


def reduced_stations(question):
    if question==3:return coverage_stations(3)
    return stations(8,12,999,1800/math.cos(math.pi/12)+3,0)


class AdaptiveRobot(Robot):
    def __init__(self,*args,mode='joint',scan_weight=1.,**kwargs):
        super().__init__(*args,**kwargs)
        self.report_mode=mode
        self.use_hex='_hex' in mode
        self.use_negative='_neg' in mode
        self.mode,self.scan_weight=mode.replace('_hex','').replace('_neg',''),scan_weight
        self.completed_scans=[]
        self.pruning=[]
        self.negative={k:[] for k in self.states}

    def measure(self,point,k,station_id=None):
        result=super().measure(point,k,station_id)
        if self.use_negative and self.question==3:
            if result=='no_signal':self.negative[k].append(np.asarray(point).copy())
            b=self.states[k]
            if b.status=='found':
                for s in self.negative[k]:
                    b.poly=exclude_negative_disk_hull(b.poly,s)
                if self.audit_callback:self.audit_callback(k,b.poly)
        return result

    def optical_points(self,poly,first_station,first_bearing,failed):
        if self.use_hex:return hex_cover(poly,first_station,first_bearing,failed)
        return super().optical_points(poly,first_station,first_bearing,failed)

    def scan(self,point):
        unknown=[k for k,b in self.states.items() if b.status=='undiscovered']
        unknown.sort(key=lambda k:(k!=self.channel,k))
        found_before=[k for k,b in self.states.items() if b.status=='found']
        for k in unknown:self.measure(point,k)
        self.completed_scans.append(np.asarray(point).copy())
        if self.mode in ('reuse','adaptive_reuse'):
            for k in found_before:
                b=self.states[k]
                c,r=minimum_enclosing_circle(b.poly)
                if r<=self.config.clear_margin or np.linalg.norm(c-point)-r>1500:continue
                if any(np.linalg.norm(point-s)<50 for s in b.attempted):continue
                self.measure(point,k)

    def can_remove(self,todo,index,extra=()):
        all_sites=self.completed_scans+[p for j,p in enumerate(todo) if j!=index]+list(extra)
        radius,witness=disk_coverage_radius(all_sites)
        return radius<=999.99,radius,witness

    def opportunistic_swap(self,todo):
        if self.question!=3 or self.mode not in ('adaptive','adaptive_reuse'):return
        if not todo or any(np.linalg.norm(self.position-p)<1 for p in self.completed_scans):return
        # 原地扫描只有能替代一个未来站点时才进行，不凭采样面积直接宣布覆盖。
        choices=[]
        for j,p in enumerate(todo):
            valid,radius,_=self.can_remove(todo,j,[self.position])
            if valid:choices.append((np.linalg.norm(p-self.position),j,radius))
        if choices:
            _,j,radius=max(choices)
            removed=todo[j].copy()
            self.scan(self.position.copy())
            todo.pop(j)
            self.pruning.append(dict(removed=removed.tolist(),replacement=self.position.tolist(),coverage_radius=radius))

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
                    path=open_route(todo,self.position);p=path[0]
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


def exclude_negative_disk_hull(poly,station,facets=32):
    """Q3无信号排除1000米圆；只删除其内接多边形，再外包为凸包。

    返回集合允许重新填入孔洞，属于安全但可能偏松的外近似；不能用于Q4。
    """
    center,r=minimum_enclosing_circle(poly)
    if np.linalg.norm(center-station)-r>=1000:return poly
    bound=999.99*math.cos(math.pi/facets)
    pieces=[]
    for angle in np.arange(facets)*360/facets:
        normal=unit(angle)
        part=clip(poly,-normal,-float(normal@station+bound))
        if len(part):pieces.extend(part)
    if not pieces:raise RuntimeError('负观测与已有定位约束冲突')
    points=np.unique(np.array(pieces),axis=0)
    if len(points)<3:return points
    try:return points[ConvexHull(points).vertices]
    except QhullError:return points
