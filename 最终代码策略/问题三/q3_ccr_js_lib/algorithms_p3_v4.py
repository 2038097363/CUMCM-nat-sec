"""Certified scan neighborhoods for problem 3; decisions use observations only."""
import math
from algorithms_p3 import ProblemThreeRobot, improved_path
from algorithms_p3_config import P3_CONFIGS, SELECTED_P3
from algorithms_p3_v4_config import V4_CONFIGS, SELECTED_V4
from robot_client import distance, center_radius


class ScanRegion:
    """Points covering the whole annular sector r=1000..1800, angle +/-30deg.

    The origin scan covers the inner disk. In each sector, squared distance
    is convex in target radius, and its angular maximum is at a sector end
    when the scanner also lies in the sector. Thus four endpoint constraints
    plus scanner sector membership suffice for CONTINUOUS coverage.
    """
    limit=999.9  # 0.1m reserve for request-coordinate rounding.

    def __init__(self,index,grid=25):
        self.index=index
        self.theta=index*math.pi/3
        self.c,self.s=math.cos(self.theta),math.sin(self.theta)
        self.corners=[(r*math.cos(self.theta+a),r*math.sin(self.theta+a))
                      for r in (1000,1800) for a in (-math.pi/6,math.pi/6)]
        self.anchor=self.world((1150,0))
        self.candidates=[self.world((x,y)) for x in range(1125,1726,grid) for y in range(-100,101,grid)
                         if self.valid(self.world((x,y)))]
        self.candidates.append(self.anchor)

    def world(self,p):
        return self.c*p[0]-self.s*p[1],self.s*p[0]+self.c*p[1]

    def valid(self,p,limit=None):
        x,y=self.c*p[0]+self.s*p[1],-self.s*p[0]+self.c*p[1]
        limit=self.limit if limit is None else limit
        return x>0 and abs(y)<=x/math.sqrt(3)+1e-8 and all(distance(p,q)<=limit+1e-8 for q in self.corners)

    def chord(self,a,b):
        dx,dy=b[0]-a[0],b[1]-a[1]
        aa=dx*dx+dy*dy
        if aa<1e-12:return a if self.valid(a) else None
        low,high=0.,1.
        for q in self.corners:
            x,y=a[0]-q[0],a[1]-q[1]
            bb=2*(x*dx+y*dy)
            cc=x*x+y*y-self.limit*self.limit
            discriminant=bb*bb-4*aa*cc
            if discriminant<0:return None
            root=math.sqrt(discriminant)
            low=max(low,(-bb-root)/(2*aa))
            high=min(high,(-bb+root)/(2*aa))
            if low>high:return None
        t=(low+high)/2
        p=a[0]+t*dx,a[1]+t*dy
        return p if self.valid(p) else None

    def closest(self,p):
        # Alternating projections, with a certified fallback if convergence
        # has not reached the intersection. Never accept an unchecked point.
        q=p
        for _ in range(80):
            for center in self.corners:
                d=distance(q,center)
                if d>self.limit:
                    q=tuple(center[k]+(q[k]-center[k])*self.limit/d for k in (0,1))
            if self.valid(q):return q
        return min(self.candidates,key=lambda q:distance(p,q))


class RegionRobot(ProblemThreeRobot):
    def __init__(self,*args,region_config=None,**kwargs):
        problem=kwargs.get('problem',args[1] if len(args)>1 else None)
        super().__init__(*args,p3_config=dict(P3_CONFIGS[SELECTED_P3]) if problem==3 else {},**kwargs)
        self.region_config=(region_config or {}) if problem==3 else {}
        self.regions=[ScanRegion(i,self.region_config.get('grid',25)) for i in range(6)] if self.region_config and not self.region_config.get('fixed') else []
        self.region_ids={r.anchor:r.index for r in self.regions}
        self.posterior_cache={}
        self.prior_points=[(1800*math.sqrt((i+.5)/8)*math.cos(2*math.pi*j/24),
                            1800*math.sqrt((i+.5)/8)*math.sin(2*math.pi*j/24))
                           for i in range(8) for j in range(24)] if self.region_config.get('gain') else []

    def weighted_locations(self,channel):
        poly=self.polygons[channel]
        signature=(tuple(poly),len(self.good[channel]),len(self.negatives.get(channel,[])))
        if channel in self.posterior_cache and self.posterior_cache[channel][0]==signature:
            return self.posterior_cache[channel][1]
        samples=[]
        a=poly[0]
        for b,c in zip(poly[1:],poly[2:]):
            area=abs((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]))/2
            if area<1e-12:continue
            for i in range(3):
                for j in range(3-i):
                    bary=[((i+1/3)/3,(j+1/3)/3)]
                    if i+j<2:bary.append(((i+2/3)/3,(j+2/3)/3))
                    for u,v in bary:
                        p=(a[0]+u*(b[0]-a[0])+v*(c[0]-a[0]),a[1]+u*(b[1]-a[1])+v*(c[1]-a[1]))
                        if math.hypot(*p)>1800:continue
                        low=max([1000]+[distance(p,q) for q,_ in self.good[channel]])
                        high=min([1500]+[distance(p,q) for q in self.negatives.get(channel,[])])
                        weight=area/9*max(0,high-low)
                        if weight>0:samples.append((p,weight))
        # Uniform source area and uniform unknown reception radius are local
        # modeling assumptions, used ONLY for ranking/probes, not certificates.
        if not samples:
            point,_=super().location(channel)
            samples=[(point,1.)]
        total=sum(w for p,w in samples)
        samples=[(p,w/total) for p,w in samples]
        self.posterior_cache[channel]=(signature,samples)
        return samples

    def posterior_point(self,channel):
        samples=self.weighted_locations(channel)
        point=tuple(sum(p[k]*w for p,w in samples) for k in (0,1))
        if self.region_config.get('posterior')=='median':
            for _ in range(12):
                weights=[w/max(1e-4,distance(p,point)) for p,w in samples]
                total=sum(weights)
                point=tuple(sum(p[k]*w for (p,_),w in zip(samples,weights))/total for k in (0,1))
        return point

    def location(self,channel):
        if not self.region_config.get('posterior'):return super().location(channel)
        point=self.posterior_point(channel)
        return point,max(distance(point,p) for p in self.polygons[channel])

    def planned_centers(self,pending):
        if not self.region_config.get('posterior_plan'):return super().planned_centers(pending)
        return {c:self.posterior_point(c) for c in sorted(pending)}

    def active_probe(self,channel):
        if not self.region_config.get('posterior_probe'):return super().active_probe(channel)
        old,bearing=self.good[channel][-1]
        theta=math.radians(bearing)
        u=(math.cos(theta),math.sin(theta))
        radius=distance(old,self.posterior_point(channel))
        samples=self.weighted_locations(channel)
        options=[]
        for forward in (.25,.5,.75,1.):
            for side in (-250,-125,-60,60,125,250):
                p=(old[0]+forward*radius*u[0]-side*u[1],old[1]+forward*radius*u[1]+side*u[0])
                future=0
                for source,weight in samples:
                    a,b=distance(old,source),distance(p,source)
                    cross=abs((old[0]-source[0])*(p[1]-source[1])-(old[1]-source[1])*(p[0]-source[0]))
                    sine=cross/max(1,a*b)
                    uncertainty=min(1500,math.radians(1.005)*math.hypot(a,b)/max(.01,sine))
                    future+=weight*(b+uncertainty)
                options.append((distance(self.position,p)+future,p))
        return min(options)[1]

    def survey_stations(self):
        if not self.regions:return super().survey_stations()
        self.remaining=[(0.,0.)]+[r.anchor for r in self.regions]
        return self.remaining

    def optimize_locations(self,pending):
        labels=[('station',self.region_ids[p]) for p in self.remaining]+[('target',c) for c in sorted(pending)]
        positions={label:p for label,p in zip(labels,list(self.remaining)+list(self.planned_centers(pending).values()))}
        unseen=[q for q in self.prior_points if all(distance(q,p)>1000 for p in self.scan_points)]
        gains={}
        def bonus(point):
            if not unseen:return 0
            if point not in gains:
                gains[point]=self.region_config.get('gain',0)*sum(distance(q,point)<=1000 for q in unseen)/len(self.prior_points)
            return gains[point]
        for _ in range(self.region_config['passes']):
            route_points=improved_path([positions[label] for label in labels],self.position,self.region_config['route'])
            available=list(labels)
            route=[]
            for point in route_points:
                label=next(label for label in available if positions[label]==point)
                route.append(label);available.remove(label)
            for i,label in enumerate(route):
                if label[0]!='station':continue
                region=self.regions[label[1]]
                before=self.position if i==0 else positions[route[i-1]]
                after=positions[route[i+1]] if i+1<len(route) else None
                if self.region_config.get('greedy'):after=None
                options=region.candidates+[positions[label],region.closest(before)]
                if after is not None:
                    options.append(region.closest(after))
                    crossing=region.chord(before,after)
                    if crossing is not None:options.append(crossing)
                positions[label]=min(options,key=lambda p:distance(before,p)+(distance(p,after) if after is not None else 0)-bonus(p))
                if not region.valid(positions[label]):raise RuntimeError('Uncertified scan point')
        # Final routing pass after location relaxation; labels preserve identity
        # even if a target estimate happens to equal a scan point.
        route_points=improved_path([positions[label] for label in labels],self.position,self.region_config['route'])
        first=next(label for label in labels if positions[label]==route_points[0])
        old_labels=[('station',self.region_ids[p]) for p in self.remaining]
        self.remaining[:]=[positions[label] for label in old_labels]
        for label in old_labels:self.region_ids[positions[label]]=label[1]
        return first,positions[first]

    def refresh_pending(self,pending):
        if not self.region_config or not self.remaining:return super().refresh_pending(pending)
        pending=set(self.polygons)-self.cleared
        if 'sweep' in self.region_config and len(self.remaining)>self.region_config['sweep']:
            pending=set()
        elif self.region_config.get('lazy'):
            pending={c for c in pending if center_radius(self.polygons[c])[1]<=self.region_config['lazy']}
        if self.region_config.get('fixed'):
            centers=self.planned_centers(pending)
            points=list(self.remaining)+list(centers.values())
            first=improved_path(points,self.position,self.region_config['route'])[0]
            if first in self.remaining:
                self.plan_station=first;self.plan_channel=None
                return set()
            self.plan_station=None;self.plan_channel=next(c for c,p in centers.items() if p==first)
            return pending
        # Called only after the origin scan has been removed from remaining.
        if any(p not in self.region_ids for p in self.remaining):return super().refresh_pending(pending)
        first,position=self.optimize_locations(pending)
        if first[0]=='station':
            self.plan_station=position;self.plan_channel=None
            return set()
        self.plan_station=None;self.plan_channel=first[1]
        return pending

    def survey(self,point):
        if self.regions and point!=(0.,0.):
            region=self.regions[self.region_ids[point]]
            # Planning leaves 0.1m headroom. Six-decimal serialization may
            # move a boundary point by <0.000001m; still demand a 0.01m reserve
            # relative to the actual 1000m guarantee at the executed point.
            if not region.valid(tuple(round(x,6) for x in point),limit=999.99):
                raise RuntimeError('Rounded scan point not certified')
        return super().survey(point)

    def after_resolve(self):
        super().after_resolve()
        if not self.region_config.get('opportunistic'):return
        covered=[p for p in self.remaining if self.regions[self.region_ids[p]].valid(self.position)]
        if not covered:return
        point=self.position
        for channel in range(1,21):
            if channel not in self.polygons and channel not in self.cleared:self.measure(point,channel)
        self.scan_points.append(point)
        for p in covered:self.remaining.remove(p)


def region_robot_class(name):
    class Configured(RegionRobot):
        def __init__(self,*args,**kwargs):
            problem=kwargs.get('problem',args[1] if len(args)>1 else None)
            if problem!=3 and name!='optimized_v4':raise ValueError('p3_v4_* candidates require problem 3')
            selected=SELECTED_V4 if name=='optimized_v4' else name
            super().__init__(*args,region_config=V4_CONFIGS[selected] if problem==3 else {},**kwargs)
    return Configured
