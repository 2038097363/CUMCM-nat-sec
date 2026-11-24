"""Second-round alternatives; no simulator imports or hidden-state access."""
import math
from robot_client import distance, center_radius


def centroid(poly):
    area=x=y=0.0
    for a,b in zip(poly,poly[1:]+poly[:1]):
        cross=a[0]*b[1]-b[0]*a[1]
        area+=cross
        x+=(a[0]+b[0])*cross
        y+=(a[1]+b[1])*cross
    if abs(area)<1e-9:
        return center_radius(poly)[0]
    return x/(3*area),y/(3*area)


VARIANTS={
    'v2_centroid':dict(location='centroid'),
    'v2_approach40':dict(approach=.4),
    'v2_approach70':dict(approach=.7),
    'v2_direct':dict(approach=1.),
    'v2_centroid_direct':dict(location='centroid',approach=1.),
    'v2_information':dict(approach='information'),
    'v2_target_route':dict(target_route=True),
    'v2_scan400':dict(scan=400),
    'v2_scan800':dict(scan=800),
    'v2_scan1200':dict(scan=1200),
    'v2_cover':dict(coverage=True),
    'v2_cover_scan':dict(coverage=True,scan=600),
    'v2_cover_direct':dict(coverage=True,scan=600,approach=1.),
    'v2_cover_information':dict(coverage=True,scan=600,approach='information'),
    'v2_joint50':dict(joint=.5,target_route=True),
    'v2_joint100':dict(joint=1.,target_route=True),
    'v2_joint150':dict(joint=1.5,target_route=True),
    'v2_joint_scan':dict(joint=1.,target_route=True,scan=1200),
    'v2_centroid_route':dict(location='centroid',target_route=True),
    'v2_certified':dict(coverage='fixed'),
    'v2_certified_scan':dict(coverage='fixed',scan=600),
    'v2_certified_replace':dict(coverage='fixed',replace=True),
    'v2_certified_route':dict(coverage='fixed',replace=True,target_route=True),
    'v2_joint75':dict(joint=.75,target_route=True),
    'v2_joint120':dict(joint=1.2,target_route=True),
    'v2_joint_centroid':dict(joint=1.,target_route=True,location='centroid'),
    'v2_joint_nearest':dict(joint='nearest'),
    'v2_joint_tour':dict(joint='tour'),
    'v2_joint_tour_centroid':dict(joint='tour',location='centroid'),
    'v2_joint_tour_scan':dict(joint='tour',scan=1200),
    'v2_tour_skip':dict(joint='tour',known='skip'),
    'v2_tour_selective':dict(joint='tour',known='selective'),
    'v2_nearest_skip':dict(joint='nearest',known='skip'),
    'v2_nearest_selective':dict(joint='nearest',known='selective'),
    'v2_joint75_selective':dict(joint=.75,target_route=True,known='selective'),
    'v2_tour_selective_centroid':dict(joint='tour',known='selective',location='centroid'),
}


SELECTED_V2={3:'v2_tour_selective_centroid',4:'v2_tour_skip'}

from algorithms import ImprovedRobot, CONFIGS, SELECTED, estimate, path_order


class DiscoveryCover:
    """Conservative continuous disk coverage via complete square cells.

    Each retained cell intersects the radius-1800 disk. A measurement certifies
    its WHOLE cell only if its farthest corner is <=1000m away. Thus this is
    stronger than sampling points; no possible source is discarded by gridding.
    Used only for problem 3's omnidirectional sources.
    """
    def __init__(self,step=100):
        self.half=step/2
        self.cells=[]
        self.cache={}
        for x in range(-1800,1800,step):
            for y in range(-1800,1800,step):
                nearx=0 if x<=0<=x+step else min(abs(x),abs(x+step))
                neary=0 if y<=0<=y+step else min(abs(y),abs(y+step))
                if math.hypot(nearx,neary)<=1800:
                    self.cells.append((x+self.half,y+self.half))
        self.uncovered=(1<<len(self.cells))-1

    def contains(self,point,cell):
        return (abs(point[0]-cell[0])+self.half)**2+(abs(point[1]-cell[1])+self.half)**2 <= 1000**2-1e-6

    def gain(self,point):
        return (self.mask(point)&self.uncovered).bit_count()

    def mask(self,point):
        point=tuple(point)
        if point not in self.cache:
            mask=0
            for i,cell in enumerate(self.cells):
                if self.contains(point,cell): mask|=1<<i
            self.cache[point]=mask
        return self.cache[point]

    def record(self,point):
        self.uncovered&=~self.mask(point)

    def union(self,points):
        mask=0
        for point in points:mask|=self.mask(point)
        return mask


class AlternativeRobot(ImprovedRobot):
    def __init__(self,*args,variant=None,**kwargs):
        problem=kwargs.get('problem',args[1] if len(args)>1 else None)
        super().__init__(*args,config=dict(CONFIGS[SELECTED[problem]]),**kwargs)
        self.variant=variant or {}
        self.scan_points=[]
        self.discovery=DiscoveryCover(50 if self.variant.get('coverage')=='fixed' else 100) if problem==3 and self.variant.get('coverage') else None
        self.remaining=[]
        self.plan_station=None
        self.plan_channel=None

    def survey_stations(self):
        if not self.discovery or self.variant.get('coverage')=='fixed':
            self.remaining=super().survey_stations()
        else:
            # Dense candidate pool, not all points will be visited.
            self.remaining=[(float(x),float(y)) for x in range(-1800,1801,300) for y in range(-1800,1801,300)
                            if math.hypot(x,y)<=1900]
        return self.remaining

    def choose_station(self,remaining):
        if self.plan_station in remaining:
            return self.plan_station
        if not self.discovery:
            return super().choose_station(remaining)
        if not self.discovery.uncovered:
            remaining[:]=[self.position]
            return self.position
        if self.variant.get('coverage')=='fixed':
            for point in sorted(remaining,key=lambda p:distance(p,self.position),reverse=True):
                others=[p for p in remaining if p!=point]
                if self.discovery.uncovered & ~self.discovery.union(others)==0:
                    remaining.remove(point)
            return super().choose_station(remaining)
        scan_cost=6*max(1,20-len(self.cleared))
        return max(remaining,key=lambda p:self.discovery.gain(p)/(scan_cost+distance(p,self.position)/5))

    def survey(self,point):
        if self.discovery and not self.discovery.uncovered:
            return
        if self.variant.get('known'):
            channels=[]
            for c in range(1,21):
                if c in self.cleared:continue
                if c not in self.polygons:
                    channels.append(c)
                elif self.variant['known']=='selective':
                    center,radius=center_radius(self.polygons[c])
                    previous,_=self.good[c][-1]
                    a=(previous[0]-center[0],previous[1]-center[1])
                    b=(point[0]-center[0],point[1]-center[1])
                    sine=abs(a[0]*b[1]-a[1]*b[0])/max(1,math.hypot(*a)*math.hypot(*b))
                    if radius>19.5 and distance(previous,point)>100 and distance(center,point)<1350 and (sine>.15 or distance(center,point)<150):
                        channels.append(c)
            if self.channel in channels:
                channels.remove(self.channel);channels.insert(0,self.channel)
            for c in channels:self.measure(point,c)
        else:
            super().survey(point)
        self.scan_points.append(point)
        if self.discovery:
            self.discovery.record(point)

    def after_resolve(self):
        super().after_resolve()
        threshold=self.variant.get('scan')
        if self.variant.get('replace') and self.discovery:
            if not self.remaining:return
            uncovered=self.discovery.uncovered & ~self.discovery.mask(self.position)
            can_replace=any(uncovered & ~self.discovery.union([q for q in self.remaining if q!=p])==0 for p in self.remaining)
            if not can_replace:return
        else:
            if not threshold or min((distance(self.position,p) for p in self.scan_points),default=math.inf)<threshold:
                return
        if self.discovery and self.discovery.gain(self.position)==0:
            return
        # All still-undiscovered channels have been measured at every scan;
        # channels already known need not be re-scanned for discovery coverage.
        for channel in range(1,21):
            if channel not in self.polygons and channel not in self.cleared:
                self.measure(self.position,channel)
        self.scan_points.append(self.position)
        if self.discovery:
            self.discovery.record(self.position)

    def refresh_pending(self,pending):
        pending=set(self.polygons)-self.cleared
        if pending and self.remaining and self.variant.get('joint'):
            if self.variant['joint'] in ('nearest','tour'):
                centers={c:estimate(self.good[c],self.polygons[c])[0] for c in sorted(pending)}
                points=list(self.remaining)+list(centers.values())
                first=(path_order(points,self.position)[0] if self.variant['joint']=='tour'
                       else min(points,key=lambda p:distance(p,self.position)))
                if first in self.remaining:
                    self.plan_station=first
                    return set()
                self.plan_station=None
                self.plan_channel=next(c for c,p in centers.items() if p==first)
                return pending
            nearest_target=min(distance(estimate(self.good[c],self.polygons[c])[0],self.position) for c in pending)
            nearest_station=min(distance(p,self.position) for p in self.remaining)
            if nearest_station<nearest_target*self.variant['joint']:
                return set()
        return pending

    def choose_pending(self,pending):
        pending=set(pending)-self.cleared
        if not pending:
            raise RuntimeError('Empty pending selection')
        if self.plan_channel in pending:
            return self.plan_channel
        if not self.variant.get('target_route'):
            return super().choose_pending(pending)
        centers={c:estimate(self.good[c],self.polygons[c])[0] for c in sorted(pending)}
        first=path_order(list(centers.values()),self.position)[0]
        return next(c for c in centers if centers[c]==first)

    def location(self,channel):
        if self.variant.get('location')=='centroid':
            point=centroid(self.polygons[channel])
            return point,max(distance(point,p) for p in self.polygons[channel])
        return estimate(self.good[channel],self.polygons[channel])

    def first_probe(self,channel,center):
        old,bearing=self.good[channel][-1]
        rule=self.variant.get('approach')
        if rule!='information':
            return tuple(old[k]+rule*(center[k]-old[k]) for k in (0,1))
        # One-step approximate information/travel objective. Use the feasible
        # polygon as prior, and model angular uncertainty at predicted range.
        theta=math.radians(bearing)
        u=(math.cos(theta),math.sin(theta))
        d=max(50,distance(old,center))
        candidates=[]
        for forward in (.25,.5,.8,1.):
            for lateral in (-150,-60,0,60,150):
                p=(old[0]+forward*d*u[0]-lateral*u[1],old[1]+forward*d*u[1]+lateral*u[0])
                if any(distance(p,q)<2 for q in self.attempts[channel]):continue
                r=max(20,distance(p,center))
                sine=abs((old[0]-center[0])*(p[1]-center[1])-(old[1]-center[1])*(p[0]-center[0]))/max(1,d*r)
                uncertainty=min(d,math.radians(1.005)*math.hypot(d,r)/max(.02,sine))
                cost=(distance(self.position,p)+r)/5+uncertainty*.3
                candidates.append((cost,p))
        return min(candidates)[1] if candidates else center

    def resolve(self,channel):
        if not self.variant.get('approach') and not self.variant.get('location'):
            return super().resolve(channel)
        if channel in self.cleared:return
        if distance(self.position,self.good[channel][-1][0])>60:
            self.measure(self.position,channel)
        for step in range(8):
            if channel in self.cleared:return
            safe,radius=center_radius(self.polygons[channel])
            if radius<=19.5:
                if self.clear(safe,channel):return
                raise RuntimeError('Certified clear failed')
            point,_=self.location(channel)
            if len(self.good[channel])==1:
                if not self.variant.get('approach'):
                    return super().resolve(channel)
                point=self.first_probe(channel,point)
                # On no-signal, fall back to shrinking cross-bearing probes.
                if self.clear(point,channel):return
                kind=self.measure(point,channel)
                if kind=='no_signal':return super().resolve(channel)
            else:
                if any(distance(point,p)<2 for p in self.attempts[channel]):
                    return super().resolve(channel)
                if self.clear(point,channel):return
                self.measure(point,channel)
        return super().resolve(channel)


def alternative_class(name):
    class Configured(AlternativeRobot):
        def __init__(self,*args,**kwargs):
            problem=kwargs.get('problem',args[1] if len(args)>1 else None)
            selected=SELECTED_V2[problem] if name=='optimized_v2' else name
            super().__init__(*args,variant=VARIANTS[selected],**kwargs)
    return Configured
