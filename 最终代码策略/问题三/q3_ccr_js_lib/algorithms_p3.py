"""Third-question-only alternatives. Observations in; no simulator state read."""
import math
from algorithms import estimate, path_order
from algorithms_v2 import AlternativeRobot, VARIANTS, SELECTED_V2, centroid
from algorithms_p3_config import P3_CONFIGS, SELECTED_P3
from robot_client import distance, clip, center_radius


def convex_hull(points):
    points=sorted(set(points))
    if len(points)<=2:return points
    def turn(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower=[]
    for p in points:
        while len(lower)>1 and turn(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(points):
        while len(upper)>1 and turn(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def exclude_no_signal(poly,position,sides=16):
    # Subtract an INSCRIBED regular polygon from the forbidden radius-1000
    # disk. Keeping its outside is conservative; never treat no_signal as a
    # direction in problem 4. The hull may reintroduce excluded locations,
    # which weakens information but preserves containment of the true source.
    current=list(poly)
    outside=[]
    apothem=1000*math.cos(math.pi/sides)-1e-5
    for k in range(sides):
        a,b=math.cos(2*math.pi*k/sides),math.sin(2*math.pi*k/sides)
        c=a*position[0]+b*position[1]+apothem
        outside.extend(clip(current,-a,-b,-c))
        current=clip(current,a,b,c)
        if not current:break
    return convex_hull(outside) if outside else poly


def improved_path(points,start,mode='multi'):
    """Exact open Held-Karp for <=11 nodes; otherwise bounded route heuristics."""
    n=len(points)
    if n<2:return list(points)
    d=[[distance(a,b) for b in points] for a in list(points)+[start]]
    if mode=='exact' and n<=11:
        cost={};parent={}
        for j in range(n):cost[1<<j,j]=d[n][j]
        for mask in range(1,1<<n):
            ends=[j for j in range(n) if mask&(1<<j)]
            for j in ends:
                previous=mask^(1<<j)
                if not previous:continue
                best,i=min((cost[previous,i]+d[i][j],i) for i in ends if i!=j)
                cost[mask,j]=best;parent[mask,j]=i
        mask=(1<<n)-1
        j=min(range(n),key=lambda j:cost[mask,j])
        route=[]
        while mask:
            route.append(j)
            prev=parent.get((mask,j))
            mask^=1<<j
            j=prev
        return [points[j] for j in reversed(route)]
    def optimize(route):
        for _ in range(10):
            improved=False
            for i in range(n-1):
                previous=n if i==0 else route[i-1]
                for j in range(i+1,n):
                    old=d[previous][route[i]]
                    new=d[previous][route[j]]
                    if j+1<n:
                        old+=d[route[j]][route[j+1]]
                        new+=d[route[i]][route[j+1]]
                    if new<old-1e-7:
                        route[i:j+1]=reversed(route[i:j+1]);improved=True
            if not improved:break
        return route
    starts=sorted(range(n),key=lambda j:d[n][j])[:min(n,5)]
    routes=[]
    for first in starts:
        route=[first];left=set(range(n))-{first}
        while left:
            if mode=='insertion':
                options=[]
                for node in sorted(left):
                    for k in range(len(route)+1):
                        prev=n if k==0 else route[k-1]
                        extra=d[prev][node]
                        if k<len(route):extra+=d[node][route[k]]-d[prev][route[k]]
                        options.append((extra,node,k))
                _,node,k=min(options);route.insert(k,node)
            else:
                node=min(left,key=lambda j:(d[route[-1]][j],j));route.append(node)
            left.remove(node)
        routes.append(optimize(route))
    route=min(routes,key=lambda r:d[n][r[0]]+sum(d[a][b] for a,b in zip(r,r[1:])))
    return [points[j] for j in route]


class ProblemThreeRobot(AlternativeRobot):
    def __init__(self,*args,p3_config=None,**kwargs):
        problem=kwargs.get('problem',args[1] if len(args)>1 else None)
        super().__init__(*args,variant=dict(VARIANTS[SELECTED_V2[problem]]),**kwargs)
        self.p3_config=(p3_config or {}) if problem==3 else {}
        self.negatives={}
        self.layout_chosen=False
        if self.p3_config.get('ring'):self.config['ring']=self.p3_config['ring']

    def measure(self,point,channel):
        kind=super().measure(point,channel)
        sides=self.p3_config.get('negative')
        if not (sides or self.p3_config.get('bisector')) or channel in self.cleared:return kind
        if kind=='no_signal':
            previous=self.negatives.setdefault(channel,[])
            if all(distance(self.position,q)>5 for q in previous):previous.append(self.position)
        if channel in self.polygons:
            poly=self.polygons[channel]
            for q in self.negatives.get(channel,[]):
                if sides:poly=exclude_no_signal(poly,q,sides)
                if self.p3_config.get('bisector'):
                    # Positive p and negative q share one unknown fixed radius:
                    # |source-p| <= R < |source-q|. Squaring cancels |source|^2
                    # and gives an exact linear, conservative half-plane.
                    for p,_ in self.good[channel]:
                        a,b=2*(q[0]-p[0]),2*(q[1]-p[1])
                        c=q[0]**2+q[1]**2-p[0]**2-p[1]**2+1e-4
                        cut=clip(poly,a,b,c)
                        if cut:poly=cut
            self.polygons[channel]=poly
        return kind

    def clear(self,point,channel):
        if self.p3_config.get('disk_clear') and channel in self.polygons:
            center,radius=center_radius(self.polygons[channel])
            if radius<19.4:
                d=distance(self.position,center)
                allowed=19.4-radius
                if d>allowed:
                    point=tuple(center[k]+(self.position[k]-center[k])*allowed/d for k in (0,1))
                else:point=self.position
        return super().clear(point,channel)

    def planned_centers(self,pending):
        return {c:(centroid(self.polygons[c]) if self.p3_config.get('planning')=='centroid'
                   else estimate(self.good[c],self.polygons[c])[0]) for c in sorted(pending)}

    def refresh_pending(self,pending):
        if self.p3_config.get('rotate') and not self.layout_chosen:
            # Only rotate the entire six-point ring immediately after the
            # origin scan, before visiting any ring point. Coverage is rotation
            # invariant; choose using discovered observations, never source truth.
            self.layout_chosen=True
            if len(self.scan_points)==1 and len(self.remaining)==6:
                centers=list(self.planned_centers(set(self.polygons)-self.cleared).values())
                candidates=[]
                radius=self.config['ring']
                for k in range(self.p3_config['rotate']):
                    offset=k*math.pi/(3*self.p3_config['rotate'])
                    ring=[(radius*math.cos(i*math.pi/3+offset),radius*math.sin(i*math.pi/3+offset)) for i in range(6)]
                    route=improved_path(ring+centers,self.position,self.p3_config.get('route','multi'))
                    length=sum(distance(a,b) for a,b in zip([self.position]+route,route))
                    candidates.append((length,k,ring))
                self.remaining[:]=min(candidates)[2]
        if not self.p3_config.get('route') and not self.p3_config.get('planning'):
            return super().refresh_pending(pending)
        pending=set(self.polygons)-self.cleared
        if not pending:return pending
        centers=self.planned_centers(pending)
        points=list(self.remaining)+list(centers.values())
        route=(improved_path(points,self.position,self.p3_config['route']) if self.p3_config.get('route')
               else path_order(points,self.position))
        first=route[0]
        if first in self.remaining:
            self.plan_station=first
            return set()
        self.plan_station=None
        self.plan_channel=next(c for c,p in centers.items() if p==first)
        return pending

    def active_probe(self,channel):
        old,bearing=self.good[channel][-1]
        center=centroid(self.polygons[channel])
        if self.p3_config['probe']=='midpoint':return center
        if self.p3_config['probe']=='halfway':return tuple((old[k]+center[k])/2 for k in (0,1))
        theta=math.radians(bearing)
        u=(math.cos(theta),math.sin(theta))
        radial=[max(5,(p[0]-old[0])*u[0]+(p[1]-old[1])*u[1]) for p in self.polygons[channel]]
        low,high=min(radial),max(radial)
        samples=[(old[0]+r*u[0],old[1]+r*u[1]) for r in [low+(high-low)*t for t in (.1,.3,.5,.7,.9)]]
        r=distance(center,old)
        options=[]
        for forward in (.25,.5,.75,1.):
            for side in (-250,-125,-60,60,125,250):
                p=(old[0]+forward*r*u[0]-side*u[1],old[1]+forward*r*u[1]+side*u[0])
                future=0
                for source in samples:
                    a,b=distance(source,old),distance(source,p)
                    cross=abs((old[0]-source[0])*(p[1]-source[1])-(old[1]-source[1])*(p[0]-source[0]))
                    sine=cross/max(1,a*b)
                    uncertainty=min(high-low,math.radians(1.005)*math.hypot(a,b)/max(.01,sine))
                    future+=b+self.p3_config['probe']*uncertainty
                score=distance(self.position,p)+future/len(samples)
                options.append((score,p))
        return min(options)[1]

    def resolve(self,channel):
        if self.p3_config.get('probe') and channel not in self.cleared:
            if distance(self.position,self.good[channel][-1][0])>60:
                self.measure(self.position,channel)
            if channel not in self.cleared and len(self.good[channel])==1:
                self.measure(self.active_probe(channel),channel)
        return super().resolve(channel)


def problem_three_class(name):
    class Configured(ProblemThreeRobot):
        def __init__(self,*args,**kwargs):
            problem=kwargs.get('problem',args[1] if len(args)>1 else None)
            if problem!=3 and name!='optimized_v3':
                raise ValueError('p3_* candidates are for problem 3 only')
            selected=SELECTED_P3 if name=='optimized_v3' else name
            super().__init__(*args,p3_config=P3_CONFIGS[selected] if problem==3 else {},**kwargs)
    return Configured
