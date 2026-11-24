"""Alternative strategies using observations only, never simulator ground truth."""
import math
from robot_client import Robot, distance, center_radius, stations


def inside(poly, point):
    signs = []
    for a, b in zip(poly, poly[1:] + poly[:1]):
        signs.append((b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0]))
    return min(signs) >= -1e-6 or max(signs) <= 1e-6


def closest_segment(point, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    norm = dx*dx+dy*dy
    t = max(0, min(1, ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/norm)) if norm else 0
    return a[0]+t*dx, a[1]+t*dy


def estimate(observations, poly):
    # Bearing line least squares; two reweightings approximate angular residuals.
    estimate, radius = center_radius(poly)
    if len(observations) < 2:
        return estimate, radius
    for iteration in range(3):
        aa = ab = bb = ac = bc = 0.0
        for point, bearing in observations:
            theta = math.radians(bearing)
            a, b = -math.sin(theta), math.cos(theta)
            c = a*point[0]+b*point[1]
            weight = 1 if iteration == 0 else 1/max(30, distance(point, estimate))**2
            aa += weight*a*a
            ab += weight*a*b
            bb += weight*b*b
            ac += weight*a*c
            bc += weight*b*c
        det = aa*bb-ab*ab
        if det <= 1e-14 * max(aa*bb, 1e-15):
            break
        candidate = ((ac*bb-bc*ab)/det, (bc*aa-ac*ab)/det)
        if not inside(poly, candidate):
            candidate = min((closest_segment(candidate, a, b)
                             for a, b in zip(poly, poly[1:]+poly[:1])),
                            key=lambda q: distance(q, candidate))
        estimate = candidate
    return estimate, max(distance(estimate, q) for q in poly)


def triangular_cover(spacing=990, offset=(0, 0)):
    # All vertices of every lattice triangle intersecting the target disk.
    # Edge length <1000 implies each vertex is within reception distance of
    # every source in that triangle. Convex containment guarantees at least
    # one vertex in each possible directional half-plane. No heading sampling.
    def point(i, j):
        return (spacing*(i+j/2)+offset[0], spacing*math.sqrt(3)/2*j+offset[1])
    selected = set()
    for i in range(-5, 6):
        for j in range(-5, 6):
            a, b, c, d = point(i,j), point(i+1,j), point(i,j+1), point(i+1,j+1)
            for triangle in ([a,b,c], [b,d,c]):
                intersects = inside(triangle, (0,0)) or min(
                    distance((0,0), closest_segment((0,0), p, q))
                    for p,q in zip(triangle, triangle[1:]+triangle[:1])) <= 1800+1e-8
                if intersects:
                    selected.update(triangle)
    return sorted(selected)


def path_order(points, start):
    remaining, route, position = list(points), [], start
    while remaining:
        best = min(remaining, key=lambda p: distance(p, position))
        remaining.remove(best)
        route.append(best)
        position = best
    # Open-path 2-opt: the endpoint is free; include suffix reversals.
    for _ in range(8):
        improved = False
        for i in range(len(route)-1):
            prev = start if i == 0 else route[i-1]
            for j in range(i+1, len(route)):
                old = distance(prev, route[i])
                new = distance(prev, route[j])
                if j+1 < len(route):
                    old += distance(route[j], route[j+1])
                    new += distance(route[i], route[j+1])
                if new < old-1e-6:
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
        if not improved:
            break
    return route


CONFIGS = {
    'baseline': {},
    'current': dict(method='current'),
    'least_squares': dict(method='ls', cross=150),
    'reuse': dict(method='ls', cross=150, reuse=True),
    'compact': dict(method='ls', cross=150, reuse=True, cover=True),
    'route': dict(method='ls', cross=150, reuse=True, cover=True, route=True),
}
for cross in (60, 100, 220):
    CONFIGS[f'route_b{cross}'] = dict(method='ls', cross=cross, reuse=True, cover=True, route=True)
for margin in (0, 60, 120):
    CONFIGS[f'route_forward{margin}'] = dict(method='ls', cross=100, forward=margin,
                                         reuse=True, cover=True, route=True)
for name, extra in {
    'adaptive': {},
    'adaptive_b100': {'cross':100},
    'adaptive_b220': {'cross':220},
    'adaptive_no_reuse': {'reuse':False},
    'adaptive_route': {'route':True},
    'adaptive_ring1250': {'ring':1250},
    'adaptive_ring1450': {'ring':1450},
    'adaptive_forward60': {'forward':60},
}.items():
    CONFIGS[name] = dict(method='ls',cross=150,reuse=True,cover=True,
                         spacing=999,offset=(0,149.85),shrink=True)
    CONFIGS[name].update(extra)


SELECTED = {3: 'adaptive_ring1250', 4: 'adaptive_route'}
CONFIGS['optimized'] = {'selection': SELECTED}


class ImprovedRobot(Robot):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or CONFIGS['route']

    def survey_stations(self):
        if not self.config.get('cover'):
            return stations(self.problem)
        if self.problem == 3:
            # Center + six ring points. Boundary worst-case distance <1000m;
            # the central triangles have covering radius ring/sqrt(3).
            radius=self.config.get('ring',1140)
            return [(0.,0.)]+[(radius*math.cos(i*math.pi/3),radius*math.sin(i*math.pi/3)) for i in range(6)]
        return triangular_cover(self.config.get('spacing',990),self.config.get('offset',(0,0)))

    def choose_station(self, remaining):
        if self.config.get('route'):
            return path_order(remaining, self.position)[0]
        return super().choose_station(remaining)

    def after_resolve(self):
        if not self.config.get('reuse'):
            return
        # Reuse each reached location for nearby known targets only. This never
        # consults undiscovered channels' locations or local generator state.
        pending = sorted(set(self.polygons)-self.cleared)
        for channel in pending:
            center, radius = center_radius(self.polygons[channel])
            old, _ = self.good[channel][-1]
            if distance(old, self.position) < 100 or distance(center, self.position) > 1350:
                continue
            u = (old[0]-center[0], old[1]-center[1])
            v = (self.position[0]-center[0], self.position[1]-center[1])
            sine = abs(u[0]*v[1]-u[1]*v[0])/max(1, math.hypot(*u)*math.hypot(*v))
            if sine > 0.15 or distance(center,self.position) < 150:
                self.measure(self.position, channel)

    def resolve(self, channel):
        if channel in self.cleared:
            return
        old, _ = self.good[channel][-1]
        if distance(self.position, old) > 60:
            self.measure(self.position, channel)
        if self.config.get('method') == 'current':
            return super().resolve(channel)
        for step in range(12):
            if channel in self.cleared:
                return
            poly = self.polygons[channel]
            safe, safe_radius = center_radius(poly)
            if safe_radius <= 19.5:
                if self.clear(safe, channel):
                    return
                raise RuntimeError('Certified clear unexpectedly failed.')
            center, radius = estimate(self.good[channel], poly)
            previous, bearing = self.good[channel][-1]
            theta = math.radians(bearing)
            ux, uy = math.cos(theta), math.sin(theta)
            if len(self.good[channel]) < 2:
                cross = self.config.get('cross', 150)
                forward = self.config.get('forward', 150)
                # Select the lateral side that requires less travel from here.
                options = [(previous[0]+forward*ux+s*cross*uy,
                            previous[1]+forward*uy-s*cross*ux) for s in (-1,1)]
                point = min(options,key=lambda p:distance(p,self.position))
                if any(distance(point,p)<3 for p in self.attempts[channel]):
                    point = options[1-options.index(point)]
                if self.config.get('shrink'):
                    # On a directional coverage boundary, both large probes
                    # can fail. Shrink rather than repeating known no-signal
                    # probes or immediately enumerating a huge clear grid.
                    options=[]
                    for level in range(6):
                        scale=0.5**level
                        pair=[(previous[0]+scale*(forward*ux+s*cross*uy),
                               previous[1]+scale*(forward*uy-s*cross*ux)) for s in (-1,1)]
                        options.extend(sorted(pair,key=lambda p:distance(p,self.position)))
                    fresh=[p for p in options if all(distance(p,q)>=2 for q in self.attempts[channel])]
                    if not fresh:
                        if self.clear(previous,channel):
                            return
                    else:
                        point=fresh[0]
                self.measure(point, channel)
            elif step % 3 != 2:
                # A failed optical attempt costs only 3s, much less than a long
                # detour to a conservative polygon center. Immediately measure
                # at the same location on failure for a strong new bearing.
                if not any(distance(center,p)<2 for p in self.attempts[channel]):
                    if self.clear(center, channel):
                        return
                    self.measure(center, channel)
                else:
                    step = 2
            if step % 3 == 2 and channel not in self.cleared:
                center, radius = estimate(self.good[channel], self.polygons[channel])
                previous, bearing = self.good[channel][-1]
                theta = math.radians(bearing)
                ux,uy=math.cos(theta),math.sin(theta)
                lateral = min(100,max(30,radius*0.2))*(-1 if step%2 else 1)
                point=(center[0]-50*ux-lateral*uy,center[1]-50*uy+lateral*ux)
                self.measure(point, channel)
        return super().resolve(channel)


def robot_class(name):
    if name.startswith('p3_v4_') or name=='optimized_v4':
        from algorithms_p3_v4 import region_robot_class
        return region_robot_class(name)
    if name.startswith('p3_') or name=='optimized_v3':
        from algorithms_p3 import problem_three_class
        return problem_three_class(name)
    if name.startswith('v2_') or name=='optimized_v2':
        from algorithms_v2 import alternative_class
        return alternative_class(name)
    if name == 'baseline':
        return Robot
    if name not in CONFIGS:
        raise ValueError(f'Unknown algorithm: {name}')
    class Configured(ImprovedRobot):
        def __init__(self,*args,**kwargs):
            problem = kwargs.get('problem', args[1] if len(args) > 1 else None)
            selected = SELECTED[problem] if name == 'optimized' else name
            super().__init__(*args,config=CONFIGS[selected],**kwargs)
    return Configured


# Keep first-round configs and the 'optimized' alias frozen for comparisons.
from algorithms_v2 import VARIANTS, SELECTED_V2
CONFIGS.update(VARIANTS)
CONFIGS['optimized_v2']={'selection':SELECTED_V2}
from algorithms_p3_config import P3_CONFIGS, SELECTED_P3
CONFIGS.update(P3_CONFIGS)
CONFIGS['optimized_v3']={'problem3':SELECTED_P3,'problem4':'unchanged optimized_v2'}
from algorithms_p3_v4_config import V4_CONFIGS, SELECTED_V4
CONFIGS.update(V4_CONFIGS)
CONFIGS['optimized_v4']={'problem3':SELECTED_V4,'problem4':'unchanged optimized_v2'}
