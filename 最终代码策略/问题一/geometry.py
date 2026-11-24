"""二维有界方位误差几何；距离单位m，输入角度单位degree。"""
from itertools import combinations
import math
import random
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

ANGLE_ERROR = 1.01


def unit(deg):
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def disk_outer(center=(0, 0), radius=1800, sides=64):
    a = np.arange(sides) * (2 * np.pi / sides)
    return np.asarray(center) + radius / math.cos(math.pi / sides) * np.c_[np.cos(a), np.sin(a)]


def clip(poly, normal, bound, eps=1e-8):
    """以n·p<=b裁剪凸多边形；eps只处理浮点边界。"""
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    if len(poly) == 0:
        return poly
    normal = np.asarray(normal, dtype=float)
    values = poly @ normal - bound
    result = []
    for i, p in enumerate(poly):
        j = (i + 1) % len(poly)
        q = poly[j]
        inside_p, inside_q = values[i] <= eps, values[j] <= eps
        if inside_p:
            result.append(p)
        if inside_p != inside_q:
            denominator = values[i] - values[j]
            if abs(denominator) > 1e-15:
                result.append(p + np.clip(values[i] / denominator, 0, 1) * (q - p))
    if not result:
        return np.empty((0, 2))
    out = [result[0]]
    for p in result[1:]:
        if np.linalg.norm(p - out[-1]) > 1e-7:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) < 1e-7:
        out.pop()
    return np.asarray(out)


def wedge_planes(station, bearing, delta=ANGLE_ERROR):
    station = np.asarray(station)
    low, high = unit(bearing - delta), unit(bearing + delta)
    normals = np.array([[low[1], -low[0]], [-high[1], high[0]]])
    return [(n, float(n @ station)) for n in normals]


def observe(poly, station, bearing, delta=ANGLE_ERROR, range_bound=1500):
    out = np.asarray(poly)
    for n, b in wedge_planes(station, bearing, delta):
        out = clip(out, n, b)
    if range_bound is not None:
        # 外切半平面交严格包含接收圆盘，不以内接多边形删真值。
        for a in np.arange(32) * 360 / 32:
            n = unit(a)
            out = clip(out, n, float(n @ station + range_bound))
    return out


def contains(poly, point, eps=1e-6):
    p, x = np.asarray(poly), np.asarray(point)
    if not len(p):
        return False
    if len(p) == 1:
        return np.linalg.norm(x - p[0]) <= eps
    if len(p) == 2:
        d = p[1] - p[0]
        t = np.clip((x - p[0]) @ d / max(d @ d, 1e-30), 0, 1)
        return np.linalg.norm(x - p[0] - t * d) <= eps
    edges = np.roll(p, -1, axis=0) - p
    rel = x - p
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    return bool(np.all(cross >= -eps) or np.all(cross <= eps))


def diameter(poly):
    p = np.asarray(poly)
    if not len(p):
        raise ValueError('空集无定位直径')
    dist2 = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(dist2), dist2.shape)
    return float(math.sqrt(dist2[i, j])), (p[i].copy(), p[j].copy())


def circle3(a, b, c):
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    mat = 2 * np.array([b-a, c-a])
    if abs(np.linalg.det(mat)) <= 1e-10:
        pair = max(combinations([a, b, c], 2), key=lambda pair: np.linalg.norm(pair[0]-pair[1]))
        center = (pair[0] + pair[1]) / 2
    else:
        center = a + np.linalg.solve(mat, [np.sum((b-a)**2), np.sum((c-a)**2)])
    return center, max(float(np.linalg.norm(x-center)) for x in [a, b, c])


def minimum_enclosing_circle(poly):
    p = [np.asarray(x, dtype=float) for x in poly]
    if not p:
        raise ValueError('空集不能计算包围圆')
    random.Random(1207).shuffle(p)
    center, radius = p[0].copy(), 0.
    for i, a in enumerate(p):
        if np.linalg.norm(a-center) <= radius + 1e-7:
            continue
        center, radius = a.copy(), 0.
        for j, b in enumerate(p[:i]):
            if np.linalg.norm(b-center) <= radius + 1e-7:
                continue
            center, radius = (a+b)/2, float(np.linalg.norm(a-b)/2)
            for c in p[:j]:
                if np.linalg.norm(c-center) > radius + 1e-7:
                    center, radius = circle3(a, b, c)
    # 上取最终距离，避免浮点算法把覆盖半径低报。
    radius = max(radius, max(float(np.linalg.norm(x-center)) for x in p))
    return center, radius


def nearest_clear_point(poly, current, radius=19.9):
    """在所有顶点的radius圆盘交内求最近点；枚举弧投影和交点。"""
    p, x = np.asarray(poly), np.asarray(current)
    center, mec_radius = minimum_enclosing_circle(p)
    if mec_radius > radius + 1e-8:
        return None

    def feasible(q):
        return np.max(np.linalg.norm(p - q, axis=1)) <= radius + 1e-7

    if feasible(x):
        return x.copy()
    candidates = [center]
    for v in p:
        d = np.linalg.norm(x-v)
        if d > 1e-12:
            q = v + radius * (x-v) / d
            if feasible(q):
                candidates.append(q)
    for a, b in combinations(p, 2):
        d = float(np.linalg.norm(a-b))
        if 1e-9 < d <= 2 * radius:
            mid = (a+b)/2
            v = np.array([a[1]-b[1], b[0]-a[0]]) / d
            height = math.sqrt(max(0, radius**2 - d**2/4))
            for q in (mid+height*v, mid-height*v):
                if feasible(q):
                    candidates.append(q)
    q = min(candidates, key=lambda q: np.linalg.norm(q-x))
    assert feasible(q)
    return q.copy()


def q1_intersection(stations, bearings, delta=1.):
    planes = [plane for s, a in zip(stations, bearings) for plane in wedge_planes(s, a, delta)]
    if len(stations) != len(bearings):
        raise ValueError('站点数与示向度数不一致')
    if not planes:
        return {'status': 'unbounded', 'diameter': math.inf}
    A, b = np.array([n for n, _ in planes]), np.array([v for _, v in planes])
    opts = {'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9}
    feasible = linprog([0, 0], A_ub=A, b_ub=b, bounds=[(None, None)]*2, method='highs', options=opts)
    if feasible.status == 2:
        return {'status': 'empty', 'diameter': None}
    if not feasible.success:
        raise RuntimeError(feasible.message)
    vertices = []
    for objective in ([1, 0], [-1, 0], [0, 1], [0, -1]):
        ext = linprog(objective, A_ub=A, b_ub=b, bounds=[(None, None)]*2, method='highs', options=opts)
        if ext.status == 3:
            return {'status': 'unbounded', 'diameter': math.inf}
        if not ext.success:
            raise RuntimeError(ext.message)
        vertices.append(ext.x)
    for i, j in combinations(range(len(planes)), 2):
        M = A[[i, j]]
        if abs(np.linalg.det(M)) > 1e-12:
            q = np.linalg.solve(M, b[[i, j]])
            if np.all(A @ q <= b + 1e-6):
                vertices.append(q)
    p = np.unique(np.round(np.asarray(vertices), 9), axis=0)
    if len(p) > 2:
        try:
            p = p[ConvexHull(p).vertices]
        except QhullError:
            _, ends = diameter(p)
            p = np.array(ends)
    D, pair = diameter(p)
    center, radius = minimum_enclosing_circle(p)
    return {'status': 'bounded', 'vertices': p.tolist(), 'diameter': D,
            'diameter_endpoints': [x.tolist() for x in pair], 'mec_center': center.tolist(),
            'mec_radius': radius, 'diameter_circle_covers': radius <= D/2+1e-7}


def principal_frame(poly):
    p = np.asarray(poly)
    center = p.mean(axis=0)
    if len(p) < 2:
        return center, np.array([1., 0.]), 0.
    _, ends = diameter(p)
    direction = ends[1] - ends[0]
    length = np.linalg.norm(direction)
    return center, direction / max(length, 1e-12), float(length)


def q2_safe_candidates(station, bearing, delta=ANGLE_ERROR, step=50):
    u = unit(bearing); v = np.array([-u[1], u[0]])
    height = 1500 * math.tan(math.radians(delta))
    local = []
    for a in np.arange(500, 1000.01, step):
        for b in np.arange(-600, 600.01, step):
            if abs(b) < 10:
                continue
            if max(math.hypot(a, b), math.hypot(a-1500, b-height), math.hypot(a-1500, b+height)) <= 1000:
                local.append(np.asarray(station) + a*u + b*v)
    return np.asarray(local)


def observation_candidates(poly, current, observations, question=3):
    center, axis, length = principal_frame(poly)
    perpendicular = np.array([-axis[1], axis[0]])
    offset = min(300., max(35., length * .22))
    points = []
    for shift in (-.15, 0, .15):
        for side in (-1, 1):
            points.append(center + shift * length * axis + side * offset * perpendicular)
    if observations:
        first_s, first_angle = observations[0]
        u = unit(first_angle); v = np.array([-u[1], u[0]])
        points.extend(np.asarray(first_s) + 750*u + side*200*v for side in (-1, 1))
    old = [np.asarray(s) for s, _ in observations]
    return [p for p in points if all(np.linalg.norm(p-s) > 2 for s in old)]


def sampled_posterior_score(poly, station, current, delta=ANGLE_ERROR):
    """选点启发式：有限位置/角误差采样，不作为最坏上界证书。"""
    p = np.asarray(poly)
    samples = list(p) + [p.mean(axis=0)]
    worst = 0.
    for g in samples:
        if np.linalg.norm(g-station) <= 5:
            continue
        angle = math.degrees(math.atan2(g[1]-station[1], g[0]-station[0]))
        for error in (-delta, 0., delta):
            out = observe(p, station, angle+error, delta, range_bound=None)
            if len(out):
                _, r = minimum_enclosing_circle(out)
                worst = max(worst, r)
    # 统一以秒计：移动时间+测向5s+后续清除前可能往返的半径代价。
    return float(np.linalg.norm(station-current)/5 + 5 + 2*worst/5), worst


def optical_cover(poly, first_station, first_bearing, failed=()):
    """只删除与凸多边形不相交的格子；每个剩余格中心覆盖整格。"""
    u = unit(first_bearing); v = np.array([-u[1], u[0]])
    origin = np.asarray(first_station)
    local = np.c_[(poly-origin) @ u, (poly-origin) @ v]
    # 首次正信号的真实距离不超过1500，收紧外接圆多边形的微小外扩。
    local = clip(local, [1, 0], 1500.)
    lower, upper = local.min(axis=0), local.max(axis=0)
    counts = np.maximum(1, np.ceil((upper-lower)/25).astype(int))
    dx = (upper-lower) / counts
    points = []
    for j in range(counts[1]):
        indices = range(counts[0]) if j % 2 == 0 else reversed(range(counts[0]))
        for i in indices:
            lo = lower + dx * [i, j]; hi = lo + dx
            q = local.copy()
            for n, b in (([1, 0], hi[0]), ([-1, 0], -lo[0]), ([0, 1], hi[1]), ([0, -1], -lo[1])):
                q = clip(q, n, b)
            if not len(q):
                continue
            center = (lo+hi)/2
            world = origin + center[0]*u + center[1]*v
            if any(np.linalg.norm(world-f) < 1e-5 for f in failed):
                continue
            points.append(world)
    return points
