"""第二问：利用首次可接收事件与未知半径的耦合，求精确稳健候选域。"""
from pathlib import Path
import argparse
import json
import math
import numpy as np
from geometry import unit, observe, minimum_enclosing_circle, clip


def reception_region(a, b, delta=1., near=5.):
    """首次direction给r>near；取闭包不改变连续最坏约束。有Omega信息时仍充分。"""
    z = a*a+b*b
    d = math.radians(delta)
    inner = a*math.cos(d)-abs(b)*math.sin(d)
    return z+near**2-2*near*inner <= 1000**2+1e-8 and z <= 2000*inner+1e-8


def bearing_interval(poly, station, delta):
    angles = np.arctan2(poly[:, 1]-station[1], poly[:, 0]-station[0])
    ref = math.atan2(poly[:, 1].mean()-station[1], poly[:, 0].mean()-station[0])
    angles = ref + np.angle(np.exp(1j*(angles-ref)))
    return np.degrees(angles.min())-delta, np.degrees(angles.max())+delta


def minimax_bounds(station, step=1., delta=1.):
    # 外包三角形包含真实首测扇形，内接扇形多边形完全属于真实首测域。
    h = 1500*math.tan(math.radians(delta))
    outer = np.array([[0., 0.], [1500., -h], [1500., h]])
    angles = np.linspace(-delta, delta, 17)
    inner = np.vstack(([0., 0.], 1500*np.array([unit(t) for t in angles])))
    # 首次返回direction已排除r<=5；下界必须只使用真实可产生方向读数的位置。
    inner = clip(inner, [-1, 0], -5.01)
    low, high = bearing_interval(outer, station, delta)
    bins = max(1, math.ceil((high-low)/step))
    edges = np.linspace(low, high, bins+1)
    lower_bound = upper_bound = 0.
    for left, right in zip(edges[:-1], edges[1:]):
        center = (left+right)/2
        # 整个读数区间的并集被扩张角锥包含，故不是只采样误差端点。
        upper_poly = observe(outer, station, center, delta+(right-left)/2, range_bound=None)
        if len(upper_poly):
            upper_bound = max(upper_bound, minimum_enclosing_circle(upper_poly)[1])
        lower_poly = observe(inner, station, center, delta, range_bound=None)
        if len(lower_poly):
            lower_bound = max(lower_bound, minimum_enclosing_circle(lower_poly)[1])
    return lower_bound, upper_bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='输出/第二问')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in np.arange(100, 1001, 100):
        for b in np.arange(100, 901, 100):
            if reception_region(a, b):
                lower, upper = minimax_bounds(np.array([a, b]), step=1)
                rows.append(dict(a=float(a), b=float(b), lower_m=lower, upper_m=upper,
                                 movement_s=math.hypot(a, b)/5, step_deg=1.))
    # 加入最大移动半径边界上的候选，避免矩形网格漏掉高质量大基线。
    for a in np.arange(600, 951, 25):
        b = -5*math.sin(math.radians(1))+math.sqrt(1000**2-(a-5*math.cos(math.radians(1)))**2)
        if reception_region(a, b):
            lower, upper = minimax_bounds(np.array([a, b]), step=1)
            rows.append(dict(a=float(a), b=float(b), lower_m=lower, upper_m=upper,
                             movement_s=math.hypot(a,b)/5, step_deg=1.))
    top = sorted(rows, key=lambda r:r['upper_m'])[:5]
    # 所有候选均保留；仅在最好的五个网格点细化角度证书，不声称连续全局最优。
    refined = []
    for row in top:
        lower, upper = minimax_bounds(np.array([row['a'], row['b']]), step=.1)
        refined.append(dict(row, lower_m=lower, upper_m=upper, step_deg=.1))
    best_upper = min(r['upper_m'] for r in refined)
    for row in rows:
        if row['lower_m'] <= best_upper:
            lower, upper = minimax_bounds(np.array([row['a'], row['b']]), step=.01)
            refined.append(dict(row, lower_m=lower, upper_m=upper, step_deg=.01))
    fixed = []
    for a, b in ((750, 300), (500, 800), (750, 600)):
        lo, hi = minimax_bounds(np.array([a, b]), step=.1)
        fixed.append(dict(a=a, b=b, lower_m=lo, upper_m=hi,
                          guaranteed_reception=bool(reception_region(a, b))))
    result = dict(delta_deg=1., first_direction_min_distance_m=5., candidates=rows, refined=refined, comparisons=fixed,
                  selected=min(refined, key=lambda r:r['upper_m']),
                  scope='几何定位质量优先；离散候选设计，上下界含连续读数，不是连续选点全局最优')
    (out/'design.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='candidates'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
