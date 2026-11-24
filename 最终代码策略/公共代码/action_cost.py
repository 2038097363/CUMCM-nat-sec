"""比较检测、试探的成本代理；不假定源位置或测角误差有概率分布。

后备矩形网格的完整执行成本是几何上界；检测反馈仅有限枚举，整个动作
评分仍是最坏分支的近似，不称为连续鲁棒最优值。任何评分不参与删真值。
"""
import math
import numpy as np
from geometry import minimum_enclosing_circle,diameter,observe,nearest_clear_point


def fallback_bound(poly,current,clear_margin=19.9):
    center,r=minimum_enclosing_circle(poly)
    if r<=clear_margin:
        q=nearest_clear_point(poly,current,clear_margin)
        return float(np.linalg.norm(q-current)/5+5)
    _,(a,b)=diameter(poly);u=(b-a)/max(np.linalg.norm(b-a),1e-15);v=np.array([-u[1],u[0]])
    local=np.c_[poly@u,poly@v];lo=local.min(axis=0);hi=local.max(axis=0)
    counts=np.maximum(1,np.ceil((hi-lo)/(clear_margin*math.sqrt(2))).astype(int))
    dx=(hi-lo)/counts;nx,ny=map(int,counts)
    corners=np.array([[x,y] for x in (lo[0]+dx[0]/2,hi[0]-dx[0]/2)
                      for y in (lo[1]+dx[1]/2,hi[1]-dx[1]/2)])
    world=corners[:,0,None]*u+corners[:,1,None]*v
    entry=np.min(np.linalg.norm(world-current,axis=1))
    internal=ny*max(0,nx-1)*dx[0]+max(0,ny-1)*dx[1]
    return float((entry+internal)/5+3*(nx*ny-1)+5)


def measurement_tail(poly,station,delta=1.01,guaranteed_signal=True):
    # 顶点、边中点和中心是固定反馈见证；不引入均匀位置先验或噪声概率。
    samples=np.vstack((poly,(poly+np.roll(poly,-1,axis=0))/2,poly.mean(axis=0)))
    angles=np.degrees(np.arctan2(samples[:,1]-station[1],samples[:,0]-station[0]))
    betas=np.unique(np.round(np.r_[angles-delta,angles,angles+delta],7))
    worst=0.
    for beta in betas:
        posterior=observe(poly,station,float(beta),delta)
        if len(posterior):worst=max(worst,fallback_bound(posterior,station))
    if np.any(np.linalg.norm(samples-station,axis=1)<=5):worst=max(worst,5.)
    # 未证明再次接收时，无信号这一不利分支必须保留；Q4尤其不能漏。
    if not guaranteed_signal:worst=max(worst,fallback_bound(poly,station))
    return worst


def clear_score(poly,current,point,clear_margin=19.9):
    movement=float(np.linalg.norm(current-point)/5)
    if np.max(np.linalg.norm(poly-point,axis=1))<=clear_margin:return movement+5
    # 失败后保守保留原位置域，随后矩形网格保证完成；不虚构成功概率。
    return max(movement+5,movement+3+fallback_bound(poly,point,clear_margin))


def detour_probe_terms(current,probe,measurement,switch):
    """先试探、失败就执行原定检测；失败增量不超过一次额外检测的成本。

    这是局部时间预算规则，没有假定成功概率，也不是全任务期望最优。
    """
    direct=np.linalg.norm(current-measurement)
    to_probe=np.linalg.norm(current-probe);back=np.linalg.norm(probe-measurement)
    failure_loss=max(0.,float(to_probe+back-direct))/5+3
    success_time=float(to_probe)/5+5
    # 若试探能成功，源在probe的20米内，后续任一清除点在其40米内。
    measure_success_lower=float(direct)/5+5+int(switch)+max(0.,float(back)-40)/5+5
    budget=5+int(switch)
    return dict(allowed=failure_loss<=budget+1e-8 and success_time<measure_success_lower-1e-8,
                failure_extra_s=failure_loss,budget_s=budget,success_s=success_time,
                measure_completion_lower_if_probe_succeeds_s=measure_success_lower)
