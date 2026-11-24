"""实验：按清除圆几何优化矩形覆盖，不向策略提供真值。

每个格子的半对角线≤19.9米；长宽比、行数及朝向以完整执行成本代理选择。
失败清除只删除被已失败20米圆完全包含的单元交集，保留连续覆盖证书。
"""
import math
import numpy as np
from geometry import clip,unit,diameter


def cells(poly,angle,nx,ny,failed=(),margin=19.9):
    u=unit(angle);v=np.array([-u[1],u[0]])
    origin=poly.mean(axis=0);local=np.c_[(poly-origin)@u,(poly-origin)@v]
    lo=local.min(axis=0);hi=local.max(axis=0);step=(hi-lo)/[nx,ny]
    assert np.linalg.norm(step)/2<=margin+1e-7
    result=[]
    for j in range(ny):
        indices=range(nx) if j%2==0 else range(nx-1,-1,-1)
        for i in indices:
            lower=lo+[i,j]*step;upper=lower+step;piece=local.copy()
            for n,b in [([1,0],upper[0]),([-1,0],-lower[0]),([0,1],upper[1]),([0,-1],-lower[1])]:
                piece=clip(piece,n,b)
                if not len(piece):break
            if not len(piece):continue
            world=origin+piece[:,0,None]*u+piece[:,1,None]*v
            if any(np.max(np.linalg.norm(world-f,axis=1))<=20-1e-7 for f in failed):continue
            center=(lower+upper)/2;point=origin+center[0]*u+center[1]*v
            assert np.max(np.linalg.norm(world-point,axis=1))<=margin+1e-6
            result.append((point,world))
    return result


def cover(poly,first_station,first_bearing,current,failed=(),mode='square28'):
    # 首测沿轴≤1500为真实约束，先裁去圆外包造成的微量伸出。
    u=unit(first_bearing);p=clip(poly,u,float(u@first_station+1500))
    if not len(p):raise RuntimeError('首测约束为空')
    angles=[first_bearing];plans=[];margin=19.9
    if mode=='oriented':
        _,(a,b)=diameter(p)
        if np.linalg.norm(b-a)>1e-9:angles.append(float(np.degrees(np.arctan2(*(b-a)[::-1]))))
        edges=np.roll(p,-1,axis=0)-p
        for edge in sorted(edges,key=lambda x:-np.linalg.norm(x))[:3]:
            if np.linalg.norm(edge)>1e-9:angles.append(float(np.degrees(np.arctan2(edge[1],edge[0]))))
    for angle in sorted(set(round(a,7) for a in angles)):
        u=unit(angle);v=np.array([-u[1],u[0]])
        local=np.c_[p@u,p@v];length,width=np.ptp(local,axis=0)
        sizes=[]
        if mode=='square28':sizes=[(max(1,math.ceil(length/28)),max(1,math.ceil(width/28)))]
        else:
            # 限制格子行数的候选搜索；每个候选均满足严格几何覆盖。
            first=max(1,math.floor(width/(2*margin))+1)
            for ny in range(first,first+4):
                dx=math.sqrt(max(0.,(2*margin)**2-(width/ny)**2))
                if dx<5:continue
                nx=max(1,math.ceil(length/dx))
                if nx*ny<=500:sizes.append((nx,ny))
        for nx,ny in sizes:
            grid=cells(p,angle,nx,ny,failed,margin)
            if not grid:continue
            points=[c for c,_ in grid]
            path=sum(np.linalg.norm(a-b) for a,b in zip(points[:-1],points[1:]))
            entry=min(np.linalg.norm(points[0]-current),np.linalg.norm(points[-1]-current))
            # 所有格子完整执行的上界；本次真源可能早清除，但不设置成功概率。
            score=(entry+path)/5+3*len(points)+2
            plans.append((score,points,dict(angle=angle,nx=nx,ny=ny,cells=len(points),
                                         full_snake_bound_s=float(score),half_diagonal=float(math.hypot(length/nx,width/ny)/2))))
    if not plans:raise RuntimeError('未找到连续覆盖网格')
    _,points,certificate=min(plans,key=lambda z:z[0])
    return points,certificate
