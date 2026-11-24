"""光学清除的六边形 Voronoi 单元覆盖，每单元外接半径为 19.9 米。

格子先与可行多边形相交；被已有失败清除圆完全包含的交集可安全删除。
返回蛇形顺序，之后可选取更短路线；不使用真实源位置。
"""
import math
import numpy as np
from geometry import unit,clip


def hex_cells(poly, first_station, first_bearing, failed=(), radius=19.9):
    u=unit(first_bearing);v=np.array([-u[1],u[0]]);origin=np.asarray(first_station)
    local=np.c_[(poly-origin)@u,(poly-origin)@v]
    local=clip(local,[1,0],1500.)
    if not len(local):return []
    lo=local.min(axis=0);hi=local.max(axis=0)
    dx=math.sqrt(3)*radius;dy=1.5*radius
    normals=np.array([unit(a) for a in range(0,360,60)])
    apothem=dx/2
    # 位移仅影响效率；任意位移的无限三角格都有相同覆盖保证。
    anchor=(lo+hi)/2
    jmin=math.floor((lo[1]-radius-anchor[1])/dy)
    jmax=math.ceil((hi[1]+radius-anchor[1])/dy)
    result=[]
    for j in range(jmin,jmax+1):
        offset=(j%2)*dx/2
        imin=math.floor((lo[0]-radius-anchor[0]-offset)/dx)
        imax=math.ceil((hi[0]+radius-anchor[0]-offset)/dx)
        indices=range(imin,imax+1) if j%2==0 else reversed(range(imin,imax+1))
        for i in indices:
            center=anchor+np.array([i*dx+offset,j*dy]);q=local.copy()
            for normal in normals:
                q=clip(q,normal,float(normal@center+apothem))
                if not len(q):break
            if not len(q):continue
            world=origin+center[0]*u+center[1]*v
            vertices=origin+q[:,0,None]*u+q[:,1,None]*v
            # 凸函数在凸多边形上的最大值由顶点取到。
            if any(np.max(np.linalg.norm(vertices-f,axis=1))<=19.99 for f in failed):continue
            assert np.max(np.linalg.norm(vertices-world,axis=1))<=radius+1e-5
            result.append((world,vertices))
    return result


def hex_cover(poly, first_station, first_bearing, failed=()):
    return [center for center,_ in hex_cells(poly,first_station,first_bearing,failed)]
