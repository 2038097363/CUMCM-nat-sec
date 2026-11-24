"""独立诊断定向发现覆盖，不使用测向误差或隐藏模拟器数据。

判据：g必须在距离<=1000的站点的凸包中。格点仅用于寻找反例；
verify_boxes以盒内固定可接收站集的凸包覆盖整盒，提供连续覆盖证书。
"""
from pathlib import Path
import argparse
import heapq
import json
import math
import numpy as np
from scipy.spatial import ConvexHull


def stations(k,l,rinner,router,phase=0):
    inner=np.radians(np.arange(k)*360/k+phase)
    outer=np.radians(np.arange(l)*360/l)
    return np.vstack(([0,0],rinner*np.c_[np.cos(inner),np.sin(inner)],
                      router*np.c_[np.cos(outer),np.sin(outer)]))


def diagnose(st,points):
    vectors=st[None,:,:]-points[:,None,:]
    near=np.linalg.norm(vectors,axis=2)<=1000
    angles=np.mod(np.arctan2(vectors[:,:,1],vectors[:,:,0]),2*np.pi)
    angles[~near]=np.inf
    ordered=np.sort(angles,axis=1)
    n=near.sum(axis=1)
    gaps=np.diff(ordered,axis=1)
    gaps[~np.isfinite(gaps)]=0
    last=ordered[np.arange(len(points)),np.maximum(0,n-1)]
    wrap=ordered[:,0]+2*np.pi-last
    maximum=np.maximum(gaps.max(axis=1),wrap)
    maximum[n<2]=2*np.pi
    # 源恰在站点时，单独由周围站证明其邻域；诊断不把零向量角度当正信号方向。
    exact=np.min(np.linalg.norm(vectors,axis=2),axis=1)<1e-8
    bad=(maximum>np.pi+1e-8)&~exact
    return np.flatnonzero(bad),maximum


def verify_boxes(st,min_side=.1,max_boxes=1000000):
    stack=[(-1800.,-1800.,1800.,1800.)]
    accepted=[];outside=[];unresolved=[];processed=0
    while stack and processed<max_boxes:
        box=stack.pop();processed+=1
        lo=np.array(box[:2]);hi=np.array(box[2:]);c=(lo+hi)/2
        if np.linalg.norm(np.clip(np.zeros(2),lo,hi))>1800+1e-7:
            outside.append(box);continue
        h=np.linalg.norm(hi-lo)/2
        indices=np.flatnonzero(np.linalg.norm(st-c,axis=1)+h<=999.99)
        corners=np.array([[lo[0],lo[1]],[lo[0],hi[1]],[hi[0],lo[1]],[hi[0],hi[1]]])
        covered=False
        if len(indices)>=3:
            hull=ConvexHull(st[indices])
            covered=np.all(corners@hull.equations[:,:2].T+hull.equations[:,2]<=-1e-7)
        if covered:
            accepted.append({'box':box,'station_indices':indices.tolist()});continue
        if max(hi-lo)<=min_side:
            unresolved.append(box);continue
        axis=int(np.argmax(hi-lo));mid=c[axis]
        h1=hi.copy();h1[axis]=mid;l2=lo.copy();l2[axis]=mid
        stack.extend([tuple(lo)+tuple(h1),tuple(l2)+tuple(hi)])
    unresolved.extend(stack)
    return dict(certified=len(unresolved)==0,processed=processed,accepted=accepted,
                outside=outside,unresolved=unresolved,range_safety_margin_m=.01)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--verify-best',action='store_true')
    args=parser.parse_args()
    out=Path('输出/定向布局核验');out.mkdir(parents=True,exist_ok=True)
    r=np.r_[np.arange(5,1801,15),1800.]
    a=np.radians(np.arange(0,360,.75))
    points=np.vstack([v*np.c_[np.cos(a),np.sin(a)] for v in r])
    rows=[]
    with np.errstate(invalid='ignore'):
        for k,l in [(7,10),(8,9),(7,11),(8,10),(7,12),(8,12),(9,12)]:
            for inner in (960,980,995):
                for extra in (1.,5.,15.):
                    outer=1800/math.cos(math.pi/l)+extra
                    phase=180/math.lcm(k,l)
                    st=stations(k,l,inner,outer,phase)
                    bad,gaps=diagnose(st,points)
                    row=dict(k=k,l=l,count=len(st),inner=inner,outer=outer,phase=phase,
                             failures=len(bad),worst_gap_deg=float(np.degrees(max(gaps))),
                             counterexample=points[bad[0]].tolist() if len(bad) else None)
                    rows.append(row)
                    print(row,flush=True)
    (out/'screening.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    candidates=sorted([r for r in rows if r['failures']==0],key=lambda r:(r['count'],r['worst_gap_deg']))
    if args.verify_best and candidates:
        for index,row in enumerate(candidates[:4]):
            st=stations(row['k'],row['l'],row['inner'],row['outer'],row['phase'])
            cert=verify_boxes(st,min_side=.25,max_boxes=300000)
            cert.update(design=row,stations=st.tolist())
            (out/f'certificate_{index}.json').write_text(json.dumps(cert),encoding='utf-8')
            print('CERT',index,cert['certified'],cert['processed'],len(cert['unresolved']),flush=True)
            if cert['certified']:break


if __name__=='__main__':main()
