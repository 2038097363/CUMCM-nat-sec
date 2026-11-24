"""连续第二点选址的分支定界：固定源集见证给站点盒下界，读数分箱给上界。

这是浮点计算的几何证书；报告数值余量，不称为形式化区间算术证明。
"""
from pathlib import Path
import argparse
import heapq
import json
import math
import time
import numpy as np
from geometry import clip, observe, unit, minimum_enclosing_circle, diameter
from q2_design import reception_region, minimax_bounds

DELTA=1.
ANGLE=math.radians(2*DELTA)
PLOW=np.vstack(([0.,0.],1500*np.array([unit(a) for a in np.linspace(-1,1,17)])))
PLOW=clip(PLOW,[-1,0],-5.01)
DISKS=[r*unit(a) for r in (5,1000) for a in (-1,1)]
SOURCE_SAMPLES=np.vstack([r*np.array([unit(a) for a in (-1.,0.,1.)]) for r in np.linspace(5.01,1500,17)])


def box_possible(box):
    lo,hi=np.array(box[:2]),np.array(box[2:])
    return all(np.linalg.norm(c-np.clip(c,lo,hi))<=1000+1e-9 for c in DISKS)


def witnesses(station):
    angles=np.degrees(np.arctan2(SOURCE_SAMPLES[:,1]-station[1],SOURCE_SAMPLES[:,0]-station[0]))
    betas=np.unique(np.round(np.r_[angles-1,angles,angles+1],8))
    candidates=[]
    for beta in betas:
        p=observe(PLOW,station,beta,1.,range_bound=None)
        if len(p)>=2:
            c=p.mean(axis=0)
            r=diameter(p)[0]/2
            candidates.append((r,c,p,float(beta)))
    return sorted(candidates,key=lambda x:x[0],reverse=True)[:3]


def persistent_witness(witness,station,h):
    r,c,p,beta=witness
    def feasible(scale):
        v=c+scale*(p-c)
        distances=np.linalg.norm(v-station,axis=1)
        if min(distances)-h<=5.001:return False
        angles=np.arctan2(v[:,1]-station[1],v[:,0]-station[0])
        gaps=np.abs(np.angle(np.exp(1j*(angles[:,None]-angles[None,:]))))
        separation=np.linalg.norm(v[:,None,:]-v[None,:,:],axis=2)
        denominators=(distances[:,None]-h)*(distances[None,:]-h)
        # 角度差梯度的模恰为两源距离/(到两源距离之积)。
        bound=gaps+h*separation/denominators
        return bool(np.max(bound)<=ANGLE-1e-10)
    if not feasible(0):return 0.,[]
    low,high=0.,1.
    for _ in range(20):
        middle=(low+high)/2
        if feasible(middle):low=middle
        else:high=middle
    return max(0.,r*low-1e-5),(c+low*(p-c)).tolist()


def box_lower(box):
    lo,hi=np.array(box[:2]),np.array(box[2:])
    station=(lo+hi)/2;h=np.linalg.norm(hi-lo)/2
    ws=witnesses(station)
    return max([persistent_witness(w,station,h) for w in ws]+[(0.,[])],key=lambda item:item[0])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gap',type=float,default=.5)
    parser.add_argument('--max-boxes',type=int,default=15000)
    args=parser.parse_args()
    out=Path('输出/第二问/连续选址核验');out.mkdir(parents=True,exist_ok=True)
    design=json.loads(Path('输出/第二问/design.json').read_text(encoding='utf-8'))
    selected=design['selected']; upper=selected['upper_m']+1e-5
    queue=[(0.,0,(0.,0.,1005.,1000.))];counter=1;expanded=0
    leaves={queue[0][2]:dict(kind='witness',lower_m=0.,points=[])}
    terminals=[]; started=time.monotonic()
    while queue and expanded<args.max_boxes:
        lower,_,box=heapq.heappop(queue)
        if upper-lower<=args.gap:
            terminals.append((lower,box));break
        lo,hi=np.array(box[:2]),np.array(box[2:])
        leaves.pop(box)
        axis=int(np.argmax(hi-lo));mid=(lo[axis]+hi[axis])/2
        bounds=[]
        left_hi=hi.copy();left_hi[axis]=mid;bounds.append(tuple(lo)+tuple(left_hi))
        right_lo=lo.copy();right_lo[axis]=mid;bounds.append(tuple(right_lo)+tuple(hi))
        for child in bounds:
            if not box_possible(child):
                leaves[child]=dict(kind='outside');continue
            lb,points=box_lower(child)
            leaves[child]=dict(kind='witness',lower_m=lb,points=points)
            if lb<=upper:
                heapq.heappush(queue,(lb,counter,child));counter+=1
            else:terminals.append((lb,child))
        expanded+=1
        if expanded%100==0:
            report=dict(expanded=expanded,active=len(queue),lower=min(queue)[0] if queue else upper,
                        upper=upper,elapsed_s=time.monotonic()-started)
            print(report,flush=True)
            (out/'progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    lower=min([v[0] for v in queue]+[v[0] for v in terminals]+[upper])
    result=dict(lower_m=lower,upper_m=upper,gap_m=upper-lower,selected=selected,
                expanded_boxes=expanded,active_boxes=len(queue),elapsed_s=time.monotonic()-started,
                domain='首次direction、delta=1度、首测点在原点；对称取b>=0；四圆盘保证接收域',
                witness_method='同一可行源点集在站点盒内仍能产生相容读数；角差梯度界；缩放直径的一半作下界，不依赖最小圆最优性',
                status='gap_reached' if upper-lower<=args.gap+1e-8 else 'box_budget_exhausted',
                numerical_margin_m=1e-5)
    (out/'certificate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'remaining_boxes.json').write_text(json.dumps(queue),encoding='utf-8')
    (out/'proof_leaves.json').write_text(json.dumps([dict(box=box,**row) for box,row in leaves.items()]),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
