"""独立复核删站：事件前缀中的逐频道负观测 + 四叉树圆并覆盖证书。

不调用策略的 Voronoi 检查，不使用面积采样。每个保留方格的四个角
都在同一个1000米圆中，才证明整个方格被覆盖；目标域外方格被舍弃。
"""
import numpy as np


def certify_disk_union(sites,max_cells=200000):
    points=np.asarray(sites,dtype=float)
    if not len(points):return dict(valid=False,reason='empty')
    stack=[(-1800.,-1800.,1800.,1800.)];leaves=0;visited=0
    while stack:
        x0,y0,x1,y1=stack.pop();visited+=1
        if visited>max_cells:return dict(valid=False,reason='budget')
        nearest=np.maximum(np.maximum([x0,y0],[-x1,-y1]),0.)
        if nearest@nearest>1800.**2:continue
        corners=np.array([[x0,y0],[x0,y1],[x1,y0],[x1,y1]])
        d2=np.sum((corners[:,None,:]-points[None,:,:])**2,axis=2)
        if np.min(np.max(d2,axis=0))<=(1000.-1e-7)**2:
            leaves+=1;continue
        if max(x1-x0,y1-y0)<.001:return dict(valid=False,reason='unresolved')
        xm=(x0+x1)/2;ym=(y0+y1)/2
        stack.extend([(x0,y0,xm,ym),(xm,y0,x1,ym),(x0,ym,xm,y1),(xm,ym,x1,y1)])
    return dict(valid=True,leaves=leaves,visited=visited)


def audit_robot_coverage(robot):
    proofs=0;geometries={};observations=0
    records=[]
    for item in robot.pruning:
        for check in item['checks']:
            records.append((check,item['future'],item['evidence_event_count']))
    for item in robot.certificates:
        records.append((item,[],item['evidence_event_count']))
    for check,future,limit in records:
        k=check['channel']
        negatives={tuple(e['position']) for e in robot.history[:limit]
                   if e.get('path')=='/measure' and e['channel']==k
                   and e['response']['measure_result']=='no_signal'}
        for p in check['negative_points']:
            assert tuple(p) in negatives,'覆盖证据没有该频道此前实际无信号记录'
            observations+=1
        key=tuple(sorted(set(tuple(p) for p in check['negative_points']+future)))
        if key not in geometries:geometries[key]=certify_disk_union(key)
        assert geometries[key]['valid'],'独立方格证书未通过'
        proofs+=1
    return dict(proofs=proofs,unique_geometries=len(geometries),
                observed_evidence_checks=observations,
                box_leaves=sum(p['leaves'] for p in geometries.values()),valid=True)
