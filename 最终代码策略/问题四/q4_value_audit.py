"""独立验收：按实际反馈检查21站访问，不读取后台真值。"""
import math
import numpy as np
from adaptive_solver import reduced_stations


def coverage_audit(robot):
    cleared={e['channel'] for e in robot.history if e.get('path')=='/clear' and e['response']['clear_result']=='success'}
    if len(cleared)==16:return dict(valid=True,reason='16个不同频道已成功清除',missing=[])
    angle=math.radians(getattr(robot,'rotation',0.))
    rot=np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    stations=reduced_stations(4)@rot.T
    missing=[]
    for k in range(1,21):
        if k in cleared:continue
        negatives=[e['position'] for e in robot.history if e.get('path')=='/measure' and e['channel']==k and e['response']['measure_result']=='no_signal']
        for i,p in enumerate(stations):
            if not negatives or min(np.linalg.norm(np.asarray(n)-p) for n in negatives)>1e-5:missing.append([k,i])
    return dict(valid=not missing,reason='所有未清除频道实际测完连续覆盖布局',missing=missing,rotation_deg=math.degrees(angle))
