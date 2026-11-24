"""问题三连续覆盖区域模块的纯内存接口适配；不调用HTTP入口，不改变算法文件。"""
from pathlib import Path
import sys,json,time
import numpy as np

LIB=Path(__file__).resolve().parent/'q3_ccr_js_lib'
sys.path.append(str(LIB))
from algorithms import robot_class
PeerClass=robot_class('optimized_v4')


class PeerOffline(PeerClass):
    def __init__(self,backend,audit_callback=None):
        super().__init__('offline-only',3,None,base_url='http://disabled.invalid')
        self.backend=backend;self.audit_callback=audit_callback;self.history=[]
        self.costs=dict(movement_s=0.,measure_s=0.,switch_s=0.,clear_s=0.)
        self.actions=dict(measure=0,clear_success=0,clear_failed=0)
        self.negative_evidence={k:[] for k in range(1,21)}

    def post(self,path,extra=None):
        if path not in ('/enter','/measure','/clear','/exit'):raise RuntimeError('仅允许四个内存动作')
        if path!='/exit' and time.monotonic()>=self.deadline:raise TimeoutError('离线运行超时')
        extra=extra or {};pos=extra.get('position');k=extra.get('channel')
        point=None if pos is None else (pos['x'],pos['y'])
        response=self.backend.post(path,point,k)
        expected=0.
        if point is not None:
            move=float(np.linalg.norm(np.asarray(point)-self.position)/5)
            self.costs['movement_s']+=move;expected+=move
            if path=='/measure':
                switch=int(self.channel!=k);self.costs['switch_s']+=switch;self.costs['measure_s']+=5
                self.actions['measure']+=1;expected+=switch+5
                if response['measure_result']=='no_signal':self.negative_evidence[k].append(point)
            else:
                success=response['clear_result']=='success';cost=5 if success else 3
                self.costs['clear_s']+=cost;expected+=cost
                self.actions['clear_success' if success else 'clear_failed']+=1
            if abs(response['virtual_time_s']-self.virtual_time-expected)>2e-4:raise RuntimeError('适配器计费不符')
            self.position=point
        self.seq+=1;self.virtual_time=response['virtual_time_s']
        event=dict(path=path,response=response)
        if point is not None:event.update(position=list(point),channel=k)
        self.history.append(event)
        self.log.write(json.dumps(event)+'\n')
        if response.get('accepted') is not True:raise RuntimeError('内存动作被拒绝')
        return response

    def measure(self,point,channel):
        result=super().measure(point,channel)
        if self.audit_callback and channel in self.polygons:self.audit_callback(channel,np.asarray(self.polygons[channel]))
        return result
