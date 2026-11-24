from adaptive_solver import AdaptiveRobot
from joint_solver import JointRobot
from optical_rectangle import cover


class OpticalMixin:
    def __init__(self,*args,optical_mode='square28',**kwargs):
        super().__init__(*args,**kwargs);self.optical_mode=optical_mode;self.optical_plans=[]

    def optical_points(self,poly,first_station,first_bearing,failed):
        if self.optical_mode=='current':return super().optical_points(poly,first_station,first_bearing,failed)
        points,cert=cover(poly,first_station,first_bearing,self.position,failed,self.optical_mode)
        self.optical_plans.append(cert);return points


class OpticalJoint(OpticalMixin,JointRobot):pass
class OpticalV2(OpticalMixin,AdaptiveRobot):pass


