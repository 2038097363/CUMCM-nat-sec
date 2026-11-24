"""原点全阴性时保留搜索期原调度，其余情况采用有界探查与连续评分。"""
from q4_completion_probe import CompletionProbeQ4


class BlindSafeQ4(CompletionProbeQ4):
    def __init__(self,backend,question=4,variant='blind_safe',**kwargs):
        super().__init__(backend,question,variant='probe1500',**kwargs)
        self.variant=variant;self.blind_origin=False

    def classify_origin(self):
        super().classify_origin()
        self.blind_origin=not any(b.status in ('found','cleared') for b in self.states.values())
        self.portfolio['blind_origin']=self.blind_origin

    def caution(self,todo):
        a,resultant,n=super().caution(todo)
        if todo and self.blind_origin:a=1.
        return a,resultant,n
