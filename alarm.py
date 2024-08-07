"""
・各tに対して一度しかalarmが鳴らないようにする。
・与えるtが単調増加でない場合のエラー処理は未実装
・ListAlarm(float('inf'))はエラーになる。
"""

class Alarm:
    def __init__(self):
        pass
    def __call__(self, batch):
        raise NotImplementedError
class RangeAlarm(Alarm):
    def __init__(self, target, start=0, step=1, end=False):
        self.target = target
        self.start = start-step
        self.step = step
        self.last_t = self.start
        self.end = end
    def __call__(self, batch):
        t = batch[self.target]
        if (t - self.start) % self.step == 0:
            if self.last_t < t:
                self.last_t = t
                return True
        if self.end and 'end' in batch:
            return True
        return False
class ListAlarm(Alarm):
    def __init__(self, target, list, end=False):
        self.target = target
        self.list_ = sorted(list)
        self.list_.append(float('inf'))
        self.last_t = None
        self.end = end
    def __call__(self, batch):
        t = batch[self.target]
        if self.list_[0] <= t:
            while self.list_[0] < t:
                del self.list_[0]
            if self.list_[0] == t:
                del self.list_[0]
                return True
        if self.end and 'end' in batch:
            return True
        return False
class SilentAlarm(Alarm):
    def __init__(self, end=False):
        self.end = end
        pass
    def __call__(self, batch):
        if self.end and 'end' in batch:
            return True
        return False

alarm_type2class = {
    'count': RangeAlarm,
    'range': RangeAlarm,
    'list': ListAlarm,
    'silent': SilentAlarm
}
def get_alarm(type='silent', **kwargs):
    return alarm_type2class[type](**kwargs)