"""
・各tに対して一度しかalarmが鳴らないようにする。
・与えるtが単調増加でない場合のエラー処理は未実装
・listはinfを投げるとエラーになる。
"""
from .utils import check_leftargs

class Alarm:
    def __init__(self):
        pass
    def __call__(self, t):
        raise NotImplementedError
class RangeAlarm(Alarm):
    def __init__(self, logger, start=0, step=1, **kwargs):
        check_leftargs(self, logger, kwargs)
        self.start = start-step
        self.step = step
        self.last_t = self.start
    def __call__(self, t):
        if (t - self.start) % self.step == 0:
            if self.last_t < t:
                self.last_t = t
                return True
        return False
class ListAlarm(Alarm):
    def __init__(self, logger, list, **kwargs):
        check_leftargs(self, logger, kwargs)
        self.list_ = sorted(list)
        self.list_.append(float('inf'))
        self.last_t = None
    def __call__(self, t):
        if self.list_[0] <= t:
            while self.list_[0] < t:
                del self.list_[0]
            if self.list_[0] == t:
                del self.list_[0]
                return True
        return False
class SilentAlarm(Alarm):
    def __init__(self, logger, **kwargs):
        check_leftargs(self, logger, kwargs)
        pass
    def __call__(self, t):
        return False

alarm_type2class = {
    'count': RangeAlarm,
    'range': RangeAlarm,
    'list': ListAlarm,
    'silent': SilentAlarm
}
def get_alarm(type, **kwargs):
    return alarm_type2class[type](**kwargs)