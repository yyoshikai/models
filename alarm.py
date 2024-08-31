"""
・各tに対して一度しかalarmが鳴らないようにする。
・与えるtが単調増加でない場合のエラー処理は未実装
・ListAlarm(float('inf'))はエラーになる。
"""
from models.utils import get_set

class Alarm:
    def __init__(self, **points):
        self.points = {key: get_set(config) for key, config in points.items()}
        self.reason = None

    def __call__(self, batch: dict):
        for key, points in self.points.values():
            if batch.get(key, None) in points:
                self.reason = key
                return self.reason
        else:
            self.reason = None
            return self.reason

def get_alarm(**kwargs) -> Alarm:
    return Alarm(**kwargs)