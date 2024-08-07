"""
・各tに対して一度しかalarmが鳴らないようにする。
・与えるtが単調増加でない場合のエラー処理は未実装
・ListAlarm(float('inf'))はエラーになる。
"""
from models.utils import get_set

class Alarm:
    def __init__(self, end=False, points: dict[str, list|dict]={}):
        self.end = end
        self.points = {key: get_set(config) for key, config in points.items()}
        self.reason = None

    def __call__(self, batch: dict):
        if self.end and batch.get('end', False):
            self.reason = 'end'
            return self.reason
        for key, points in self.points.values():
            if batch.get(key, None) in points:
                self.reason = key
                return self.reason
        else:
            self.reason = None
            return self.reason
