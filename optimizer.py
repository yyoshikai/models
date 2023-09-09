import math
from collections import defaultdict
import torch
from torch import optim 
from torch.optim import lr_scheduler


# From https://github.com/szc19990412/TransMIL
class RAdam(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super().__init__(params, defaults)

    @torch.no_grad() # SGDクラスに入っていたので入れてみた。
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) changed for deprecation
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt(
                            (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0 and group['weight_decay'] is not None:
                    p_data_fp32.add_(p_data_fp32, alpha=-group['weight_decay'] * group['lr'])

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size)

                p.data.copy_(p_data_fp32)
        return loss

class LookaheadOptimizer(optim.Optimizer):
    def __init__(self, params, base, base_args={}):
        self.base_optimizer = optimizer_type2class[base](params, **base_args)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults
        self.state = defaultdict(dict)
        self._optimizer_step_pre_hooks = {} # エラーが出たため
        self._optimizer_step_post_hooks = {} # エラーが出たため

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(fast_p.data - slow, alpha=group['lookahead_alpha'])
            fast_p.data.copy_(slow)
    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)
    def step(self, closure=None):
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss
    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }
    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super().load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)

optimizer_type2class = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'radam': RAdam,
    'lookahead': LookaheadOptimizer
}
def get_optimizer(type, **kwargs):
    return optimizer_type2class[type](**kwargs)

scheduler_type2class = {
    'multistep': lr_scheduler.MultiStepLR,
    'linear': lr_scheduler.LinearLR,
    'exponential': lr_scheduler.ExponentialLR
}
def get_scheduler(optimizer, type, last_epoch=-1, **kwargs):
    if type in scheduler_type2class:
        return scheduler_type2class[type](optimizer=optimizer, **kwargs)
    else:
        if type == 'warmup':
            warmup_step = kwargs['warmup']
            schedule = lambda step: min((warmup_step/(step+1))**0.5, (step+1)/warmup_step)
        elif type == 'reciprocal':
            warmup_step = kwargs['warmup']
            schedule = lambda step: ((step+1)/warmup_step)**-0.5
        elif type == 'noam':
            # for old Transformer
            factor = kwargs['d_model']**-0.5
            wfactor = kwargs['warmup']**-1.5
            schedule = lambda step: factor*min((step+1)**-0.5, (step+1)*wfactor)
        else:
            raise ValueError(f"Unsupported type of scheduler: {type}")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule, last_epoch=last_epoch)