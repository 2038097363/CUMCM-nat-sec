"""自建场景验证；策略只看接口响应，真值只供独立评估与作图。"""
from pathlib import Path
from dataclasses import dataclass
import argparse
import csv
import hashlib
import json
import math
import random
import time
import numpy as np
from geometry import contains
from solve import Robot, configuration


@dataclass(frozen=True)
class Source:
    channel: int
    x: float
    y: float
    radius: float
    direction: float | None


def scenario(question, seed):
    rng = random.Random(seed)
    n = (10, 13, 16)[seed % 3]
    channels = rng.sample(range(1, 21), n)
    pattern = ('uniform', 'boundary', 'cluster')[(seed//3) % 3]
    cluster_angle = rng.random()*2*math.pi
    cluster = np.array([900*math.cos(cluster_angle), 900*math.sin(cluster_angle)])
    sources = []
    for i, k in enumerate(channels):
        a = rng.random()*2*math.pi
        if pattern == 'uniform':
            r = 1800*math.sqrt(rng.random()); p = np.array([r*math.cos(a), r*math.sin(a)])
        elif pattern == 'boundary':
            r = 1800 if i % 2 == 0 else rng.uniform(1680, 1800)
            p = np.array([r*math.cos(a), r*math.sin(a)])
        else:
            p = cluster+np.array([rng.gauss(0, 230), rng.gauss(0, 230)])
            if np.linalg.norm(p)>1800:
                p *= 1800/np.linalg.norm(p)
        radius_mode = (seed//9) % 4
        radius = (1000., 1250., 1500., rng.uniform(1000, 1500))[radius_mode]
        direction = None
        if question == 4 and i < 1+(seed//2)%(n-1):
            if pattern == 'boundary':
                direction = math.degrees(math.atan2(p[1], p[0])) + (90 if seed % 2 else 0)
            else:
                direction = rng.uniform(0, 360)
        sources.append(Source(k, float(p[0]), float(p[1]), radius, direction))
    error_mode = ('smooth', 'plus', 'minus', 'checker', 'hash')[(seed//5) % 5]
    return sources, {'seed': seed, 'question': question, 'source_count': n,
                     'pattern': pattern, 'error_mode': error_mode, 'radius_mode': radius_mode}


class LocalSimulator:
    def __init__(self, sources, seed, error_mode='smooth'):
        self._sources = {s.channel: s for s in sources}
        self._cleared = set()
        self.seed, self.error_mode = seed, error_mode
        self.position = np.zeros(2); self.channel = 1; self.time = 0.
        self.entered = False; self.exited = False

    def error(self, p, channel):
        if self.error_mode == 'plus':
            return 1.
        if self.error_mode == 'minus':
            return -1.
        if self.error_mode == 'checker':
            return 1. if (math.floor(p[0]/50)+math.floor(p[1]/50)+channel) % 2 else -1.
        if self.error_mode == 'hash':
            token = f'{self.seed}:{channel}:{p[0]:.9f}:{p[1]:.9f}'.encode()
            return int.from_bytes(hashlib.sha256(token).digest()[:8], 'big')/(2**64-1)*2-1
        return .65*math.sin(p[0]/131+channel+self.seed)+.35*math.cos(p[1]/97-channel)

    def post(self, path, position=None, channel=None):
        if path == '/enter':
            if self.entered:
                raise RuntimeError('重复进入')
            self.entered = True
            return {'accepted': True, 'virtual_time_s': 0., 'remaining_real_duration_s': 1200,
                    'max_virtual_duration_s': 360000}
        if not self.entered or self.exited:
            raise RuntimeError('会话未开放')
        if path == '/exit':
            self.exited = True
            return {'accepted': True, 'virtual_time_s': self.time, 'exit_reason': 'user_exit'}
        point = np.asarray(position, dtype=float)
        if not np.all(np.isfinite(point)) or np.max(np.abs(point)) > 2_000_000 or channel not in range(1, 21):
            raise ValueError('非法动作参数')
        self.time += float(np.linalg.norm(point-self.position)/5)
        self.position = point.copy()
        source = self._sources.get(channel)
        active = source is not None and channel not in self._cleared
        distance = math.hypot(point[0]-source.x, point[1]-source.y) if active else math.inf
        if path == '/clear':
            success = distance <= 20
            self.time += 5 if success else 3
            if success:
                self._cleared.add(channel)
            return {'accepted': True, 'virtual_time_s': self.time,
                    'clear_result': 'success' if success else 'no_target_in_range'}
        if path != '/measure':
            raise ValueError(path)
        self.time += 5 + int(channel != self.channel); self.channel = channel
        signal = active and distance <= source.radius
        if signal and source.direction is not None and distance > 1e-12:
            u = np.array([math.cos(math.radians(source.direction)), math.sin(math.radians(source.direction))])
            signal = float(u @ (point-np.array([source.x, source.y]))) >= -1e-9
        result = {'accepted': True, 'virtual_time_s': self.time}
        if not signal:
            result['measure_result'] = 'no_signal'
        elif distance <= 5:
            result['measure_result'] = 'near'
        else:
            angle = math.degrees(math.atan2(source.y-point[1], source.x-point[0]))
            result.update(measure_result='direction', svd_deg=round((angle+self.error(point, channel)) % 360, 2) % 360)
        if self.time >= 360000:
            raise TimeoutError('虚拟世界100小时上限')
        return result


def run_case(question, seed, policy, config_override=None, keep_history=False):
    sources, meta = scenario(question, seed)
    backend = LocalSimulator(sources, seed, meta['error_mode'])
    truths = {s.channel: np.array([s.x, s.y]) for s in sources}
    audits = {'updates': 0, 'containment_failures': 0}

    def independent_audit(channel, polygon):
        audits['updates'] += 1
        if not contains(polygon, truths[channel], eps=1e-3):
            audits['containment_failures'] += 1
            raise AssertionError('独立评估器发现真值不在可行域')

    cfg = configuration(policy)
    for key, value in (config_override or {}).items():
        setattr(cfg, key, value)
    robot = Robot(backend, question, cfg, audit_callback=independent_audit)
    result = robot.run()
    result.update(meta)
    result.update(audits)
    result['clear_fraction'] = result['cleared']/len(sources)
    result['full_success'] = result['status'] == 'completed' and result['cleared'] == len(sources)
    result['evidence_type'] = 'synthetic_local_simulation'
    if keep_history:
        result['sources_for_evaluation_only'] = [vars(s) for s in sources]
        result['history'] = robot.history
    return result


def summarize(results):
    groups = {}
    for r in results:
        groups.setdefault((r['question'], r['policy']), []).append(r)
    output = []
    for (question, policy), runs in groups.items():
        times = np.array([r['average_time_s'] for r in runs if r['average_time_s'] is not None])
        output.append({'question': question, 'policy': policy, 'cases': len(runs),
                       'full_successes': sum(r['full_success'] for r in runs),
                       'full_success_rate': sum(r['full_success'] for r in runs)/len(runs),
                       'mean_clear_fraction': float(np.mean([r['clear_fraction'] for r in runs])),
                       'mean_average_time_s': float(np.mean(times)),
                       'median_average_time_s': float(np.median(times)),
                       'p90_average_time_s': float(np.quantile(times, .9)),
                       'max_average_time_s': float(np.max(times)),
                       'pooled_average_time_s': sum(r['virtual_time_s'] for r in runs)/max(1,sum(r['cleared'] for r in runs)),
                       'mean_real_time_s': float(np.mean([r['real_time_s'] for r in runs])),
                       'containment_failures': sum(r['containment_failures'] for r in runs),
                       'mean_costs': {k: float(np.mean([r['costs'][k] for r in runs])) for k in runs[0]['costs']}})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=4)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--questions', type=int, nargs='+', default=[3, 4])
    parser.add_argument('--policies', nargs='+', default=['baseline', 'coverage_only', 'active', 'proposed'])
    parser.add_argument('--out', default='输出/本地仿真')
    parser.add_argument('--probe-radius', type=float)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = []
    for q in args.questions:
        for seed in range(args.seed_start, args.seed_start+args.count):
            for policy in args.policies:
                overrides = {} if args.probe_radius is None else {'probe_radius': args.probe_radius}
                r = run_case(q, seed, policy, overrides)
                results.append(r)
                print(f'Q{q} seed={seed} {policy}: {r["status"]}, {r["cleared"]}/{r["source_count"]}, '
                      f'{r["average_time_s"]:.2f}s/source, real={r["real_time_s"]:.2f}s', flush=True)
                (out/'runs.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = summarize(results)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (out/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['question', 'seed', 'policy', 'source_count', 'cleared', 'clear_fraction', 'full_success',
                  'virtual_time_s', 'average_time_s', 'real_time_s', 'pattern', 'error_mode']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
