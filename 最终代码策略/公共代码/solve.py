"""四问共用求解器；真实接口只允许显式核验后的演练会话。"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import argparse
import json
import math
import time
import uuid
import numpy as np
import requests
from geometry import (ANGLE_ERROR, disk_outer, observe, minimum_enclosing_circle,
                      nearest_clear_point, observation_candidates,
                      sampled_posterior_score, optical_cover, q1_intersection)


@dataclass
class PolicyConfig:
    name: str = 'proposed'
    coverage: str = 'compact'
    nearest_clear: bool = True
    probe_radius: float = 100.
    max_measure_attempts: int = 5
    delta: float = ANGLE_ERROR
    clear_margin: float = 19.9
    active_score: bool = False
    q3_ring_radius: float = 1150.


def configuration(name):
    if name == 'baseline':
        return PolicyConfig(name=name, coverage='original', nearest_clear=False,
                            probe_radius=0, max_measure_attempts=5, active_score=False)
    if name == 'coverage_only':
        return PolicyConfig(name=name, nearest_clear=False, probe_radius=0,
                            active_score=False)
    if name == 'active':
        return PolicyConfig(name=name, probe_radius=0, active_score=True)
    if name == 'proposed':
        return PolicyConfig()
    raise ValueError(name)


def coverage_stations(question, family='compact', q3_radius=1150.):
    if question == 3:
        r = 1600 if family == 'original' else q3_radius
        angles = np.arange(6) * np.pi / 3
        return np.vstack(([0, 0], r * np.c_[np.cos(angles), np.sin(angles)]))
    if family == 'original':
        return np.array([(900*(m+n/2), 900*np.sqrt(3)*n/2)
                         for m in range(-5, 6) for n in range(-5, 6)
                         if m*m+m*n+n*n <= 9], dtype=float)
    angles = np.arange(12) * np.pi / 6
    inner = 990 * np.c_[np.cos(angles+np.pi/12), np.sin(angles+np.pi/12)]
    outer = 1810 / np.cos(np.pi/12) * np.c_[np.cos(angles), np.sin(angles)]
    return np.vstack(([0, 0], inner, outer))


def open_route(points, start):
    """最近邻初始化加开放路线2-opt；不要求回到原点。"""
    todo = [np.asarray(p) for p in points]
    route, x = [], np.asarray(start)
    while todo:
        i = min(range(len(todo)), key=lambda i: np.linalg.norm(todo[i]-x))
        x = todo.pop(i); route.append(x)
    path = [np.asarray(start)] + route
    for _ in range(20):
        changed = False
        for i in range(1, len(path)-1):
            for j in range(i+1, len(path)):
                old = np.linalg.norm(path[i]-path[i-1])
                new = np.linalg.norm(path[j]-path[i-1])
                if j+1 < len(path):
                    old += np.linalg.norm(path[j+1]-path[j])
                    new += np.linalg.norm(path[j+1]-path[i])
                if new < old-1e-7:
                    path[i:j+1] = reversed(path[i:j+1]); changed = True
        if not changed:
            break
    return path[1:]


@dataclass
class Belief:
    status: str = 'undiscovered'
    poly: np.ndarray = field(default_factory=disk_outer)
    observations: list = field(default_factory=list)
    attempted: list = field(default_factory=list)
    failed_clears: list = field(default_factory=list)
    scanned: set = field(default_factory=set)
    radius_history: list = field(default_factory=list)


class PracticeHTTP:
    def __init__(self, robot_id, guard_path, question, base_url='http://127.0.0.1:2026'):
        guard = json.loads(Path(guard_path).read_text(encoding='utf-8-sig'))
        if guard.get('mode') != 'practice' or guard.get('question') != question:
            raise RuntimeError('仅允许经过界面核验的演练模式')
        if not guard.get('case_code') or time.time()-guard.get('verified_unix_time', 0) > 300:
            raise RuntimeError('演练核验凭据缺失或已超过5分钟；禁止发送enter')
        if base_url not in ('http://127.0.0.1:2026', 'http://localhost:2026'):
            raise RuntimeError('只允许题给本机演练接口')
        self.guard = guard
        self.robot_id = robot_id
        self.base_url = base_url
        self.session = requests.Session()
        self.session.trust_env = False
        self.prefix = uuid.uuid4().hex[:12]
        self.sequence = 0

    def post(self, path, position=None, channel=None):
        if path not in ('/enter', '/measure', '/clear', '/exit'):
            raise ValueError('未知动作')
        self.sequence += 1
        payload = {'arena_id': 'default', 'robot_id': self.robot_id,
                   'request_id': f'{self.prefix}-{self.sequence}'}
        if position is not None:
            payload['position'] = {'x': float(position[0]), 'y': float(position[1])}
            payload['channel'] = int(channel)
        # 同一动作网络重试保持ID、路径和内容完全一致。
        last = None
        for attempt in range(3):
            try:
                response = self.session.post(self.base_url+path, json=payload, timeout=5)
                response.raise_for_status()
                data = response.json()
                if data.get('accepted') is not True:
                    raise RuntimeError(f'{path}: 动作被拒绝，禁止更新机器狗状态')
                return data
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
                if attempt < 2:
                    time.sleep(.3*(attempt+1))
        raise RuntimeError(f'{path}: 重试后仍无确定响应') from last


class FormalHTTP(PracticeHTTP):
    """正式测试专用客户端；协议相同，但只接受短时有效的正式确认凭据。"""
    def __init__(self, robot_id, guard_path, question, base_url='http://127.0.0.1:2026'):
        guard = json.loads(Path(guard_path).read_text(encoding='utf-8-sig'))
        if guard.get('mode') != 'formal' or guard.get('question') != question:
            raise RuntimeError('正式入口只接受题号一致的正式测试凭据')
        if guard.get('authorization') != 'user_authorized_formal_test':
            raise RuntimeError('缺少用户正式测试授权标记，禁止发送enter')
        if not guard.get('case_code') or time.time()-guard.get('verified_unix_time', 0) > 180:
            raise RuntimeError('正式测试凭据缺失或已超过3分钟；禁止发送enter')
        if base_url not in ('http://127.0.0.1:2026', 'http://localhost:2026'):
            raise RuntimeError('只允许题给本机模拟器接口')
        self.guard = guard
        self.robot_id = robot_id
        self.base_url = base_url
        self.session = requests.Session()
        self.session.trust_env = False
        self.prefix = uuid.uuid4().hex[:12]
        self.sequence = 0


class Robot:
    def __init__(self, backend, question, config=None, log_path=None, audit_callback=None):
        self.backend, self.question = backend, question
        self.config = config or PolicyConfig()
        self.position = np.zeros(2)
        self.channel = 1
        self.virtual_time = 0.
        self.states = {k: Belief() for k in range(1, 21)}
        self.history = []
        self.costs = {'movement_s': 0., 'switch_s': 0., 'measure_s': 0., 'clear_s': 0.}
        self.actions = {'measure': 0, 'clear_success': 0, 'clear_failed': 0}
        self.audit_callback = audit_callback
        self.log_file = None
        if log_path:
            f = Path(log_path); f.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = f.open('w', encoding='utf-8')
        self.started = time.monotonic()
        self.deadline = math.inf

    def record(self, data):
        self.history.append(data)
        if self.log_file:
            self.log_file.write(json.dumps(data, ensure_ascii=False)+'\n')
            self.log_file.flush()

    def act(self, path, point=None, channel=None):
        if time.monotonic() > self.deadline-20 and path != '/exit':
            raise TimeoutError('现实时间不足，主动保留退出时间')
        old_time = self.virtual_time
        response = self.backend.post(path, point, channel)
        event = {'path': path, 'response': response}
        if point is not None:
            point = np.asarray(point)
            event.update(position=point.tolist(), channel=int(channel))
            move = float(np.linalg.norm(point-self.position)/5)
            self.costs['movement_s'] += move
            if path == '/measure':
                switch = float(self.channel != channel)
                self.costs['switch_s'] += switch
                self.costs['measure_s'] += 5
                self.actions['measure'] += 1
                expected = move+switch+5
                self.channel = channel
            else:
                success = response['clear_result'] == 'success'
                seconds = 5 if success else 3
                self.costs['clear_s'] += seconds
                self.actions['clear_success' if success else 'clear_failed'] += 1
                expected = move+seconds
            self.position = point.copy()
            if abs(response['virtual_time_s']-old_time-expected) > 2e-4:
                raise RuntimeError('本地计时与模拟器不一致')
        self.virtual_time = response['virtual_time_s']
        self.record(event)
        return response

    def measure(self, point, k, station_id=None):
        b = self.states[k]
        r = self.act('/measure', point, k)
        b.attempted.append(np.asarray(point).copy())
        if station_id is not None:
            b.scanned.add(station_id)
        result = r['measure_result']
        if result == 'near':
            b.status = 'found'
            self.clear(point, k)
        elif result == 'direction':
            b.status = 'found'
            b.observations.append((np.asarray(point).copy(), r['svd_deg']))
            b.poly = observe(b.poly, np.asarray(point), r['svd_deg'], self.config.delta)
            if not len(b.poly):
                raise RuntimeError(f'频道{k}可行集为空：检查误差/量化/单位，禁止伪造位置')
            _, radius = minimum_enclosing_circle(b.poly)
            b.radius_history.append(radius)
            if self.audit_callback:
                self.audit_callback(k, b.poly)
            self.record({'belief_channel': k, 'polygon': b.poly.tolist(), 'mec_radius': radius})
        return result

    def clear(self, point, k):
        success = self.act('/clear', point, k)['clear_result'] == 'success'
        if success:
            self.states[k].status = 'cleared'
        else:
            self.states[k].failed_clears.append(np.asarray(point).copy())
        return success

    def localize_and_clear(self, k):
        b, cfg = self.states[k], self.config
        probes = 0
        for _ in range(cfg.max_measure_attempts):
            if b.status == 'cleared':
                return
            center, radius = minimum_enclosing_circle(b.poly)
            if radius <= cfg.clear_margin:
                q = nearest_clear_point(b.poly, self.position, cfg.clear_margin) if cfg.nearest_clear else center
                if not self.clear(q, k):
                    raise RuntimeError('保证清除点失败，停止并检查模型')
                return
            if radius <= cfg.probe_radius and probes < 2 and all(np.linalg.norm(center-f)>3 for f in b.failed_clears):
                probes += 1
                if self.clear(center, k):
                    return
            candidates = observation_candidates(b.poly, self.position, b.observations, self.question)
            candidates = [s for s in candidates if all(np.linalg.norm(s-a)>2 for a in b.attempted)]
            if self.question == 3:
                safe = [s for s in candidates if np.max(np.linalg.norm(b.poly-s, axis=1)) <= 1000]
                if safe:
                    candidates = safe
            if not candidates:
                break
            if cfg.active_score:
                point = min(candidates, key=lambda s: sampled_posterior_score(b.poly, s, self.position, cfg.delta)[0])
            else:
                point = min(candidates, key=lambda s: np.linalg.norm(s-self.position))
            self.measure(point, k)
        if b.status == 'cleared':
            return
        center, radius = minimum_enclosing_circle(b.poly)
        if radius <= cfg.clear_margin:
            q = nearest_clear_point(b.poly, self.position, cfg.clear_margin) if cfg.nearest_clear else center
            if self.clear(q, k):
                return
            raise RuntimeError('保证清除点失败')
        first_s, first_a = b.observations[0]
        points = self.optical_points(b.poly, first_s, first_a, b.failed_clears)
        routes = [points, list(reversed(points)), open_route(points, self.position)]
        def route_length(route):
            chain = [self.position]+list(route)
            return sum(np.linalg.norm(y-x) for x, y in zip(chain[:-1], chain[1:]))
        # 保留蛇形覆盖路线作为长度上界，启发式重排只能缩短而不能恶化它。
        route = min(routes, key=route_length)
        for point in route:
            if self.clear(point, k):
                return
        raise RuntimeError(f'频道{k}光学覆盖完成后仍未清除')

    def optical_points(self, poly, first_station, first_bearing, failed):
        return optical_cover(poly, first_station, first_bearing, failed)

    def run(self):
        exit_status = 'completed'
        entered = False
        try:
            enter = self.act('/enter')
            entered = True
            self.deadline = time.monotonic()+enter.get('remaining_real_duration_s', 1200)
            stations = coverage_stations(self.question, self.config.coverage, self.config.q3_ring_radius)
            remaining = list(range(len(stations)))
            while remaining:
                # 当前实际位置变化后重选最近扫描站，避免强制回到原路线。
                route = open_route([stations[i] for i in remaining], self.position)
                point = route[0]
                idx = min(remaining, key=lambda i: np.linalg.norm(stations[i]-point))
                remaining.remove(idx)
                channels = [k for k, b in self.states.items() if b.status == 'undiscovered']
                channels.sort(key=lambda k: (k != self.channel, k))
                for k in channels:
                    self.measure(point, k, idx)
                pending = [k for k, b in self.states.items() if b.status == 'found']
                while pending:
                    k = min(pending, key=lambda j: np.linalg.norm(self.states[j].poly.mean(axis=0)-self.position))
                    pending.remove(k)
                    self.localize_and_clear(k)
                if sum(b.status == 'cleared' for b in self.states.values()) == 16:
                    break
            for b in self.states.values():
                if b.status == 'undiscovered' and len(b.scanned) == len(stations):
                    b.status = 'absent_certified'
            if not (all(b.status in ('cleared', 'absent_certified') for b in self.states.values()) or
                    sum(b.status == 'cleared' for b in self.states.values()) == 16):
                raise RuntimeError('结束证书不完整')
        except Exception as exc:
            exit_status = f'failed: {type(exc).__name__}: {exc}'
            self.record({'failure': exit_status})
        finally:
            try:
                if entered:
                    self.act('/exit')
            except Exception as exc:
                self.record({'exit_error': str(exc)})
                if exit_status == 'completed':
                    exit_status = 'failed: exit response unavailable'
            if self.log_file:
                self.log_file.close()
        n = sum(b.status == 'cleared' for b in self.states.values())
        return {'question': self.question, 'policy': self.config.name, 'status': exit_status,
                'cleared': n, 'virtual_time_s': self.virtual_time,
                'average_time_s': self.virtual_time/n if n else None,
                'real_time_s': time.monotonic()-self.started, 'costs': self.costs, 'actions': self.actions,
                'states': {str(k): b.status for k, b in self.states.items()},
                'station_count': len(coverage_stations(self.question, self.config.coverage)),
                'configuration': asdict(self.config)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--practice-question', type=int, choices=[3, 4])
    parser.add_argument('--robot-id')
    parser.add_argument('--practice-guard')
    parser.add_argument('--policy', default='proposed', choices=['baseline', 'coverage_only', 'active', 'proposed'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--q1-input')
    args = parser.parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    if args.q1_input:
        data = json.loads(Path(args.q1_input).read_text(encoding='utf-8-sig'))
        result = q1_intersection(data['stations'], data['bearings'], data.get('delta', 1))
    else:
        if not all([args.practice_question, args.robot_id, args.practice_guard]):
            parser.error('真实连接必须提供practice-question、robot-id及5分钟内的practice-guard')
        client = PracticeHTTP(args.robot_id, args.practice_guard, args.practice_question)
        robot = Robot(client, args.practice_question, configuration(args.policy), out.with_suffix('.jsonl'))
        result = robot.run()
        result['case_code'] = client.guard['case_code']
        result['evidence_type'] = 'official_practice_not_formal'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
