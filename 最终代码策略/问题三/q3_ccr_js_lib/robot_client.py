"""Offline robot algorithm; the run_local entry point verifies server identity."""
import json
import io
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clip(poly, a, b, c):
    if not poly:
        return []
    result = []
    previous = poly[-1]
    fp = a * previous[0] + b * previous[1] - c
    for point in poly:
        fq = a * point[0] + b * point[1] - c
        if (fp <= 1e-8) != (fq <= 1e-8):
            t = fp / (fp - fq)
            result.append((previous[0] + t * (point[0] - previous[0]),
                           previous[1] + t * (point[1] - previous[1])))
        if fq <= 1e-8:
            result.append(point)
        previous, fp = point, fq
    return result


def arena():
    # Circumscribed polygon: never discards a valid point in the target disk.
    r = 1800 / math.cos(math.pi / 64)
    return [(r * math.cos(2 * math.pi * k / 64),
             r * math.sin(2 * math.pi * k / 64)) for k in range(64)]


def update(poly, position, bearing):
    angle = math.radians(bearing)
    # Includes the half-centidegree rounding of the returned bearing.
    epsilon = math.radians(1.005001)
    for theta, sign in ((angle - epsilon, 1), (angle + epsilon, -1)):
        a, b = sign * math.sin(theta), -sign * math.cos(theta)
        poly = clip(poly, a, b, a * position[0] + b * position[1])
    for k in range(32):
        a, b = math.cos(2 * math.pi * k / 32), math.sin(2 * math.pi * k / 32)
        poly = clip(poly, a, b, a * position[0] + b * position[1] + 1500.000001)
    if not poly:
        raise RuntimeError('Bearing constraints are inconsistent; stopped instead of guessing.')
    return poly


def center_radius(poly):
    # Start from a diameter midpoint, then expand to enclose all vertices.
    p = max(poly, key=lambda p: distance(p, poly[0]))
    q = max(poly, key=lambda q: distance(q, p))
    center = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
    radius = distance(p, q) / 2
    for point in poly:
        d = distance(center, point)
        if d > radius:
            nr = (radius + d) / 2
            t = (nr - radius) / d
            center = (center[0] + t * (point[0] - center[0]),
                      center[1] + t * (point[1] - center[1]))
            radius = nr
    return center, max(distance(center, p) for p in poly)


def stations(problem):
    if problem == 3:
        # Center + eight ring points cover the whole arena with radius < 1000.
        return [(0.0, 0.0)] + [(1200 * math.cos(k * math.pi / 4),
                                1200 * math.sin(k * math.pi / 4)) for k in range(8)]
    # Triangular lattice covering radius 800/sqrt(3) < 500.
    # Every directional reception half-disk of radius >=1000 contains a disk
    # of radius 500; a lattice point lies inside it. The radius-2800 cutoff
    # retains every such witness for a source in the radius-1800 arena.
    points = []
    for row in range(-5, 6):
        for column in range(-5, 6):
            x = 800 * column + (400 if row % 2 else 0)
            y = row * 400 * math.sqrt(3)
            if math.hypot(x, y) <= 2800:
                points.append((x, y))
    return points


class Robot:
    def __init__(self, robot_id, problem, log_path, base_url='http://127.0.0.1:2027'):
        self.robot_id, self.problem, self.base_url = robot_id, problem, base_url
        self.position, self.channel = (0.0, 0.0), 1
        self.seq, self.virtual_time, self.cleared = 0, 0.0, set()
        self.polygons, self.good, self.attempts = {}, {}, {}
        self.measure_count, self.clear_count = 0, 0
        self.started = time.monotonic()
        self.deadline = self.started + 1180
        self.log = Path(log_path).open('x', encoding='utf-8') if log_path is not None else io.StringIO()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(self, path, extra=None):
        if path != '/exit' and time.monotonic() >= self.deadline:
            raise TimeoutError('Local runtime safety margin reached.')
        self.seq += 1
        payload = {'arena_id': 'default', 'robot_id': self.robot_id,
                   'request_id': f'practice-{self.seq}'}
        payload.update(extra or {})
        data = json.dumps(payload, allow_nan=False).encode('utf-8')
        response = None
        for retry in range(3):
            try:
                request = urllib.request.Request(self.base_url + path, data=data,
                          headers={'Content-Type': 'application/json'}, method='POST')
                with self.opener.open(request, timeout=5) as result:
                    response = json.load(result)
                break
            except urllib.error.HTTPError as exc:
                raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode("utf-8", "replace")}') from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if retry == 2:
                    raise
                time.sleep(0.2 * (retry + 1))
        # Preserve action/response evidence without recording the team identity.
        recorded = dict(payload, robot_id='<TEAM_ID>')
        self.log.write(json.dumps({'path': path, 'request': recorded, 'response': response},
                                  ensure_ascii=False) + '\n')
        self.log.flush()
        if response.get('accepted') is not True:
            raise RuntimeError(f'{path} rejected: {response}')
        self.virtual_time = response['virtual_time_s']
        if 'position' in payload:
            self.position = (payload['position']['x'], payload['position']['y'])
        return response

    def measure(self, point, channel):
        point = (round(point[0], 6), round(point[1], 6))
        response = self.post('/measure', {'position': {'x': point[0], 'y': point[1]},
                                          'channel': channel})
        self.channel = channel
        self.measure_count += 1
        self.attempts.setdefault(channel, []).append(point)
        kind = response['measure_result']
        if kind == 'near':
            self.clear(point, channel)
        elif kind == 'direction':
            self.polygons[channel] = update(self.polygons.get(channel, arena()), point,
                                            response['svd_deg'])
            self.good.setdefault(channel, []).append((point, response['svd_deg']))
        return kind

    def clear(self, point, channel):
        response = self.post('/clear', {'position': {'x': round(point[0], 6),
                                                    'y': round(point[1], 6)},
                                       'channel': channel})
        self.clear_count += 1
        if response['clear_result'] == 'success':
            self.cleared.add(channel)
            print(f'Cleared channel {channel}; total={len(self.cleared)}; '
                  f'virtual_time={self.virtual_time:.2f}s', flush=True)
            return True
        return False

    def resolve(self, channel):
        for step in range(24):
            if channel in self.cleared:
                return
            poly = self.polygons[channel]
            center, radius = center_radius(poly)
            if radius <= 19.5:
                if not self.clear(center, channel):
                    raise RuntimeError('Clear failed inside a certified 19.5m enclosing circle.')
                return
            previous, bearing = self.good[channel][-1]
            angle = math.radians(bearing)
            ux, uy = math.cos(angle), math.sin(angle)
            if step == 0 and len(self.good[channel]) == 1:
                # Obtain a cross-bearing while reducing the source distance.
                point = (previous[0] + 150 * ux - 180 * uy,
                         previous[1] + 150 * uy + 180 * ux)
            elif step % 3 == 1:
                point = center
            else:
                # Approach from the latest signal-bearing side, with alternate
                # lateral offsets to escape a directional coverage boundary.
                d = distance(previous, center)
                back = min(100, d * 0.25)
                lateral = min(90, max(30, radius * 0.25)) * (-1 if step % 2 else 1)
                point = (center[0] - back * ux - lateral * uy,
                         center[1] - back * uy + lateral * ux)
            if any(distance(point, old) < 2 for old in self.attempts[channel]):
                point = (point[0] + 25 * math.cos(step * 2.4),
                         point[1] + 25 * math.sin(step * 2.4))
            self.measure(point, channel)
            if step % 3 == 1 and channel not in self.cleared:
                if self.clear(point, channel):
                    return
        # Bounded fallback covers the current polygon by radius-20 clear disks.
        # Clear operates optically and therefore works from either emission side.
        poly = self.polygons[channel]
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        fallback = [(x, y)
                    for x in range(math.floor(min(xs) / 25) * 25, math.ceil(max(xs) / 25) * 25 + 1, 25)
                    for y in range(math.floor(min(ys) / 25) * 25, math.ceil(max(ys) / 25) * 25 + 1, 25)]
        if len(fallback) > 2500:
            raise RuntimeError('Localization did not converge; fallback exceeds its action budget.')
        fallback.sort(key=lambda p: distance(p, self.position))
        for point in fallback:
            if self.clear(point, channel):
                return
        raise RuntimeError('Exhausted the localization fallback without clearing the source.')

    def survey_stations(self):
        return stations(self.problem)

    def choose_station(self, remaining):
        return min(remaining, key=lambda p: distance(p, self.position))

    def after_resolve(self):
        pass

    def choose_pending(self, pending):
        return min(pending, key=lambda c: distance(center_radius(self.polygons[c])[0], self.position))

    def refresh_pending(self, pending):
        return pending

    def survey(self, point):
        channels = [c for c in range(1, 21) if c not in self.cleared]
        if self.channel in channels:
            channels.remove(self.channel)
            channels.insert(0, self.channel)
        for channel in channels:
            self.measure(point, channel)

    def run(self):
        entry = self.post('/enter')
        self.started = time.monotonic()
        self.deadline = self.started + max(0, entry['remaining_real_duration_s'] - 15)
        print(f'Entered problem {self.problem} practice; available seconds='
              f'{entry["remaining_real_duration_s"]}', flush=True)
        remaining = self.survey_stations()
        completed = 0
        error = None
        try:
            while remaining and len(self.cleared) < 16:
                point = self.choose_station(remaining)
                remaining.remove(point)
                self.survey(point)
                pending = self.refresh_pending(set(self.polygons) - self.cleared)
                while pending:
                    channel = self.choose_pending(pending)
                    self.resolve(channel)
                    pending.remove(channel)
                    self.after_resolve()
                    pending = self.refresh_pending(pending)
                completed += 1
                print(f'Survey station {completed}/{completed + len(remaining)}; '
                      f'cleared={len(self.cleared)}; actions={self.seq}', flush=True)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            print(error, flush=True)
        exit_response = None
        try:
            exit_response = self.post('/exit')
        except Exception as exc:
            error = (error + '; ' if error else '') + f'Exit failed: {exc}'
        summary = {'problem': self.problem, 'mode': 'local_offline',
                   'cleared_count': len(self.cleared), 'cleared_channels': sorted(self.cleared),
                   'virtual_time_s': self.virtual_time,
                   'average_clear_time_s': self.virtual_time / len(self.cleared) if self.cleared else None,
                   'client_elapsed_s': time.monotonic() - self.started,
                   'measure_actions': self.measure_count, 'clear_actions': self.clear_count,
                   'stations_completed': completed,
                   'coverage_completed': not remaining or len(self.cleared) == 16,
                   'exit_response': exit_response, 'error': error,
                   'actual_source_count': None,
                   'note': 'The local runner fills ground truth after the session ends.'}
        self.log.close()
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary


if __name__ == '__main__':
    from run_local import main as run_offline
    run_offline()
