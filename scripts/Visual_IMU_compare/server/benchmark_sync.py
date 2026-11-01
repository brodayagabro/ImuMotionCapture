#!/usr/bin/env python3
"""LED synchronization benchmark: unchanged RTSP publishers, receive-time analysis.

Run in the existing server environment. No pose models or GPU are loaded.
See SYNC_BENCHMARK.md for the measurement limits and LED stimulus protocol.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import asynccontextmanager
import csv
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit, urlunsplit
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
import uvicorn

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger('led-sync')
FIELDS = ['frame_id', 'segment', 'receive_monotonic_ns', 't_s', 'brightness',
          'width', 'height', 'roi_x', 'roi_y', 'roi_w', 'roi_h', 'roi_video_frame']
SCOPE = ('Relative timing of visible LED transitions after server decode; includes '
         'camera sampling/exposure, encoding, transport and decoding. '
         'Not absolute latency or a measurement of exposure synchronization alone.')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def finite(value, low, high, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def checked_roi(roi, size):
    if len(roi) != 4 or len(size) != 2 or any(type(v) is not int for v in [*roi, *size]):
        raise ValueError('ROI must be four integer pixels; image_size must be [width,height]')
    x, y, w, h = roi
    width, height = size
    if min(width, height) < 2 or min(x, y) < 0 or min(w, h) < 2 or x+w > width or y+h > height:
        raise ValueError('ROI is outside the source image or smaller than 2x2')
    return list(roi)


@dataclass
class Camera:
    id: str
    source: str
    roi: list | None = None
    image_size: list | None = None


def read_cameras(args):
    if args.demo:
        return [Camera(f'cam{i+1}', f'demo:{i}', [280, 200, 80, 80], [640, 480]) for i in range(3)]
    entries = []
    if args.config:
        entries.extend(json.loads(args.config.read_text())['cameras'])
    for spec in args.camera:
        name, sep, source = spec.partition('=')
        if not sep:
            raise ValueError('--camera needs NAME=RTSP_URL')
        entries.append(dict(id=name, source=source))
    rois = json.loads(args.rois.read_text()) if args.rois else {}
    cameras = []
    for entry in entries:
        name, source = entry['id'], entry['source']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', name):
            raise ValueError('Camera IDs must be 1..48 letters/digits/_/-')
        if not isinstance(source, str) or not source.startswith(('rtsp://', 'rtsps://')):
            raise ValueError('This receiver accepts RTSP URLs; use demo for offline validation')
        item = rois.get(name, entry)
        roi, size = item.get('roi'), item.get('image_size')
        if roi is not None:
            checked_roi(roi, size or [])
        cameras.append(Camera(name, source, roi, size))
    if not 2 <= len(cameras) <= 8 or len({c.id for c in cameras}) != len(cameras):
        raise ValueError('Specify 2..8 cameras with unique IDs')
    if len({c.source for c in cameras}) != len(cameras):
        raise ValueError('Duplicate source URLs do not represent independent cameras')
    return cameras


def safe_source(source):
    parts = urlsplit(source)
    host = parts.hostname or ''
    if ':' in host:
        host = '['+host+']'
    return urlunsplit((parts.scheme, host + (f':{parts.port}' if parts.port else ''), parts.path, '', ''))


def led_value(image, roi, channel):
    x, y, w, h = roi
    crop = image[y:y+h, x:x+w]
    plane = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if channel == 'gray' else crop[:, :, {'blue':0, 'green':1, 'red':2}[channel]]
    return float(np.mean(plane))


def events_from_samples(rows, min_contrast=20, stable_frames=2):
    """Schmitt thresholds; event time is a bracket midpoint, not sub-frame truth."""
    if len(rows) < 20:
        return [], {'status':'too_few_frames', 'frames':len(rows)}
    ts = np.array([r['t_s'] for r in rows])
    values = np.array([r['brightness'] for r in rows])
    if not np.isfinite(ts).all() or not np.isfinite(values).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('Non-finite samples or non-increasing receive timestamps')
    low_level, high_level = np.percentile(values, [10, 90])
    span = float(high_level-low_level)
    dt = float(np.median(np.diff(ts)))
    gap_limit = max(.2, 4*dt)
    stats = dict(status='ok', frames=len(rows), low_level=float(low_level), high_level=float(high_level),
                 contrast=span, median_frame_interval_ms=dt*1000,
                 delivered_fps=(len(rows)-1)/(ts[-1]-ts[0]),
                 gap_count=int(np.sum(np.diff(ts)>gap_limit)),
                 segments=len({r['segment'] for r in rows}))
    if span < min_contrast:
        stats['status'] = 'low_contrast'
        return [], stats
    low, high = low_level+.35*span, low_level+.65*span
    stats.update(threshold_low=float(low), threshold_high=float(high))
    events = []
    state = candidate = previous = None
    count = 0
    last_stable = first_candidate = None
    for row in rows:
        if previous is not None and (row['segment'] != previous['segment'] or row['t_s']-previous['t_s'] > gap_limit):
            state = candidate = last_stable = first_candidate = None
            count = 0
        previous = row
        value = row['brightness']
        label = 0 if value <= low else 1 if value >= high else None
        if label is None:
            candidate, count = None, 0
            continue
        if label == state:
            last_stable = row
            candidate, count = None, 0
            continue
        if label != candidate:
            candidate, count, first_candidate = label, 1, row
        else:
            count += 1
        if count >= stable_frames:
            if state is not None and last_stable is not None:
                lo, hi = last_stable['t_s'], first_candidate['t_s']
                if hi-lo <= gap_limit:
                    events.append(dict(kind='rise' if label else 'fall', time_s=(lo+hi)/2,
                        lo_s=lo, hi_s=hi, segment=row['segment'],
                        frame_before=last_stable['frame_id'], frame_after=first_candidate['frame_id']))
            state, last_stable, candidate, count = label, row, None, 0
    stats['events'] = len(events)
    return events, stats


def periodic(events):
    periods = []
    rising = [e for e in events if e['kind'] == 'rise']
    for a, b in zip(rising, rising[1:]):
        if a['segment'] == b['segment']:
            periods.append(b['time_s']-a['time_s'])
    return len(periods) >= 6 and np.percentile(periods, 90)-np.percentile(periods, 10) < .12*np.median(periods)


def match_events(reference, camera, offset, tolerance):
    """One-to-one, chronological matches of equal edge polarity."""
    pairs, next_index = [], 0
    times = np.array([e['time_s'] for e in camera])
    for i, event in enumerate(reference):
        target = event['time_s']+offset
        start = max(next_index, int(np.searchsorted(times, target-tolerance)))
        end = int(np.searchsorted(times, target+tolerance, side='right'))
        choices = [j for j in range(start, end) if camera[j]['kind'] == event['kind']]
        if choices:
            j = min(choices, key=lambda k: abs(times[k]-target))
            pairs.append((i, j))
            next_index = j+1
    return pairs


def compare_events(reference, camera, max_lag=1.0, tolerance=.08, min_events=12):
    result = dict(status='insufficient_events', accepted=False, matched_events=0,
                  suggested_offset_ms=None, scope=SCOPE)
    if min(len(reference), len(camera)) < min_events:
        return result
    # Candidate translations from nearby same-polarity events, not frame indices.
    times = np.array([e['time_s'] for e in camera])
    differences = []
    for event in reference:
        lo, hi = np.searchsorted(times, [event['time_s']-max_lag, event['time_s']+max_lag])
        differences.extend(camera[j]['time_s']-event['time_s'] for j in range(lo, hi)
                           if camera[j]['kind'] == event['kind'])
    if not differences:
        result['status'] = 'no_matches_in_search_range'
        return result
    step = tolerance/3
    bins = np.round(np.array(differences)/step).astype(int)
    unique, counts = np.unique(bins, return_counts=True)
    candidates = [float(np.median(np.array(differences)[bins == b]))
                  for b in unique[np.argsort(counts)[-40:]]]
    hypotheses = []
    for offset in candidates:
        for _ in range(2):
            pairs = match_events(reference, camera, offset, tolerance)
            if not pairs:
                break
            offset = float(np.median([camera[j]['time_s']-reference[i]['time_s'] for i, j in pairs]))
        if not pairs or abs(offset) > max_lag:
            continue
        residual = [abs(camera[j]['time_s']-reference[i]['time_s']-offset) for i, j in pairs]
        score = len(pairs)-.25*sum(r/tolerance for r in residual)
        hypotheses.append((score, offset, pairs))
    if not hypotheses:
        result['status'] = 'no_matches_in_search_range'
        return result
    hypotheses.sort(key=lambda h: h[0], reverse=True)
    score, offset, pairs = hypotheses[0]
    alternative = next((h for h in hypotheses[1:] if abs(h[1]-offset) > 3*tolerance), None)
    coverage = len(pairs)/min(len(reference), len(camera))
    is_periodic = periodic(reference) or periodic(camera)
    ambiguous = is_periodic or alternative is not None and alternative[0] >= .9*score
    boundary = abs(offset) > max_lag-tolerance
    status = ('insufficient_matches' if len(pairs) < min_events or coverage < .75 else
              'ambiguous_pattern' if ambiguous else 'lag_at_search_boundary' if boundary else 'estimate_available')
    matched = []
    for i, j in pairs:
        a, b = reference[i], camera[j]
        matched.append(dict(reference_event=i, camera_event=j, kind=a['kind'], time_s=a['time_s'],
            delta_ms=1000*(b['time_s']-a['time_s']),
            delta_lo_ms=1000*(b['lo_s']-a['hi_s']), delta_hi_ms=1000*(b['hi_s']-a['lo_s']),
            reference_segment=a['segment'], camera_segment=b['segment'],
            reference_frame=a['frame_after'], camera_frame=b['frame_after']))
    delta = np.array([p['delta_ms'] for p in matched])
    median = float(np.median(delta))
    trend = None
    detrended_p95 = None
    trend_span = matched[-1]['time_s']-matched[0]['time_s']
    continuous = len({(p['reference_segment'], p['camera_segment']) for p in matched}) == 1
    if len(matched) >= min_events and trend_span >= 20 and continuous:
        x = np.array([p['time_s'] for p in matched]); x -= np.mean(x)
        slope, intercept = np.polyfit(x, delta, 1)
        residual = delta-(slope*x+intercept)
        scale = max(1.0, 1.4826*float(np.median(np.abs(residual-np.median(residual)))))
        keep = np.abs(residual-np.median(residual)) <= 4*scale
        if np.count_nonzero(keep) >= min_events:
            slope, intercept = np.polyfit(x[keep], delta[keep], 1)
            trend = float(slope*60)  # ms/min, not oscillator ppm.
            detrended_p95 = float(np.percentile(np.abs(delta-(slope*x+intercept)), 95))
    result.update(status=status, accepted=status == 'estimate_available', matched_events=len(pairs),
        match_fraction=coverage, periodic_pattern=bool(is_periodic),
        offset_median_ms=median, offset_p05_ms=float(np.percentile(delta, 5)),
        offset_p95_ms=float(np.percentile(delta, 95)),
        absolute_skew_p95_ms=float(np.percentile(np.abs(delta), 95)),
        jitter_p95_ms=float(np.percentile(np.abs(delta-median), 95)),
        transition_bracket_halfwidth_median_ms=float(np.median([(p['delta_hi_ms']-p['delta_lo_ms'])/2 for p in matched])),
        observed_trend_ms_per_min=trend, detrended_jitter_p95_ms=detrended_p95,
        trend_span_s=trend_span, continuous_segment=continuous,
        alternative_offset_ms=alternative[1]*1000 if alternative else None,
        suggested_offset_ms=-median if status == 'estimate_available' else None, matches=matched)
    # A conservative sampling-scale diagnostic, not a confidence interval or a
    # claim that a linear trend comes from camera oscillator drift.
    floor = 2*result['transition_bracket_halfwidth_median_ms']*60/max(trend_span, 1e-9)
    result['trend_sampling_floor_ms_per_min'] = floor
    result['trend_above_sampling_floor'] = bool(trend is not None and abs(trend)>floor)
    return result


def load_samples(path):
    rows = []
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if len(rows) >= 1_000_000:
                raise ValueError('Sample count exceeds analysis limit of 1,000,000 per camera')
            row = {k: float(row[k]) if k in ('t_s', 'brightness') else int(row[k]) for k in FIELDS}
            rows.append(row)
    return rows


def analyze(directory, max_lag=None, match_window=None, min_contrast=None):
    manifest = json.loads((directory/'session.json').read_text())
    options = manifest['analysis']
    max_lag = options['max_lag_ms'] if max_lag is None else max_lag
    match_window = options['match_window_ms'] if match_window is None else match_window
    min_contrast = options['min_contrast'] if min_contrast is None else min_contrast
    finite(max_lag,10,10000,'max_lag_ms')
    finite(match_window,1,1000,'match_window_ms')
    finite(min_contrast,1,255,'min_contrast')
    if match_window >= max_lag:
        raise ValueError('Match window must be smaller than the lag search range')
    report = dict(schema='visual-led-sync/1', session_id=manifest['session_id'], scope=SCOPE,
                  synthetic=manifest['synthetic'], reference=manifest['reference'],
                  analysis=dict(max_lag_ms=max_lag, match_window_ms=match_window, min_contrast=min_contrast),
                  cameras={}, comparisons={}, traces={}, warnings=list(manifest.get('warnings', [])))
    if not manifest.get('completed'):
        report['warnings'].append('Session was not finalized; results are provisional')
    for spec in manifest['cameras']:
        name = spec['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,48}', name):
            raise ValueError('Invalid camera ID in session')
        rows = load_samples(directory/f'{name}.csv')
        events, stats = events_from_samples(rows, min_contrast)
        report['cameras'][name] = dict(**stats, events_data=events)
        # Downsample only the plot; analysis always uses every recorded row.
        step = max(1, math.ceil(len(rows)/2400))
        report['traces'][name] = [[r['t_s'], r['brightness']] for r in rows[::step]]
    reference = report['reference']
    for name in report['cameras']:
        if name == reference:
            continue
        result = compare_events(report['cameras'][reference]['events_data'],
                                report['cameras'][name]['events_data'], max_lag/1000, match_window/1000)
        result['reference'] = reference
        if report['warnings']:
            result.update(status='capture_invalid',accepted=False,suggested_offset_ms=None)
        report['comparisons'][name] = result
        with (directory/f'{name}-matches.csv').open('w', newline='') as handle:
            matches = result.get('matches', [])
            if matches:
                writer = csv.DictWriter(handle, fieldnames=list(matches[0]))
                writer.writeheader(); writer.writerows(matches)
    dump(directory/'report.json', report)
    (directory/'report.html').write_text(report_page(report), encoding='utf-8')
    return report


def pattern_edges(duration=7200):
    rng = np.random.default_rng(20261030)
    times, t = [], 1.0
    while t < duration:
        times.append(t)
        t += float(rng.uniform(.20, .85))
    return np.array(times)


PATTERN = pattern_edges()


def synthetic_value(t):
    return 220.0 if np.searchsorted(PATTERN, t, side='right') % 2 else 25.0


def demo_session(output, duration=120):
    directory = output / ('demo-'+uuid.uuid4().hex[:8])
    directory.mkdir(parents=True)
    specs = [Camera('cam1', 'demo:0'), Camera('cam2', 'demo:1'), Camera('cam3', 'demo:2')]
    manifest = dict(schema='visual-led-sync/1', session_id=directory.name, synthetic=True,
        reference='cam1', scope=SCOPE, completed=True, cameras=[asdict(c) for c in specs],
        analysis=dict(max_lag_ms=1000, match_window_ms=80, min_contrast=20),
        expected=dict(cam2_delay_ms=120, cam3_delay_ms=-40, cam3_trend_ms_per_min=12))
    rng = np.random.default_rng(33)
    for i, spec in enumerate(specs):
        with (directory/f'{spec.id}.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader()
            for n, sample_t in enumerate(np.arange(0, duration, 1/30)):
                delay, slope = ((0, 0), (.12, 0), (-.04, .0002))[i]
                t = sample_t + float(rng.uniform(0, .003))
                row = dict(frame_id=n+1, segment=1, receive_monotonic_ns=int(t*1e9), t_s=t,
                    brightness=float(synthetic_value(sample_t-delay-slope*sample_t)+rng.normal(0, 1)),
                    width=640, height=480, roi_x=280, roi_y=200, roi_w=80, roi_h=80, roi_video_frame=-1)
                writer.writerow(row)
    dump(directory/'session.json', manifest)
    return directory, analyze(directory)


class Receiver:
    def __init__(self, engine, spec, index):
        self.engine, self.spec, self.index = engine, spec, index
        self.lock = threading.Lock()
        self.image = None
        self.received_ns = 0
        self.sequence = 0
        self.segment = 0
        self.status = 'connecting'
        self.brightness = None
        self.tail = deque(maxlen=300)
        self.preview = None
        self.preview_time = 0
        self.thread = threading.Thread(target=self.run, daemon=True, name='led-'+spec.id)

    def state(self):
        with self.lock:
            age = (time.monotonic_ns()-self.received_ns)/1e9 if self.received_ns else None
            return dict(id=self.spec.id, status=self.status, size=list(self.image.shape[1::-1]) if self.image is not None else None,
                        roi=self.spec.roi, frame_id=self.sequence, segment=self.segment,
                        brightness=self.brightness, age_s=age, trace=list(self.tail))

    def accept(self, image, received_ns):
        self.sequence += 1
        try:
            self.engine.ingest(self, image, received_ns)
        except (OSError,cv2.error) as exc:
            with self.engine.lock:
                self.engine.error = f'{self.spec.id}: recording/processing failed: {exc}'
                if self.engine.session:
                    self.engine.session['fatal'] = True
                    self.engine.session['manifest']['warnings'].append(self.engine.error)
            LOG.exception('Capture processing failed')

    def run(self):
        if self.engine.args.demo:
            self.segment = 1
            deadline = time.monotonic()
            while not self.engine.stop.is_set():
                t = time.monotonic()-self.engine.origin
                delay, slope = ((0, 0), (.12, 0), (-.04, .0002))[self.index]
                image = np.full((480,640,3), 15, np.uint8)
                cv2.rectangle(image, (280,200), (359,279), (int(synthetic_value(t-delay-slope*t)),)*3, -1)
                cv2.putText(image, 'SYNTHETIC '+self.spec.id, (20,40), cv2.FONT_HERSHEY_SIMPLEX, .8, (220,220,220), 2)
                self.accept(image, time.monotonic_ns())
                deadline += 1/30
                self.engine.stop.wait(max(0, deadline-time.monotonic()))
            return
        while not self.engine.stop.is_set():
            cap = None
            try:
                cap = cv2.VideoCapture(self.spec.source, cv2.CAP_FFMPEG,
                    [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,3000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,2000])
                if not cap.isOpened():
                    raise OSError('Cannot open source')
                self.segment += 1
                while not self.engine.stop.is_set():
                    ok, image = cap.read()
                    received_ns = time.monotonic_ns()  # Before ROI, disk, JPEG or UI work.
                    if not ok or image is None:
                        break
                    self.accept(image, received_ns)
            except (OSError, cv2.error):
                LOG.warning('%s disconnected; retrying', self.spec.id)
            finally:
                if cap is not None:
                    cap.release()
                with self.lock:
                    self.status, self.image, self.preview = 'reconnecting', None, None
            self.engine.stop.wait(1)


class Engine:
    def __init__(self, args, cameras):
        self.args = args
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.finished = threading.Event()
        self.origin = time.monotonic()
        self.cameras = {c.id: Receiver(self,c,i) for i,c in enumerate(cameras)}
        self.reference = args.reference or cameras[0].id
        if self.reference not in self.cameras:
            raise ValueError('Reference camera is not in the configured list')
        self.session = None
        self.analyzing = False
        self.last_report = None
        self.last_directory = None
        self.error = None
        self.triggered = False
        self.monitor_thread = threading.Thread(target=self.monitor, daemon=True)

    def start(self):
        self.args.output.mkdir(parents=True, exist_ok=True)
        for camera in self.cameras.values():
            camera.thread.start()
        self.monitor_thread.start()

    def shutdown(self):
        self.stop.set()
        for camera in self.cameras.values():
            camera.thread.join(timeout=6)
        # A running analysis must finish writing its report before process exit.
        self.monitor_thread.join()
        if self.session:
            self.end_recording()

    def snapshot(self):
        with self.lock:
            return dict(cameras=[c.state() for c in self.cameras.values()], reference=self.reference,
                recording=self.session is not None, analyzing=self.analyzing, synthetic=self.args.demo,
                elapsed_s=(time.monotonic_ns()-self.session['start_ns'])/1e9 if self.session else 0,
                duration_s=self.session['duration'] if self.session else self.args.duration,
                directory=str(self.last_directory) if self.last_directory else None,
                report_available=self.last_report is not None, error=self.error)

    def set_roi(self, name, roi, size):
        with self.lock:
            if self.session or self.analyzing:
                raise ValueError('Stop recording before changing the ROI')
            camera = self.cameras[name]
            if camera.state()['size'] != size:
                raise ValueError('Image size changed; refresh the preview and select again')
            camera.spec.roi, camera.spec.image_size = checked_roi(roi,size), list(size)
            dump(self.args.output/'rois.json', {c.spec.id:dict(roi=c.spec.roi,image_size=c.spec.image_size)
                                              for c in self.cameras.values()})

    def begin_recording(self, duration=None):
        with self.lock:
            if self.session or self.analyzing:
                raise ValueError('A recording or analysis is already running')
            duration = finite(self.args.duration if duration is None else duration, 5, 3600, 'duration')
            for camera in self.cameras.values():
                state = camera.state()
                if state['status'] != 'online' or state['age_s'] is None or state['age_s'] > 2:
                    raise ValueError(f'{camera.spec.id}: waiting for a live frame')
                if camera.spec.roi is None or camera.spec.image_size != state['size']:
                    raise ValueError(f'{camera.spec.id}: select an LED ROI for the current resolution')
            directory = self.args.output / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:6])
            directory.mkdir(parents=True)
            manifest = dict(schema='visual-led-sync/1', session_id=directory.name, synthetic=self.args.demo,
                created_utc=datetime.now(timezone.utc).isoformat(), reference=self.reference, scope=SCOPE,
                channel=self.args.channel, duration_requested_s=duration, save_roi_video=self.args.save_roi_video,
                software=dict(opencv=cv2.__version__,numpy=np.__version__), completed=False,
                cameras=[dict(**asdict(c.spec), source_redacted=safe_source(c.spec.source)) for c in self.cameras.values()],
                analysis=dict(max_lag_ms=self.args.max_lag_ms, match_window_ms=self.args.match_window_ms,
                              min_contrast=self.args.min_contrast), warnings=[])
            for spec in manifest['cameras']:
                spec.pop('source')
            files, writers = {}, {}
            try:
                for name in self.cameras:
                    files[name] = (directory/f'{name}.csv').open('x', newline='')
                    writers[name] = csv.DictWriter(files[name], fieldnames=FIELDS)
                    writers[name].writeheader()
                dump(directory/'session.json', manifest)
            except BaseException:
                for handle in files.values():
                    handle.close()
                raise
            self.last_report, self.error = None, None
            self.last_directory = directory
            self.session = dict(directory=directory, manifest=manifest, files=files, writers=writers,
                                start_ns=time.monotonic_ns(), duration=duration, videos={}, counts={}, flush=time.monotonic())
            self.triggered = True
            self.finished.clear()
            return directory.name

    def ingest(self, camera, image, received_ns):
        # All session mutations share this lock; timestamps precede lock/disk work.
        with self.lock:
            spec = camera.spec
            size = list(image.shape[1::-1])
            valid = spec.roi is not None and spec.image_size == size
            brightness = led_value(image, spec.roi, self.args.channel) if valid else None
            with camera.lock:
                camera.image, camera.received_ns, camera.brightness = image, received_ns, brightness
                camera.status = 'online'
                if brightness is not None:
                    camera.tail.append([(received_ns/1e9)-self.origin,brightness])
            session = self.session
            if not session or session.get('fatal') or received_ns < session['start_ns']:
                return
            if not valid:
                warning = f'{spec.id}: resolution changed or ROI invalid; frames omitted, end this session and reselect ROI'
                if warning not in session['manifest']['warnings']:
                    session['manifest']['warnings'].append(warning)
                return
            t = (received_ns-session['start_ns'])/1e9
            x,y,w,h = spec.roi
            video_index = -1
            if self.args.save_roi_video:
                key = (spec.id,camera.segment)
                if key not in session['videos']:
                    # AVI is for visual review; its fixed playback FPS is not a timebase.
                    path = session['directory']/f'{spec.id}-segment{camera.segment}-roi.avi'
                    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),30,(320,160))
                    if not writer.isOpened():
                        raise OSError('Cannot open ROI video writer')
                    session['videos'][key] = writer
                    session['counts'][key] = 0
                canvas = np.zeros((160,320,3),np.uint8)
                crop = image[y:y+h,x:x+w]
                factor = min(160/w,120/h)
                crop = cv2.resize(crop,(max(1,round(w*factor)),max(1,round(h*factor))))
                canvas[:crop.shape[0],:crop.shape[1]] = crop
                cv2.putText(canvas,f'{spec.id} f={camera.sequence}',(5,135),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                cv2.putText(canvas,f't={t:.4f}s I={brightness:.1f}',(5,153),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                session['videos'][key].write(canvas)
                video_index = session['counts'][key]
                session['counts'][key] += 1
            row = dict(frame_id=camera.sequence,segment=camera.segment,receive_monotonic_ns=received_ns,t_s=t,
                       brightness=brightness,width=size[0],height=size[1],roi_x=x,roi_y=y,roi_w=w,roi_h=h,
                       roi_video_frame=video_index)
            session['writers'][spec.id].writerow(row)
            if time.monotonic()-session['flush']>1:
                for handle in session['files'].values():
                    handle.flush()
                session['flush'] = time.monotonic()

    def end_recording(self):
        with self.lock:
            if not self.session:
                raise ValueError('No recording is running')
            session, self.session, self.analyzing = self.session, None, True
        try:
            close_error = None
            for handle in session['files'].values():
                try:
                    handle.close()
                except OSError as exc:
                    close_error = exc
            for writer in session['videos'].values():
                writer.release()
            if close_error:
                raise close_error
            session['manifest']['duration_actual_s'] = (time.monotonic_ns()-session['start_ns'])/1e9
            session['manifest']['completed'] = not session.get('fatal',False)
            dump(session['directory']/'session.json', session['manifest'])
            report = analyze(session['directory'])
            with self.lock:
                self.last_report = report
            return report
        except Exception as exc:
            with self.lock:
                self.error = f'Analysis failed: {exc}'
            LOG.exception('Analysis failed; raw CSV retained')
            return None
        finally:
            with self.lock:
                self.analyzing = False
                self.finished.set()

    def monitor(self):
        while not self.stop.wait(.1):
            try:
                if self.args.autostart and not self.triggered:
                    try:
                        self.begin_recording()
                    except ValueError:
                        pass  # Wait for configured live cameras, never choose an ROI automatically.
                with self.lock:
                    expired = self.session and (self.session.get('fatal') or
                        (time.monotonic_ns()-self.session['start_ns'])/1e9 >= self.session['duration'])
                if expired:
                    try:
                        self.end_recording()
                    except ValueError:
                        pass
                # Preview encoding is separated from the decode threads.
                for camera in self.cameras.values():
                    with self.lock, camera.lock:
                        image, received_ns, value = camera.image, camera.received_ns, camera.brightness
                        roi, sequence = camera.spec.roi, camera.sequence
                    if image is None or time.monotonic_ns()-received_ns > 2e9:
                        continue
                    scale = min(1,640/image.shape[1])
                    preview = cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)))
                    if roi:
                        x,y,w,h = [round(v*scale) for v in roi]
                        cv2.rectangle(preview,(x,y),(x+w,y+h),(0,220,255),2)
                    label = f'{camera.spec.id}  f={sequence}  t={received_ns/1e9-self.origin:.3f}s'
                    if value is not None:
                        label += f'  I={value:.1f}'
                    cv2.putText(preview,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                    ok, encoded = cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY,80])
                    if ok:
                        with camera.lock:
                            camera.preview, camera.preview_time = encoded.tobytes(), time.monotonic()
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                LOG.exception('Monitor error')
                self.stop.wait(1)


class RoiRequest(BaseModel):
    roi: list[int]
    image_size: list[int]


class StartRequest(BaseModel):
    duration: float | None = None


def make_app(engine):
    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        try:
            yield
        finally:
            engine.shutdown()
    app = FastAPI(title='LED synchronization benchmark',lifespan=lifespan)

    @app.get('/',response_class=HTMLResponse)
    def index():
        return PAGE

    @app.get('/api/state')
    def state():
        return engine.snapshot()

    @app.get('/frame/{name}.jpg')
    def frame(name: str):
        if name not in engine.cameras:
            raise HTTPException(404)
        camera = engine.cameras[name]
        with camera.lock:
            data = camera.preview if time.monotonic()-camera.preview_time<2 else None
        if data is None:
            raise HTTPException(503,'Waiting for camera')
        return Response(data,media_type='image/jpeg',headers={'Cache-Control':'no-store'})

    @app.post('/api/roi/{name}')
    def roi(name: str, request: RoiRequest):
        if name not in engine.cameras:
            raise HTTPException(404)
        try:
            engine.set_roi(name,request.roi,request.image_size)
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc
        return {'saved':True}

    @app.post('/api/start')
    def start(request: StartRequest):
        try:
            return {'session_id':engine.begin_recording(request.duration)}
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc

    @app.post('/api/stop')
    def stop():
        try:
            engine.end_recording()
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc
        return {'saved':engine.last_report is not None,'error':engine.error}

    @app.get('/api/report')
    def report():
        if engine.last_report is None:
            raise HTTPException(409,'No completed analysis')
        return engine.last_report

    @app.get('/report',response_class=HTMLResponse)
    def report_html():
        if engine.last_report is None:
            raise HTTPException(409,'No completed analysis')
        return report_page(engine.last_report)
    return app


PAGE = '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Проверка синхронизации камер</title><style>
body{background:#111820;color:#edf2f8;font:16px system-ui;margin:0}main{max-width:1350px;margin:auto;padding:24px}
h1{font-size:26px}p{color:#b6c3d0}button,input{font:inherit;padding:8px;margin:4px;border-radius:6px}button{cursor:pointer}
a{color:#7bd7ff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:16px}
.card{background:#1c2835;border-radius:10px;padding:14px}canvas{width:100%;display:block;touch-action:none}#error{color:#ffad9c;white-space:pre-wrap}
</style><main><h1>Проверка синхронизации по светодиоду</h1>
<p>Выделите светодиод прямоугольником на каждом изображении. Используйте одну общую неравномерную последовательность вспышек.</p>
<label>Длительность, с <input id="duration" type="number" min="5" max="3600" value="120"></label>
<button id="start">Начать запись</button><button id="stop" disabled>Остановить и рассчитать</button>
<a id="report" href="/report" target="_blank" hidden>Открыть отчёт</a><p id="status">Подключение…</p><p id="error"></p>
<div class="grid" id="cameras"></div><p>Результат относится к потокам после декодирования на сервере. Он не отделяет экспозицию от задержек передачи.</p></main>
<script>
const cards={},el=id=>document.getElementById(id);let current;
async function post(url,body={}){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.detail||r.status);return d}
function failure(e){el('error').textContent=e.message}
el('start').onclick=async()=>{try{await post('/api/start',{duration:Number(el('duration').value)});el('error').textContent=''}catch(e){failure(e)}};
el('stop').onclick=async()=>{el('stop').disabled=true;try{await post('/api/stop')}catch(e){failure(e)}};
function card(info){const box=document.createElement('div');box.className='card';
 const title=document.createElement('h2');title.textContent=info.id;box.append(title);
 const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;box.append(canvas);
 const text=document.createElement('p');box.append(text);el('cameras').append(box);
 const result={canvas,text,info,image:null,drag:null};cards[info.id]=result;
 function point(e){const rect=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width,(e.clientX-rect.left)*canvas.width/rect.width)),Math.max(0,Math.min(canvas.height,(e.clientY-rect.top)*canvas.height/rect.height))]}
 canvas.onpointerdown=e=>{if(current?.recording||current?.analyzing||!result.info.size)return;canvas.setPointerCapture(e.pointerId);result.drag=point(e)};
 canvas.onpointermove=e=>{if(!result.drag)return;const ctx=canvas.getContext('2d'),end=point(e);if(result.image)ctx.drawImage(result.image,0,0);ctx.strokeStyle='#ffea6b';ctx.lineWidth=2;ctx.strokeRect(...result.drag,end[0]-result.drag[0],end[1]-result.drag[1])};
 canvas.onpointerup=async e=>{if(!result.drag)return;const start=result.drag,end=point(e);result.drag=null;
 const [w,h]=result.info.size;const x=Math.round(Math.min(start[0],end[0])*w/canvas.width),y=Math.round(Math.min(start[1],end[1])*h/canvas.height);
 const right=Math.min(w,Math.round(Math.max(start[0],end[0])*w/canvas.width)),bottom=Math.min(h,Math.round(Math.max(start[1],end[1])*h/canvas.height));
 try{await post('/api/roi/'+info.id,{roi:[x,y,right-x,bottom-y],image_size:[w,h]});el('error').textContent=''}catch(e){failure(e)}};
 canvas.onpointercancel=()=>{result.drag=null};
 async function frame(){try{const r=await fetch('/frame/'+info.id+'.jpg?t='+Date.now());if(!r.ok)throw Error('no frame');const image=await createImageBitmap(await r.blob());if(result.image)result.image.close();result.image=image;
 if(!result.drag){canvas.width=image.width;canvas.height=image.height;canvas.getContext('2d').drawImage(image,0,0)}}catch(e){if(!result.drag){const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height)}}setTimeout(frame,200)}frame();return result}
async function poll(){try{const r=await fetch('/api/state'),first=!current;current=await r.json();
 if(first)el('duration').value=current.duration_s;el('duration').disabled=current.recording||current.analyzing;
 el('status').textContent=(current.synthetic?'СИНТЕТИЧЕСКИЙ ТЕСТ · ':'')+(current.recording?`Запись: ${current.elapsed_s.toFixed(1)} / ${current.duration_s} с`:current.analyzing?'Расчёт…':'Готов к настройке')+' · Опорная камера: '+current.reference;
 el('start').disabled=current.recording||current.analyzing;el('stop').disabled=!current.recording;el('report').hidden=!current.report_available;
 if(current.error)el('error').textContent=current.error;
 for(const info of current.cameras){const c=cards[info.id]||card(info);c.info=info;c.text.textContent=info.status+' · '+(info.size?info.size.join(' × '):'—')+' · ROI: '+(info.roi?info.roi.join(', '):'выделите светодиод')+' · Яркость: '+(info.brightness?.toFixed(1)??'—')}
 }catch(e){failure(e)}setTimeout(poll,500)}poll();
</script></html>'''


def report_page(report):
    encoded = json.dumps(report,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    return '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Отчёт синхронизации LED</title><style>body{font:16px system-ui;background:#f5f7fa;color:#172637;margin:24px}main{max-width:1200px;margin:auto}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border:1px solid #cdd7e1;text-align:left}canvas{width:100%;background:white;border:1px solid #ccd5df;margin:10px 0}p{line-height:1.5}input{width:300px}pre{white-space:pre-wrap}.bad{color:#9d2e24}</style><main>
<h1>Синхронизация наблюдаемых потоков</h1><p id="meta"></p><p>Положительный сдвиг: камера видит ту же вспышку позже опорной по серверным меткам после декодирования. Экспозиция, кодирование и доставка не разделены.</p>
<table><thead><tr><th>Камера</th><th>Статус</th><th>Пар событий</th><th>Медиана, мс</th><th>Разброс p95, мс</th><th>Тренд, мс/мин</th><th>Полуширина интервала, мс</th></tr></thead><tbody id="rows"></tbody></table>
<p>Разброс p95 = 95-й процентиль |сдвиг − медиана|. Полуширина интервала перехода характеризует дискретизацию наблюдений, не является статистическим доверительным интервалом. Тренд — изменение относительной задержки, не оценка частоты часов камеры.</p>
<p id="warnings" class="bad"></p><label>Начало окна <input id="start" type="range" min="0" value="0" step="0.1"></label> <label>Длина окна, с <input id="window" type="number" min="1" value="20" style="width:80px"></label>
<h2>Яркость светодиода, 0–255</h2><canvas id="signal" width="1100" height="280"></canvas>
<h2>Разность времени одинаковых переходов, мс</h2><canvas id="offset" width="1100" height="280"></canvas><p id="legend"></p>
<p>Сырые значения и номера кадров: файлы camera.csv и camera-matches.csv рядом с report.json. Рекомендованный offset_ms, если оценка однозначна, имеет обратный знак медианы; автоматически настройки сервера не меняются.</p>
</main><script>const data='''+encoded+''';
const el=id=>document.getElementById(id),number=v=>v==null?'—':v.toFixed(2);const colors=['#126fc1','#d85417','#198344','#944bc3','#a27709','#167d82'];
el('meta').textContent=(data.synthetic?'СИНТЕТИЧЕСКИЕ ДАННЫЕ · ':'')+data.session_id+' · Опорная камера: '+data.reference;
for(const [name,r] of Object.entries(data.comparisons)){const tr=document.createElement('tr');const trend=number(r.observed_trend_ms_per_min)+(r.observed_trend_ms_per_min!=null&&!r.trend_above_sampling_floor?' (ниже разрешения метода)':'');for(const v of [name,r.status,r.matched_events,number(r.offset_median_ms),number(r.jitter_p95_ms),trend,number(r.transition_bracket_halfwidth_median_ms)]){const td=document.createElement('td');td.textContent=v;tr.append(td)}if(!r.accepted)tr.className='bad';el('rows').append(tr)}
el('warnings').textContent=data.warnings.join('; ')+' '+Object.entries(data.cameras).filter(([n,c])=>c.status!=='ok'||c.segments>1||c.gap_count).map(([n,c])=>n+': '+c.status+', сегментов '+(c.segments??0)+', пауз '+(c.gap_count??0)).join('; ');
const end=Math.max(1,...Object.values(data.traces).map(t=>t.length?t[t.length-1][0]:0));el('start').max=end;
function plot(id,series,minY,maxY){const c=el(id),ctx=c.getContext('2d'),from=+el('start').value,span=Math.max(1,+el('window').value||20);ctx.clearRect(0,0,c.width,c.height);const x=t=>55+(t-from)/span*(c.width-75),y=v=>c.height-35-(v-minY)/(maxY-minY)*(c.height-55);ctx.font='13px sans-serif';
for(let i=0;i<=4;i++){const v=minY+(maxY-minY)*i/4;ctx.strokeStyle='#e2e7ee';ctx.beginPath();ctx.moveTo(55,y(v));ctx.lineTo(c.width-20,y(v));ctx.stroke();ctx.fillStyle='#455';ctx.fillText(v.toFixed(0),4,y(v)+4);ctx.fillText((from+span*i/4).toFixed(1)+' с',x(from+span*i/4)-15,c.height-8)}
Object.entries(series).forEach(([name,points],i)=>{ctx.strokeStyle=colors[i%colors.length];ctx.beginPath();let first=true;for(const [t,v] of points){if(t<from||t>from+span){first=true;continue}if(first){ctx.moveTo(x(t),y(v));first=false}else ctx.lineTo(x(t),y(v))}ctx.stroke()})}
function draw(){plot('signal',data.traces,0,255);const pairs=Object.fromEntries(Object.entries(data.comparisons).map(([n,r])=>[n,(r.matches||[]).map(p=>[p.time_s,p.delta_ms])]));const vals=Object.values(pairs).flat().map(p=>p[1]);let low=Math.min(0,...vals),high=Math.max(0,...vals);if(high-low<10)high=low+10;plot('offset',pairs,low-5,high+5)}
el('start').oninput=el('window').oninput=draw;el('legend').textContent='Яркость: '+Object.keys(data.traces).join(' / ')+'; сдвиг: '+Object.keys(data.comparisons).join(' / ')+' (синий / оранжевый / зелёный / фиолетовый)';draw();</script></html>'''


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    serve = sub.add_parser('serve',help='Receive streams and host ROI/recording interface')
    serve.add_argument('--camera',action='append',default=[],metavar='NAME=RTSP_URL')
    serve.add_argument('--config',type=Path)
    serve.add_argument('--rois',type=Path,help='Previously saved rois.json')
    serve.add_argument('--demo',action='store_true',help='Synthetic live camera views, no physical cameras')
    serve.add_argument('--reference')
    serve.add_argument('--host',default='127.0.0.1')
    serve.add_argument('--port',type=int,default=8082)
    serve.add_argument('--output',type=Path,default=HERE/'runtime/sync')
    serve.add_argument('--duration',type=float,default=120)
    serve.add_argument('--channel',choices=['gray','red','green','blue'],default='gray')
    serve.add_argument('--save-roi-video',action='store_true',help='Annotated LED crops; adds encoding/disk load')
    serve.add_argument('--autostart',action='store_true',help='Start once all configured ROIs and streams are ready')
    serve.add_argument('--exit-after-recording',action='store_true')
    analyze_cmd = sub.add_parser('analyze',help='Re-analyze a saved session without camera access')
    analyze_cmd.add_argument('directory',type=Path)
    for cmd in (serve,analyze_cmd):
        cmd.add_argument('--max-lag-ms',type=float,default=1000 if cmd is serve else None)
        cmd.add_argument('--match-window-ms',type=float,default=80 if cmd is serve else None)
        cmd.add_argument('--min-contrast',type=float,default=20 if cmd is serve else None)
    demo = sub.add_parser('demo',help='Generate a fast, reproducible synthetic session and report')
    demo.add_argument('--output',type=Path,default=HERE/'runtime/sync')
    demo.add_argument('--duration',type=float,default=120)
    args = parser.parse_args(argv)
    try:
        if hasattr(args,'duration'):
            finite(args.duration,5,3600,'duration')
        for name,low,high in [('max_lag_ms',10,10000),('match_window_ms',1,1000),('min_contrast',1,255)]:
            if getattr(args,name,None) is not None:
                finite(getattr(args,name),low,high,name)
        if args.command == 'serve':
            if not 1<=args.port<=65535 or args.match_window_ms>=args.max_lag_ms:
                raise ValueError('Invalid port or match window >= lag search range')
            if args.demo and (args.camera or args.config or args.rois):
                raise ValueError('--demo cannot be mixed with real sources/config/ROIs')
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = arguments(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    try:
        if args.command == 'analyze':
            report = analyze(args.directory,args.max_lag_ms,args.match_window_ms,args.min_contrast)
            print(json.dumps({n:{k:v for k,v in r.items() if k!='matches'} for n,r in report['comparisons'].items()},indent=2))
        elif args.command == 'demo':
            directory,report = demo_session(args.output,args.duration)
            print('Saved:',directory/'report.html')
            print(json.dumps({n:{k:v for k,v in r.items() if k!='matches'} for n,r in report['comparisons'].items()},indent=2))
        else:
            os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','rtsp_transport;tcp')
            engine = Engine(args,read_cameras(args))
            server = uvicorn.Server(uvicorn.Config(make_app(engine),host=args.host,port=args.port,log_level='warning'))
            if args.exit_after_recording:
                def exit_when_done():
                    while not engine.stop.is_set():
                        if engine.finished.wait(.2):
                            server.should_exit = True
                            return
                threading.Thread(target=exit_when_done,daemon=True).start()
            LOG.info('LED benchmark: http://%s:%d (no pose models)',args.host,args.port)
            server.run()
            if engine.error:
                return 1
    except (OSError,ValueError,KeyError,cv2.error) as exc:
        LOG.error('%s',exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
