#!/usr/bin/env python3
"""Measure the installed RTMlib presets on one cached image, without a GUI.

Always runs pose estimation for ONE fixed ROI, even when the image contains no
person. This is a compute benchmark, not a detection/pose accuracy evaluation.
Camera decode and source resizing are outside the measured loops. Timings
around ORT session.run include host/device copies and return of CPU outputs;
they are not isolated GPU kernel timings.
"""
import argparse
import csv
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source', required=True, help='RTSP URL or local image/video')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--presets', nargs='+', default=['balanced','lightweight','performance'],
                        choices=['balanced','lightweight','performance'])
    parser.add_argument('--sizes', nargs='+', default=['640x360','960x540','1280x720','1920x1080'])
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--ort-threads', type=int, help='Rebuild sessions with N CPU threads and spinning disabled; test only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 10 or args.warmup < 1 or args.repeat < 1:
        parser.error('Use iterations >= 10, warmup >= 1 and repeat >= 1')
    if args.ort_threads is not None and args.ort_threads < 1:
        parser.error('--ort-threads must be positive')
    try:
        args.sizes = [tuple(map(int, size.lower().split('x'))) for size in args.sizes]
        assert all(len(size)==2 and min(size)>=32 and max(size)<=4096 for size in args.sizes)
    except (ValueError, AssertionError):
        parser.error('Sizes must be WIDTHxHEIGHT within 32..4096')
    return args


def main():
    args = arguments()
    sys.path.insert(0, str(args.server_dir.resolve()))
    import visual_mocap_server as server
    import cv2
    import numpy as np

    def quantiles(values):
        return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                    p95_ms=float(np.percentile(values,95)), min_ms=float(np.min(values)))

    def gpu_state():
        result=subprocess.run(['nvidia-smi','--query-gpu=index,name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw,pstate',
                               '--format=csv'],capture_output=True,text=True)
        return result.stdout.strip()

    source_path = Path(args.source)
    image = cv2.imread(str(source_path)) if source_path.is_file() and source_path.suffix.lower() in {'.jpg','.jpeg','.png'} else None
    if image is None:
        cap=cv2.VideoCapture(args.source,cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000])
        try:
            for _ in range(15):
                ok,image=cap.read()
                if not ok: raise RuntimeError('Could not read source image')
        finally: cap.release()
    # Keep one source image only in RAM, never save webcam imagery to the report.
    images={f'{w}x{h}':cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA) for w,h in args.sizes}
    report={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'device':args.device,'python':platform.python_version(),'cpu_count':os.cpu_count(),
            'loadavg_before':os.getloadavg(),'gpu_before':gpu_state(),
            'packages':{p:importlib.metadata.version(p) for p in ('numpy','opencv-contrib-python-headless','onnxruntime-gpu')},
            'source_size':[image.shape[1],image.shape[0]],'iterations':args.iterations,
            'warmup':args.warmup,'repeats':args.repeat,'forced_single_roi':True,
            'ort_threads_no_spin':args.ort_threads,
            'scope':'detector + one pose crop; CPU preprocessing and ORT copies included; no decode/resize/preview/3D/network',
            'results':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    raw_path=args.output.with_suffix('.csv')
    with raw_path.open('w',newline='') as raw:
        writer=csv.DictWriter(raw,fieldnames=['preset','size','repeat','iteration','detector_ms','pose_ms','detector_ort_ms','pose_ort_ms','total_ms'])
        writer.writeheader()
        for preset in args.presets:
            print('Loading preset:',preset,flush=True)
            model=server.PoseModel(server.arguments(['--repos',str(args.server_dir/'repositories'),'--device',args.device,'--preset',preset]))
            if args.ort_threads is not None:
                import onnxruntime as ort
                for tool in (model.body.det_model,model.body.pose_model):
                    provider_options=tool.session.get_provider_options()
                    providers=[(name,provider_options[name]) for name in tool.session.get_providers()]
                    options=ort.SessionOptions()
                    options.intra_op_num_threads=args.ort_threads
                    options.inter_op_num_threads=1
                    options.add_session_config_entry('session.intra_op.allow_spinning','0')
                    options.add_session_config_entry('session.inter_op.allow_spinning','0')
                    tool.session=ort.InferenceSession(tool.onnx_model,sess_options=options,providers=providers)
                    tool.session.disable_fallback()
            for tool in (model.body.det_model,model.body.pose_model):
                if args.device.startswith('cuda') and tool.session.get_providers()[0] != 'CUDAExecutionProvider':
                    raise RuntimeError('CUDA must be the first provider in the benchmark session')
            measured={}
            originals=[]
            for key,tool in [('detector',model.body.det_model),('pose',model.body.pose_model)]:
                original=tool.session.run
                originals.append((tool.session,original))
                def timed(*a,_key=key,_run=original,**kw):
                    start=time.perf_counter_ns()
                    value=_run(*a,**kw)
                    measured[_key]=(time.perf_counter_ns()-start)/1e6
                    return value
                tool.session.run=timed
            stats={size:[] for size in images}
            for repeat in range(args.repeat):
                # Reverse order to reduce the effect of warming/clocks across sizes.
                order=list(images) if repeat%2==0 else list(reversed(images))
                for size in order:
                    frame=images[size]; h,w=frame.shape[:2]
                    bbox=[[.30*w,.05*h,.70*w,.95*h]]
                    for iteration in range(-args.warmup,args.iterations):
                        t0=time.perf_counter_ns()
                        model.body.det_model(frame)
                        t1=time.perf_counter_ns()
                        points,scores=model.body.pose_model(frame,bboxes=bbox)
                        t2=time.perf_counter_ns()
                        if points.shape!=(1,17,2) or not np.isfinite(points).all():
                            raise RuntimeError('Invalid pose output')
                        if iteration>=0:
                            sample=dict(preset=preset,size=size,repeat=repeat,iteration=iteration,
                                detector_ms=(t1-t0)/1e6,pose_ms=(t2-t1)/1e6,total_ms=(t2-t0)/1e6,
                                detector_ort_ms=measured['detector'],pose_ort_ms=measured['pose'])
                            stats[size].append(sample); writer.writerow(sample)
                    raw.flush()
                    print(json.dumps({'preset':preset,'size':size,'repeat':repeat,
                                      **quantiles([row['total_ms'] for row in stats[size][-args.iterations:]])}),flush=True)
            for size,samples in stats.items():
                row={'preset':preset,'size':size,'samples':len(samples),
                     'providers':{key:tool.session.get_providers() for key,tool in
                                  [('detector',model.body.det_model),('pose',model.body.pose_model)]},
                     'provider_options':{key:tool.session.get_provider_options() for key,tool in
                                         [('detector',model.body.det_model),('pose',model.body.pose_model)]},
                     'network_inputs':{'detector':model.body.det_model.session.get_inputs()[0].shape,
                                       'pose':model.body.pose_model.session.get_inputs()[0].shape}}
                for field in ['detector_ms','pose_ms','detector_ort_ms','pose_ort_ms','total_ms']:
                    row[field]=quantiles([sample[field] for sample in samples])
                row['images_per_s']=1000/row['total_ms']['mean_ms']
                # Capacity estimate only: no extra RTSP decoders or calibration.
                row['sequential_3_images_per_s_estimate']=1000/(3*row['total_ms']['mean_ms'])
                report['results'].append(row)
            for session,original in originals: session.run=original
            del originals,model,session,original,tool,timed
            gc.collect()
            report['gpu_after']=gpu_state()
            args.output.write_text(json.dumps(report,indent=2)+'\n')
    report['gpu_after']=gpu_state()
    report['loadavg_after']=os.getloadavg()
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('Saved:',args.output,'and',raw_path,flush=True)


if __name__=='__main__':
    main()
