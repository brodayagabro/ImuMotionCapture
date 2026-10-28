import argparse
from pathlib import Path

from scripts.ER2026.paths import RECORDS_ROOT, OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description='ER2026 offline processing (source recordings are read-only)')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('analyze', help='Audit indexed BVH and compare paired recordings')
    audit.add_argument('--records', type=Path, default=RECORDS_ROOT)
    audit.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'processing')
    audit.add_argument('--plots', action='store_true', help='Also render PNG/SVG figures (requires matplotlib)')
    audit.add_argument('--labels', type=Path, help='Also compare manually reviewed pose intervals in pairs')
    audit.add_argument('--reviewed-labels', action='store_true', help='Compare reviewed intervals from JSON beside each BVH')
    annotate = commands.add_parser('annotate', help='Manual interval annotation with pose playback')
    annotate.add_argument('--records', type=Path, default=RECORDS_ROOT)
    annotate.add_argument('--labels', type=Path, help='Override JSON directory; default: beside each BVH')
    train = commands.add_parser('train', help='Train and evaluate a pose baseline on reviewed intervals')
    train.add_argument('--records', type=Path, default=RECORDS_ROOT)
    train.add_argument('--labels', type=Path, help='Override JSON directory; default: beside each BVH')
    train.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'classifier')
    train.add_argument('--source', choices=['noitom', 'custom'], default='noitom')
    train.add_argument('--group', choices=['subject', 'recording'], default='subject')
    train.add_argument('--subjects', type=Path, help='JSON mapping recording subject IDs to physical participant IDs')
    predict = commands.add_parser('predict', help='Predict non-overlapping pose windows from a BVH')
    predict.add_argument('bvh', type=Path)
    predict.add_argument('--model', type=Path, default=OUTPUT_ROOT / 'classifier' / 'model.npz')
    predict.add_argument('--output', type=Path, required=True)
    export = commands.add_parser('export-labels', help='Export reviewed mocap intervals and wall-clock hints')
    export.add_argument('--records', type=Path, default=RECORDS_ROOT)
    export.add_argument('--labels', type=Path, help='Override JSON directory; default: beside each BVH')
    export.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'labels.csv')
    args = parser.parse_args()
    try:
        if args.command == 'analyze':
            from .pipeline import run
            report = run(args.records, args.output, args.labels, use_labels=args.reviewed_labels)
            if args.plots:
                from .plots import render
                render(args.output)
            print(f"{report['indexed_recordings']} recordings; {report['successful_comparisons']} pairs -> {args.output}")
        elif args.command == 'annotate':
            from .annotator import launch
            launch(args.records, args.labels)
        elif args.command == 'train':
            from .classifier import train
            train(args.records, args.labels, args.output, args.source, args.group, args.subjects)
        elif args.command == 'predict':
            from .classifier import predict_file
            predict_file(args.bvh, args.model, args.output)
        elif args.command == 'export-labels':
            from .export import export_labels
            rows = export_labels(args.records, args.labels, args.output)
            print(f'Exported {len(rows)} reviewed intervals -> {args.output}')
    except (ValueError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
