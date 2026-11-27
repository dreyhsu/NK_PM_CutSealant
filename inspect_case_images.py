"""Rejudge saved case crops and write contour overlays/JSON beside each input.

Run from the repository root:
    python inspect_case_images.py
    python inspect_case_images.py case_crops/ng
    python inspect_case_images.py path/to/case.png --config config/config.ini
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from cutsealant.config import ConfigManager
from cutsealant.inspection import SealantInspector


IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
OVERLAY_SUFFIX = '.inspection.png'


def image_paths(inputs):
    """Snapshot inputs before writing; never reprocess our generated overlays."""
    found = set()
    for source in inputs:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f'Input does not exist: {source}')
        candidates = source.rglob('*') if source.is_dir() else [source]
        for path in candidates:
            if (path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    and not path.name.lower().endswith(OVERLAY_SUFFIX)):
                found.add(path.resolve())
    return sorted(found)


def contour_overlay(image, analysis):
    """Keep the crop intact in size and add a footer explaining the contours."""
    overlay = image.copy()
    color = {'Pass': (0, 210, 0), 'NG': (0, 0, 255)}.get(
        analysis.result, (0, 200, 255))
    if analysis.contour is not None:
        cv2.drawContours(overlay, [analysis.contour], -1, (255, 255, 0), 1)
    if analysis.approximation is not None:
        cv2.drawContours(overlay, [analysis.approximation], -1, color, 1)
    h, w = image.shape[:2]
    overlay = cv2.copyMakeBorder(overlay, 0, 72, 0, max(0, 480 - w),
                                 cv2.BORDER_CONSTANT, value=(25, 25, 25))
    cv2.putText(overlay, f'Result: {analysis.result}', (8, h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1, cv2.LINE_AA)
    cv2.putText(overlay, 'Cyan: contour | Green/red: judged polygon', (8, h + 53),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 230, 230), 1, cv2.LINE_AA)
    return overlay


def inspect_image(path, inspector, seed):
    """Do one analysis so the JSON, overlay and result all describe the same run."""
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    cv2.setRNGSeed(seed)
    analysis = inspector.analyze_crop(image)
    overlay_path = path.with_name(path.name + OVERLAY_SUFFIX)
    report_path = path.with_name(path.name + '.inspection.json')
    report = {
        'image': str(path),
        'result': analysis.result,
        'error': analysis.error,
        'kmeans_clusters': inspector.context.detection_config['kmeans_clusters'],
        'kmeans_rounds': inspector.context.detection_config['kmeans_rounds'],
        'rng_seed': seed,
        'cluster_centers': (analysis.centers.flatten().tolist()
                            if analysis.centers is not None else []),
        'contour_area': (cv2.contourArea(analysis.contour)
                         if analysis.contour is not None else None),
        'judged_polygon': (analysis.approximation.reshape(-1, 2).tolist()
                           if analysis.approximation is not None else []),
        'overlay': None,
    }
    if image is not None:
        success, encoded = cv2.imencode('.png', contour_overlay(image, analysis))
        if not success:
            raise OSError(f'Could not encode overlay for {path}')
        overlay_path.write_bytes(encoded.tobytes())
        report['overlay'] = str(overlay_path)
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='*', help='Image files or folders (searched recursively)')
    parser.add_argument('--config', type=Path, default=Path('config/config.ini'))
    parser.add_argument('--seed', type=int, default=42,
                        help='OpenCV RNG seed per image; default 42 for repeatable offline runs')
    args = parser.parse_args(argv)
    if not args.config.is_file():
        parser.error(f'Configuration does not exist: {args.config}')
    if not -(2**31) <= args.seed < 2**31:
        parser.error('--seed must fit in a signed 32-bit integer')
    try:
        config = ConfigManager(args.config)
        context = SimpleNamespace(detection_config=config.get_detection_config(), logger=None)
        if any(context.detection_config[key] <= 0 for key in ('kmeans_clusters', 'kmeans_rounds')):
            parser.error('K-means clusters and rounds must be positive')
        inspector = SealantInspector(context)
        context._img_kmeans = inspector._img_kmeans
        context._loop_closed_identify = inspector._loop_closed_identify
        paths = image_paths(args.inputs or [config.get_case_crop_config()['output_path']])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not paths:
        parser.error('No input images found')
    failed = False
    for path in paths:
        try:
            report = inspect_image(path, inspector, args.seed)
            print(f'{path}: {report["result"]}' +
                  (f' ({report["error"]})' if report['error'] else ''))
            failed |= report['result'] == 'Unavailable'
        except (OSError, cv2.error) as exc:
            print(f'{path}: ERROR ({exc})')
            failed = True
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
