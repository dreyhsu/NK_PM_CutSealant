import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from cutsealant.config import ConfigManager
from inspect_case_images import image_paths


def test_crop_config_defaults_off_and_supports_old_ini(tmp_path):
    path = tmp_path / 'config.ini'
    config = ConfigManager(path)
    assert config.get_case_crop_config() == {'enabled': False, 'output_path': './case_crops/'}
    del config.config['CASE_CROP_SAVING']
    with path.open('w') as stream:
        config.config.write(stream)
    assert ConfigManager(path).get_case_crop_config()['enabled'] is False


@pytest.mark.parametrize('result,threshold,folder', [('Pass', 10, 'ok'), ('NG', 5, 'ng')])
def test_saves_exact_threshold_crop_once_and_preserves_manual_evidence(
        rig, tmp_path, result, threshold, folder):
    rig.p.case_crop_config = {'enabled': True, 'output_path': str(tmp_path / '裁切')}
    rig.judge.return_value = result
    image = np.full((200, 200, 3), 35, np.uint8)
    for _ in range(threshold - 1):
        rig.feed((3,), image=image)
    assert not (tmp_path / '裁切').exists()
    image[160:180, 120:150] = [10, 70, 230]
    workflow = rig.feed((3,), image=image)
    paths = list((tmp_path / '裁切' / folder).glob('*.png'))
    assert len(paths) == 1
    saved = cv2.imdecode(np.frombuffer(paths[0].read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(saved, image[160:180, 120:150])
    assert '_carrier_000001_position_04_frame_' in paths[0].name
    assert f'{threshold:09d}.png' in paths[0].name
    for _ in range(12):
        rig.feed((3,), image=np.full_like(image, 255))
    if result == 'NG':
        rig.p._handle_key(ord('M'), *rig.state)
        rig.p.manual_review_window.sealant_states[0]['user_agrees'] = False
        rig.p._handle_key(13, *rig.state)
        assert workflow.positions[3].passed
    assert list((tmp_path / '裁切').rglob('*.png')) == paths
    assert np.array_equal(saved, workflow.positions[3].crop)


def test_disabled_crop_saving_writes_nothing(rig, tmp_path):
    rig.p.case_crop_config = {'enabled': False, 'output_path': str(tmp_path / 'crops')}
    for _ in range(10):
        rig.feed((0,))
    rig.judge.return_value = 'NG'
    for _ in range(5):
        workflow = rig.feed((1,))
    assert workflow.positions[0].passed and workflow.positions[1].needs_review
    assert not (tmp_path / 'crops').exists()


def test_unavailable_observations_never_save_quality_crops(rig, tmp_path):
    rig.p.case_crop_config = {'enabled': True, 'output_path': str(tmp_path / 'crops')}
    rig.judge.return_value = 'Unavailable'
    for _ in range(12):
        rig.feed((0,))
    assert not (tmp_path / 'crops').exists()


def test_new_carrier_saves_another_crop_without_overwriting(rig, tmp_path):
    root = tmp_path / 'crops'
    rig.p.case_crop_config = {'enabled': True, 'output_path': str(root)}
    for _ in range(10):
        rig.feed((0,))
    first = next((root / 'ok').glob('*.png'))
    first_bytes = first.read_bytes()
    rig.absent(5, 2)
    for _ in range(26):
        rig.feed((0,))
    paths = list((root / 'ok').glob('*.png'))
    assert len(paths) == 2
    assert any('_carrier_000002_' in p.name for p in paths)
    assert first.read_bytes() == first_bytes


def test_disk_failure_does_not_block_decisions(rig, tmp_path):
    blocked = tmp_path / 'file_not_directory'
    blocked.write_text('occupied')
    rig.p.case_crop_config = {'enabled': True, 'output_path': str(blocked)}
    for _ in range(10):
        workflow = rig.feed((0,))
    assert workflow.ready
    assert rig.p.logger.log_error.call_args.args[0] == 'case_crop_save_error'


def synthetic_case(concave=False):
    image = np.zeros((100, 100, 3), np.uint8)
    image[2:7, 2:7] = 100
    points = [[10, 10], [80, 10], [80, 80], [10, 80]]
    if concave:
        points.insert(2, [45, 45])
    cv2.fillPoly(image, [np.array(points, np.int32)], (255, 255, 255))
    return image


def test_offline_cli_writes_matching_contours_and_json_and_skips_outputs(tmp_path, processor):
    config = ConfigManager()
    root = tmp_path / '裁切'
    config.config['CASE_CROP_SAVING'] = {'enabled': 'false', 'output_path': str(root)}
    with open(config.config_file, 'w') as stream:
        config.config.write(stream)
    originals = []
    for folder, concave in [('ok', False), ('ng', True)]:
        directory = root / folder
        directory.mkdir(parents=True)
        path = directory / 'case.png'
        image = synthetic_case(concave)
        path.write_bytes(cv2.imencode('.png', image)[1].tobytes())
        originals.append((path, path.read_bytes(), image))
    script = Path(__file__).resolve().parents[1] / 'inspect_case_images.py'
    reports = []
    for run in range(2):
        completed = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
                                   capture_output=True, text=True, timeout=20)
        assert completed.returncode == 0, completed.stderr
        assert len(completed.stdout.splitlines()) == 2
        for path, original, image in originals:
            report = json.loads(path.with_name(path.name + '.inspection.json').read_text())
            assert report['result'] == ('Pass' if path.parent.name == 'ok' else 'NG')
            cv2.setRNGSeed(42)
            analysis = processor.inspection.analyze_crop(image)
            assert report['result'] == analysis.result
            assert report['judged_polygon'] == analysis.approximation.reshape(-1, 2).tolist()
            overlay = cv2.imdecode(np.frombuffer(Path(report['overlay']).read_bytes(), np.uint8),
                                   cv2.IMREAD_COLOR)
            x, y = analysis.approximation[0, 0]
            assert overlay[y, x].tolist() == ([0, 210, 0] if analysis.result == 'Pass' else [0, 0, 255])
            assert path.read_bytes() == original
            if run == 0:
                reports.append(report)
            else:
                assert report in reports
    assert image_paths([root, originals[0][0]]) == sorted(p.resolve() for p, _, _ in originals)


def test_offline_cli_continues_after_unreadable_image(tmp_path):
    ConfigManager()
    path = tmp_path / 'broken.png'
    path.write_bytes(b'')
    good = tmp_path / 'good.png'
    good.write_bytes(cv2.imencode('.png', synthetic_case())[1].tobytes())
    script = Path(__file__).resolve().parents[1] / 'inspect_case_images.py'
    completed = subprocess.run([sys.executable, str(script), str(tmp_path)], cwd=tmp_path,
                               capture_output=True, text=True, timeout=20)
    assert completed.returncode == 1
    report = json.loads(path.with_name(path.name + '.inspection.json').read_text())
    assert report['result'] == 'Unavailable' and report['error']
    assert report['overlay'] is None
    assert good.with_name(good.name + '.inspection.png').is_file()
