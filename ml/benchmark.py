"""Run the five fixed regression slices; keep candidates and history immutable.

These are historical regression cases, not an untouched test set. Outputs
contain metrics and provenance, never source photographs. First model is the
baseline. A report is complete only if all expected slices were evaluated.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import deployment_eval as D
import exam

HERE = Path(__file__).resolve().parent
EXPECTED = {'labels': {'контроль': 22, 'подборка': 1, 'студия': 17},
            'labels3': {'без лака': 13}, 'labels8': {'свои': 10}}


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


def dataset_manifest(root=HERE):
    manifest = []
    for name, expected in EXPECTED.items():
        work = root / name
        task = json.loads((work / 'task.json').read_text(encoding='utf-8'))
        counts = {}
        for t in task['items']:
            gt = work / 'instances' / f'{t["id"]}.png'
            if not gt.exists():
                continue
            photo = work / 'photos' / t['file']
            if not photo.is_file():
                raise FileNotFoundError(photo)
            counts[t['part']] = counts.get(t['part'], 0) + 1
            manifest.append(dict(work=name, id=t['id'], part=t['part'],
                                 image_sha256=sha256(photo), mask_sha256=sha256(gt),
                                 group=t.get('group')))
        if counts != expected:
            raise ValueError(f'{name}: regression membership changed: {counts} != {expected}')
    digest = hashlib.sha256(json.dumps(manifest, ensure_ascii=False,
                                      sort_keys=True).encode('utf-8')).hexdigest()
    return manifest, digest


def compare(base_rows, candidate_rows):
    base = {r['key']: r for r in base_rows}
    if set(base) != {r['key'] for r in candidate_rows}:
        raise ValueError('Cannot compare different benchmark populations')
    gained, lost, common = [], [], []
    common_iou_base, common_iou_candidate = [], []
    for r in candidate_rows:
        b = base[r['key']]
        for nail, iou in r['per_nail'].items():
            biou = b['per_nail'][nail]
            if biou is None and iou is not None:
                gained.append(f'{r["key"]}/{nail}')
            elif biou is not None and iou is None:
                lost.append(f'{r["key"]}/{nail}')
            elif biou is not None and iou is not None:
                common.append(f'{r["key"]}/{nail}')
                common_iou_base.append(biou)
                common_iou_candidate.append(iou)
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return dict(gained=gained, lost=lost, common_found=len(common),
                common_iou_baseline=mean(common_iou_base),
                common_iou_candidate=mean(common_iou_candidate))


def write_markdown(path, report):
    lines = ['# Проверка моделей примерки', '',
             f'Путь маски: `{report["pipeline"]}`; набор `{report["dataset_sha256"]}`.', '',
             'Все пять срезов — фиксированная проверка регрессий. Это не новый независимый тест.',
             'Android-путь воспроизведён программно; точное совпадение растеризации PIL/Skia на телефоне ещё не проверено.', '',
             '| Модель | Срез | Найдено | Все FP, px | FP / эталон | IoU найденных | Граница F1 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for item in report['models']:
        for part, m in item['parts'].items():
            fmt = lambda v: '—' if v is None else f'{v:.4f}'
            lines.append(f'| {item["name"]} | {part} | {m["found"]}/{m["nails"]} | '
                         f'{m["fp"]} | {fmt(m["fp_rate"])} | {fmt(m["iou"])} | '
                         f'{fmt(m["boundary_fscore"])} |')
    lines += ['', 'FP — все предсказанные пиксели вне эталона, включая соединённую с ногтем покраску кожи.',
              'Историческое stray_px — только отдельные ложные пятна; сохранено отдельно в JSON.',
              'IoU найденных зависит от состава найденных ногтей; сравнение на общих ногтях — в JSON.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pipeline', choices=['legacy', 'android'], default='android')
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    # Never overwrite a previous experiment report.
    if args.out.exists():
        raise SystemExit(f'Output already exists: {args.out}')
    manifest, dataset_sha = dataset_manifest()
    models = [Path(p).resolve(strict=True) for p in args.model]
    args.out.mkdir(parents=True)
    report = dict(schema_version=2, created_at=datetime.now(timezone.utc).isoformat(),
                  complete=False, pipeline=args.pipeline, dataset_sha256=dataset_sha,
                  manifest=manifest, models=[])
    seen = set()
    try:
        for model in models:
            digest = sha256(model)
            if digest in seen:
                print(f'Duplicate model SHA, skipped: {model.name}', flush=True)
                continue
            seen.add(digest)
            item = dict(name=f'{model.stem}-{digest[:8]}', sha256=digest,
                        source=str(model), parts={}, rows=[], metadata={})
            for work in EXPECTED:
                rows, metadata = exam.evaluate_model(model, work, args.pipeline,
                                                     threads=args.threads)
                for r in rows:
                    r['key'] = f'{work}/{r["id"]}'
                item['rows'].extend(rows)
                item['metadata'][work] = metadata
                for part in EXPECTED[work]:
                    group = [r for r in rows if r['part'] == part]
                    m = D.aggregate(group)
                    item['parts'][part] = m
                    print(f'{model.stem}: {part} {m["found"]}/{m["nails"]}, '
                          f'FP={m["fp"]}, FP/GT={m["fp_rate"]}', flush=True)
            if report['models']:
                base = report['models'][0]
                item['comparison'] = compare(base['rows'], item['rows'])
                item['regression_parts'] = [p for p in item['parts'] if
                    item['parts'][p]['found'] < base['parts'][p]['found'] or
                    item['parts'][p]['fp_rate'] > base['parts'][p]['fp_rate']]
            report['models'].append(item)
            (args.out / 'benchmark.json').write_text(json.dumps(report,
                ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        report['complete'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (args.out / 'benchmark.json').write_text(json.dumps(report,
            ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        write_markdown(args.out / 'comparison.md', report)
    print(f'Complete report: {args.out / "comparison.md"}', flush=True)


if __name__ == '__main__':
    main()
