"""Bounded, local adaptation with a frozen encoder and old-model replay.

The legacy training scripts and release assets are never modified. All data,
draws, checkpoints and validation reports belong to a NEW --out-dir. Source
shares mean shares of ALL image presentations, not a share of the gold subset.
No release exam is used to select a checkpoint. Own validation is held out by
shooting group from own-train, while dataset_gold/split.json val stays fixed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def image_hash(path):
    # Decoded RGB catches identical pictures copied under other names/formats.
    with Image.open(path) as im:
        im = im.convert('RGB')
        return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


@dataclass(frozen=True)
class Sample:
    id: str
    source: str
    group: str
    image: str
    mask: str
    instances: str | None = None
    part: str = ''


def own_group_split(samples, fraction, seed):
    groups = sorted({s.group for s in samples})
    if not 0 < fraction < 1 or len(groups) < 2:
        raise ValueError('Own validation needs >=2 shooting groups and 0<val-frac<1')
    ordered = sorted(groups, key=lambda g: hashlib.sha256(
        f'{seed}:{g}'.encode('utf-8')).hexdigest())
    count = min(len(groups) - 1, max(1, round(len(groups) * fraction)))
    held = set(ordered[:count])
    return ([s for s in samples if s.group not in held],
            [s for s in samples if s.group in held])


def check_paths(sample):
    for p in (sample.image, sample.mask, sample.instances):
        if p is not None and not Path(p).is_file():
            raise ValueError(f'Missing {sample.source}/{sample.id}: {p}')
    with Image.open(sample.image) as im, Image.open(sample.mask) as mask:
        if im.size != mask.size:
            raise ValueError(f'Image/mask size mismatch: {sample.source}/{sample.id}')
        if sample.instances:
            with Image.open(sample.instances) as inst:
                if im.size != inst.size:
                    raise ValueError(f'Instance size mismatch: {sample.source}/{sample.id}')


def audit_samples(train_sources, val_domains):
    """Reject exact image leakage; record membership and label/file identities."""
    train = [s for values in train_sources.values() for s in values]
    val = [s for values in val_domains.values() for s in values]
    # A gold subset may be repeated in aggregate reports, never in this input.
    def record(s):
        check_paths(s)
        d = asdict(s)
        d.update(image_sha256=sha256_file(s.image), rgb_sha256=image_hash(s.image),
                 mask_sha256=sha256_file(s.mask))
        if s.instances:
            d['instances_sha256'] = sha256_file(s.instances)
        return d
    train_records, val_records = [record(s) for s in train], [record(s) for s in val]
    val_hashes = {s['rgb_sha256']: s for s in val_records}
    for s in train_records:
        if s['rgb_sha256'] in val_hashes:
            other = val_hashes[s['rgb_sha256']]
            raise ValueError(f'Exact train/validation image leakage: '
                             f'{s["source"]}/{s["id"]} == '
                             f'{other["source"]}/{other["id"]}')
    train_groups = {s.group for s in train if s.source == 'own'}
    val_groups = {s.group for s in val if s.source == 'own'}
    if train_groups & val_groups:
        raise ValueError('Own shooting groups overlap train/validation')
    return {'train': train_records, 'validation': val_records,
            'own_train_groups': sorted(train_groups),
            'own_validation_groups': sorted(val_groups),
            'leakage_check': 'decoded RGB exact hash across all train/val sources; '
                             'own shooting-group separation; near-duplicates not checked'}


def load_sources(args):
    auto_dir, gold_dir, own_dir = map(Path, (args.auto_dir, args.gold_dir, args.own_dir))
    review_dir = Path(args.own_review_dir).resolve() if args.own_review_dir else None
    # Explicitly reject accidentally passing the named release exams as a source.
    forbidden = {'labels', 'labels3', 'labels8'}
    for path in (auto_dir, gold_dir, own_dir):
        if path.resolve().name.casefold() in forbidden:
            raise ValueError(f'Release exam cannot be a training source: {path}')
    split = read_json(gold_dir / 'split.json')
    control = split.get('by_part', {}).get('контроль')
    if control is None:
        raise ValueError('Gold split must identify контроль so it can be excluded')
    val_ids = set(split['val'])
    if val_ids & set(split['train']):
        raise ValueError('Gold split train/val IDs overlap')
    excluded = set(control['train']) | set(control['val'])
    excluded |= set(split.get('own', [])) | set(split.get('cc0', []))
    parts = {i: p for p, grp in split['by_part'].items()
             for key in ('train', 'val') for i in grp[key]}

    def gold(i):
        return Sample(i, 'gold', f'gold:{i}', str(gold_dir / 'images' / f'{i}.jpg'),
                      str(gold_dir / 'masks' / f'{i}.png'),
                      str(gold_dir / 'instances' / f'{i}.png'), parts.get(i, 'round2'))

    gold_train = [gold(i) for i in split['train'] if i not in excluded]
    if any(s.id.startswith(('own-', 'cc0-')) or s.part == 'подборка' for s in gold_train):
        raise ValueError('Gold has unexpected own/CC0/collage rows; use clean base split')
    groups_path = auto_dir / 'groups.json'
    groups = read_json(groups_path) if groups_path.exists() else {}
    auto = [Sample(p.stem, 'auto', str(groups.get(p.name, p.stem)), str(p),
                   str(auto_dir / 'masks' / f'{p.stem}.png'))
            for p in sorted((auto_dir / 'images').iterdir())
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    own = []
    for row in read_json(own_dir / 'task.json')['items']:
        if not row.get('group') or row.get('part') != 'свои':
            raise ValueError('Each own image must have a shooting group and part=свои')
        iid = row['id']
        source_mask = own_dir / 'masks' / f'{iid}.png'
        source_instances = own_dir / 'instances' / f'{iid}.png'
        if review_dir:
            reviewed_mask = review_dir / 'masks' / f'{iid}.png'
            reviewed_instances = review_dir / 'instances' / f'{iid}.png'
            reviewed_meta = review_dir / 'meta' / f'{iid}.json'
            exists = (reviewed_mask.is_file(), reviewed_instances.is_file(), reviewed_meta.is_file())
            if any(exists) and not all(exists):
                raise ValueError(f'Incomplete reviewed annotation for own/{iid}')
            if all(exists):
                source_mask, source_instances = reviewed_mask, reviewed_instances
        own.append(Sample(row['id'], 'own', row['group'],
                          str(own_dir / 'photos' / row['file']),
                          str(source_mask), str(source_instances), 'свои'))
    if len({s.id for s in own}) != len(own):
        raise ValueError('Duplicate own IDs')
    own_train, own_val = own_group_split(own, args.own_val_frac, args.seed)
    if review_dir:
        task_manifest = review_dir / 'task-manifest.json'
        if task_manifest.is_file():
            reviewed_ids = {str(x['id']) for x in read_json(task_manifest).get('items', [])}
            train_ids = {s.id for s in own_train}
            if not reviewed_ids <= train_ids:
                raise ValueError('Review task contains own validation or unknown IDs')
            expected = set(reviewed_ids)
            actual = {s.id for s in own_train if Path(s.instances).parent == review_dir / 'instances'}
            if actual != expected:
                raise ValueError(f'Review annotations incomplete: expected {len(expected)}, found {len(actual)}')
    sources = {'auto': auto, 'gold': gold_train, 'own': own_train}
    if any(not s for s in sources.values()):
        raise ValueError('All three training sources must be nonempty')
    domains = {'own': own_val}
    for iid in split['val']:
        s = gold(iid)
        domains.setdefault('gold/' + s.part, []).append(s)
    return sources, domains


def make_draws(sources, total, own_share, seed):
    """Fixed exact whole-run source counts, shared between all ablation arms."""
    if not 0 <= own_share < 1 or total < 1:
        raise ValueError('Need 0<=own-share<1 and at least one draw')
    n_own = round(total * own_share)
    n_gold = (total - n_own) // 2
    names = (['own'] * n_own + ['gold'] * n_gold
             + ['auto'] * (total - n_own - n_gold))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    return [{'source': name, 'index': int(rng.integers(len(sources[name]))),
             'augmentation_seed': int(rng.integers(2 ** 31))} for name in names]


def training_pair(sample, size, seed):
    import train as T
    rng = np.random.default_rng(seed)
    with Image.open(sample.image) as f:
        im = f.convert('RGB')
    with Image.open(sample.mask) as f:
        mask = f.convert('L')
    rot = int(rng.integers(4))
    if rot:
        op = (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
              Image.Transpose.ROTATE_270)[rot - 1]
        im, mask = im.transpose(op), mask.transpose(op)
    if rng.random() < 0.5:
        im, mask = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(.9, 1.1)))
    im, mask = T.letterbox_pair(im, mask, size)
    x = torch.from_numpy(np.asarray(im, np.float32).copy() / 255).permute(2, 0, 1)
    y = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None]
    return x, y


def freeze_encoder(net):
    for p in net.enc.parameters():
        p.requires_grad_(False)
    return adaptation_train_mode(net)


def adaptation_train_mode(net):
    net.train()
    net.enc.eval()
    # Small batches would otherwise change running statistics despite a frozen encoder.
    for module in net.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)
    return net


def replay_distillation(student_logits, teacher_logits, target, replay, confidence=.9):
    """Retain confident old predictions only when they agree with supervision.

    Old mistakes and ALL own samples are excluded, so teacher predictions cannot
    override new human labels. Foreground/background means receive equal weight.
    """
    p = teacher_logits.detach().sigmoid()
    replay = replay.to(device=p.device, dtype=torch.bool).reshape(-1, 1, 1, 1)
    positive = (target > .5) & (p >= confidence) & replay
    negative = (target <= .5) & (p <= 1 - confidence) & replay
    losses = F.binary_cross_entropy_with_logits(student_logits, p, reduction='none')
    terms = [losses[m].mean() for m in (positive, negative) if bool(m.any())]
    return torch.stack(terms).mean() if terms else student_logits.sum() * 0


def validate(net, domains, size):
    import deployment_eval as E
    net.eval()
    result = {}
    with torch.inference_mode():
        for name, samples in domains.items():
            rows = []
            for s in samples:
                with Image.open(s.image) as f:
                    im = f.convert('RGB')
                square = E.prepare_input(im, size)
                rgb = np.asarray(square, dtype=np.uint8)
                x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None]
                prob = net(x).sigmoid()[0, 0].numpy()
                pred = E.postprocess(prob, rgb, original_size=im.size)
                with Image.open(s.instances) as f:
                    gt = np.asarray(f).copy()
                row = E.score_frame(gt, pred)
                row.update(id=s.id, source=s.source, part=s.part, group=s.group)
                rows.append(row)
            result[name] = {'aggregate': E.aggregate(rows), 'frames': rows}
    return result


def selection_key(report):
    values = [v['aggregate'] for v in report.values()]
    return (sum(float(v['recall']) for v in values) / len(values),
            -sum(float(v['fp_rate']) for v in values) / len(values),
            sum(float(v.get('iou') or 0) for v in values) / len(values))


def passes_gate(candidate, baseline):
    if set(candidate) != set(baseline):
        return False
    for name in baseline:
        c, b = candidate[name]['aggregate'], baseline[name]['aggregate']
        if (c['recall'] + 1e-12 < b['recall']
                or c['fp_rate'] > b['fp_rate'] + 1e-12):
            return False
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--init', required=True, type=Path)
    ap.add_argument('--out-dir', required=True, type=Path)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--steps-per-epoch', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--own-share', type=float, default=.1)
    ap.add_argument('--distill-weight', type=float, default=1.)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--size', type=int, default=576)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--own-val-frac', type=float, default=.25)
    ap.add_argument('--auto-dir', type=Path, default=HERE / 'dataset_merged')
    ap.add_argument('--gold-dir', type=Path, default=HERE / 'dataset_gold')
    ap.add_argument('--own-dir', type=Path, default=HERE / 'own-train')
    ap.add_argument('--own-review-dir', type=Path, default=None,
                    help='directory saved by annotation_review.py; only train-fold own masks are used')
    ap.add_argument('--dry-run', action='store_true', help='audit and write immutable manifest only')
    args = ap.parse_args(argv)
    for field in ('epochs', 'steps_per_epoch', 'batch_size', 'threads'):
        if getattr(args, field) < 1:
            ap.error(f'{field} must be positive')
    if args.size < 32 or args.size % 32 or args.lr <= 0 or args.distill_weight < 0:
        ap.error('size must be a multiple of32, lr>0, distill-weight>=0')
    if not args.init.is_file():
        ap.error('initial checkpoint does not exist')
    if args.out_dir.exists():
        ap.error('--out-dir must be a NEW directory; existing results are immutable')
    sources, domains = load_sources(args)
    audit = audit_samples(sources, domains)
    draws = make_draws(sources, args.epochs * args.steps_per_epoch * args.batch_size,
                       args.own_share, args.seed)
    trace = [dict(d, id=sources[d['source']][d['index']].id) for d in draws]
    init_hash = sha256_file(args.init)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), config=config,
                    init_sha256=init_hash, data=audit, draw_counts=dict(Counter(d['source'] for d in draws)),
                    augmentation='rot90 + horizontal flip + brightness uniform[0.9,1.1]; seeded per draw',
                    selection='per-domain recall >= baseline and fp_rate <= baseline; '
                              'rank macro recall, then lower macro fp_rate, then macro IoU',
                    release_tests_used=False,
                    draw_trace_sha256=hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest())
    args.out_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.out_dir / 'manifest.json', manifest)
    write_json(args.out_dir / 'draws.json', trace)
    print(f'Audit complete: {manifest["draw_counts"]}; val domains '
          f'{ {k:len(v) for k,v in domains.items()} }', flush=True)
    if args.dry_run:
        return 0
    import train as T
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    T.SIZE = args.size
    net = T.NailNet(pretrained=False)
    state = torch.load(args.init, map_location='cpu', weights_only=True)
    if sha256_file(args.init) != init_hash:
        raise ValueError('Initial checkpoint changed during preparation')
    net.load_state_dict(state, strict=True)
    teacher = T.NailNet(pretrained=False)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    freeze_encoder(net)
    torch.save(state, args.out_dir / 'baseline.pt')
    baseline = validate(net, domains, args.size)
    write_json(args.out_dir / 'epoch-000.json', baseline)
    history = [{'epoch': 0, 'gate_passed': True, 'selected': True, 'metrics': baseline}]
    best_key, selected_epoch = selection_key(baseline), 0
    write_json(args.out_dir / 'selection.json', {'epoch': 0, 'checkpoint': 'baseline.pt',
               'improved': False, 'init_sha256': init_hash})
    criterion = T.BCEDiceLoss()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                           lr=args.lr, weight_decay=1e-4)
    offset = 0
    for epoch in range(1, args.epochs + 1):
        adaptation_train_mode(net)
        supervised_total = distilled_total = 0.
        for step in range(args.steps_per_epoch):
            batch = draws[offset:offset + args.batch_size]
            offset += args.batch_size
            pairs = [training_pair(sources[d['source']][d['index']], args.size,
                                   d['augmentation_seed']) for d in batch]
            x, y = torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])
            replay = torch.tensor([d['source'] != 'own' for d in batch])
            opt.zero_grad(set_to_none=True)
            logits = net(x)
            supervised = criterion(logits, y)
            if args.distill_weight:
                with torch.no_grad():
                    reference = teacher(x)
                distilled = replay_distillation(logits, reference, y, replay)
            else:
                distilled = logits.sum() * 0
            loss = supervised + args.distill_weight * distilled
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite loss at epoch{epoch} step{step}')
            loss.backward()
            opt.step()
            supervised_total += float(supervised.detach())
            distilled_total += float(distilled.detach())
        report = validate(net, domains, args.size)
        checkpoint = f'epoch-{epoch:03}.pt'
        torch.save(net.state_dict(), args.out_dir / checkpoint)
        write_json(args.out_dir / f'epoch-{epoch:03}.json', report)
        gate = passes_gate(report, baseline)
        selected = gate and selection_key(report) > best_key
        if selected:
            best_key, selected_epoch = selection_key(report), epoch
            write_json(args.out_dir / 'selection.json', {'epoch': epoch, 'checkpoint': checkpoint,
                       'improved': True, 'init_sha256': init_hash})
        history.append({'epoch': epoch, 'gate_passed': gate, 'selected': selected,
                        'supervised_loss': supervised_total / args.steps_per_epoch,
                        'distill_loss': distilled_total / args.steps_per_epoch, 'metrics': report})
        write_json(args.out_dir / 'history.json', history)
        concise = {k: {'found': v['aggregate']['found'], 'nails': v['aggregate']['nails'],
                       'fp_rate': v['aggregate']['fp_rate']} for k, v in report.items()}
        print(f'Epoch {epoch}/{args.epochs} gate={gate} selected={selected}: '
              f'{json.dumps(concise, ensure_ascii=False)}', flush=True)
    write_json(args.out_dir / 'completed.json', {'epochs': args.epochs, 'optimizer_steps':
               args.epochs * args.steps_per_epoch, 'selected_epoch': selected_epoch,
               'improved': selected_epoch != 0, 'release_decision': 'not evaluated/not deployed'})
    print(f'Complete; selected epoch={selected_epoch}. Release exams still required.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
