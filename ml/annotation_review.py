"""Локальная браузерная проверка/коррекция own-train масок.

Запуск из корня rustore-app:
  python ml/annotation_review.py

Берёт только train-строки source=own из манифеста завершённого эксперимента.
Validation и экзамены исключаются. Результат пишется в новый набор файлов и
никогда не перезаписывает исходные фотографии или маски.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading

import numpy as np
from PIL import Image
import io

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_MANIFEST = PROJECT / 'ML-результаты' / '2026-09-14' / 'adapt-own5-384-distill1' / 'manifest.json'
DEFAULT_OUT = PROJECT / 'ML-результаты' / '2026-09-15' / 'manual-review-own-train'
ITEMS: dict[str, dict] = {}
OUT = DEFAULT_OUT
SAVE_LOCK = threading.Lock()


def load_task(manifest_path: Path) -> list[dict]:
    doc = json.loads(manifest_path.read_text(encoding='utf-8'))
    own_root = (HERE / 'own-train').resolve()
    train = doc.get('data', {}).get('train', [])
    val = doc.get('data', {}).get('validation', [])
    val_keys = {(s.get('source'), s.get('id')) for s in val}
    selected = []
    for row in train:
        if row.get('source') != 'own':
            continue
        if (row.get('source'), row.get('id')) in val_keys:
            raise ValueError(f'Own train/validation overlap at {row.get("id")}')
        sample_id = str(row.get('id', ''))
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sample_id):
            raise ValueError(f'Unsafe sample id: {sample_id!r}')
        paths = {key: Path(row[key]).resolve(strict=True) for key in ('image', 'mask', 'instances')}
        for key, path in paths.items():
            if not path.is_relative_to(own_root):
                raise ValueError(f'{key} must be under own-train, got {path}')
        with Image.open(paths['image']) as photo, Image.open(paths['instances']) as inst:
            if photo.size != inst.size:
                raise ValueError(f'Image/instances size mismatch: {sample_id}')
        selected.append(dict(id=sample_id, group=str(row.get('group', '')), file=paths['image'].name,
                             image=str(paths['image']), mask=str(paths['mask']),
                             instances=str(paths['instances'])))
    if not selected:
        raise ValueError('Manifest contains no own training images')
    if any(('own', s['id']) in val_keys for s in selected):
        raise ValueError('Validation images cannot be part of the annotation task')
    return selected


def send_json(handler, value, status=200):
    body = json.dumps(value, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.send_header('Cache-Control', 'no-store')
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_bytes(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def sample(self, sample_id: str):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', sample_id):
            return None
        return ITEMS.get(sample_id)

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
            self.send_bytes((HERE / 'annotation_review.html').read_bytes(), 'text/html; charset=utf-8')
            return
        if path == '/api/task':
            result = []
            for s in ITEMS.values():
                saved = (OUT / 'instances' / f'{s["id"]}.png').is_file()
                result.append({k: s[k] for k in ('id', 'group', 'file') } | {'saved': saved})
            send_json(self, {'items': result, 'count': len(result), 'local_only': True})
            return
        match = re.fullmatch(r'/api/(photo|mask)/([A-Za-z0-9_-]+)', path)
        if match:
            kind, sample_id = match.groups()
            s = self.sample(sample_id)
            if not s:
                send_json(self, {'error': 'Нет такого кадра'}, 404)
                return
            if kind == 'photo':
                target = Path(s['image'])
            else:
                source = 'source=1' in self.path
                target = Path(s['instances']) if source else OUT / 'instances' / f'{sample_id}.png'
                if not target.is_file():
                    target = Path(s['instances'])
            self.send_bytes(target.read_bytes(), mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
            return
        send_json(self, {'error': 'Не найдено'}, 404)

    def do_POST(self):
        if self.path.split('?', 1)[0] != '/api/save':
            send_json(self, {'error': 'Не найдено'}, 404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length <= 0 or length > 32 * 1024 * 1024:
                raise ValueError('Некорректный размер запроса')
            request = json.loads(self.rfile.read(length))
            sample_id = str(request.get('id', ''))
            s = self.sample(sample_id)
            if not s:
                raise ValueError('Нет такого кадра')
            data = str(request.get('png', ''))
            encoded = data.split(',', 1)[-1]
            raw = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                rgba = np.asarray(im.convert('RGBA'))
            with Image.open(s['image']) as original:
                expected = original.size
            if (rgba.shape[1], rgba.shape[0]) != expected:
                raise ValueError(f'Размер маски не совпадает с фото: {(rgba.shape[1], rgba.shape[0])} != {expected}')
            labels = rgba[..., 0].astype(np.uint8)
            ids = [int(v) for v in np.unique(labels) if v]
            if len(ids) > 64:
                raise ValueError('Слишком много экземпляров для одного кадра')
            mask = (labels > 0).astype(np.uint8) * 255
            instance_path = OUT / 'instances' / f'{sample_id}.png'
            mask_path = OUT / 'masks' / f'{sample_id}.png'
            meta_path = OUT / 'meta' / f'{sample_id}.json'
            with SAVE_LOCK:
                for folder in (instance_path.parent, mask_path.parent, meta_path.parent):
                    folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(labels).save(instance_path.with_suffix('.tmp.png'))
                Image.fromarray(mask).save(mask_path.with_suffix('.tmp.png'))
                instance_path.with_suffix('.tmp.png').replace(instance_path)
                mask_path.with_suffix('.tmp.png').replace(mask_path)
                metadata = {'id': sample_id, 'file': s['file'], 'group': s['group'],
                            'nails': len(ids), 'areas': {str(i): int((labels == i).sum()) for i in ids},
                            'reviewed_at': datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
                            'source': 'own-train training split', 'manual_review': True}
                meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
            send_json(self, {'ok': True, 'nails': len(ids), 'id': sample_id})
        except (ValueError, KeyError, json.JSONDecodeError, binascii.Error, OSError) as exc:
            send_json(self, {'error': str(exc)}, 400)
        except Exception as exc:
            send_json(self, {'error': f'{type(exc).__name__}: {exc}'}, 500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    global ITEMS, OUT
    if not args.manifest.is_file():
        parser.error(f'Manifest not found: {args.manifest}')
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    task = load_task(args.manifest.resolve())
    ITEMS = {s['id']: s for s in task}
    (OUT / 'task-manifest.json').write_text(json.dumps({
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source_manifest': str(args.manifest.resolve()),
        'split': 'training only; source=own; excludes validation and all exams',
        'items': [{k: s[k] for k in ('id', 'group', 'file')} for s in task],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Локальная разметка: http://127.0.0.1:{args.port}', flush=True)
    print(f'Кадров: {len(task)}; результат отдельно: {OUT}', flush=True)
    print('Сервер доступен только на этом компьютере. Для остановки нажмите Ctrl+C.', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
