"""Инструмент ручной разметки ногтей с подсказкой SAM.

Запуск:
    python label_tool.py
    → откроется http://127.0.0.1:8765

Как это работает. Тяжёлая часть SAM — кодировщик картинки (несколько секунд
на процессоре), и он не зависит от кликов. Поэтому кодировщик прогоняется
один раз на фото, результат кладётся на диск, а на каждый клик работает
только декодер маски — это десятки миллисекунд, то есть контур появляется
сразу. Пока человек размечает текущее фото, соседний поток считает
кодировщик для следующего, и ожидания не видно вовсе.

Разметка ведётся и сохраняется в рабочем размере (labels/photos, длинная
сторона 1024). Один и тот же пиксель у человека, у SAM и в файле маски —
никаких пересчётов координат и потерь на округлении.

Что пишется на диск после «Сохранить»:
    labels/masks/<id>.png       0/255 — обычная бинарная маска для обучения
    labels/instances/<id>.png   0..N — каждый ноготь своим номером
    labels/meta/<id>.json       сколько ногтей, площади, когда, сколько кликов

Кодировщик и веса: SAM (Apache-2.0). MobileSAM берётся первым, если лежит
рядом, иначе полноразмерный vit_b — он медленнее, но контуры точнее, а для
эталона это важнее скорости.
"""
import base64
import io
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

# По умолчанию torch берёт половину ядер. Кодировщик SAM здесь — самая долгая
# операция во всём инструменте, и лишние потоки экономят на ней минуты.
torch.set_num_threads(os.cpu_count() or 2)

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'labels')
PHOTOS = os.path.join(WORK, 'photos')
MASKS = os.path.join(WORK, 'masks')
INSTANCES = os.path.join(WORK, 'instances')
META = os.path.join(WORK, 'meta')
CACHE = os.path.join(WORK, 'cache')

PORT = 8765
# Порядок важен: mobile_sam (vit_t) быстрее в разы, но контуры у него
# заметно грубее. Для эталонной разметки точность дороже — поэтому если
# рядом лежит полноразмерный vit_b, берём его.
CHECKPOINTS = [
    ('vit_b', os.path.join(HERE, 'sam_vit_b.pth')),
    ('vit_b', os.path.join(os.environ.get('TEMP', '/tmp'), 'mmds', 'sam_vit_b.pth')),
    ('vit_t', os.path.join(HERE, 'mobile_sam.pt')),
]

# Замок рекурсивный: префетч держит его на всё время «посчитать следующее
# фото и вернуть предиктор на текущее», а внутри ещё раз берёт его prime().
# Без этого клик мог попасть в промежуток, когда в предикторе лежит уже
# СЛЕДУЮЩИЙ снимок, а _primed говорит, что текущий, — и маска считалась бы
# по чужой фотографии.
_lock = threading.RLock()
_predictor = None
_sam_kind = None


def load_sam():
    global _predictor, _sam_kind
    for kind, path in CHECKPOINTS:
        if not os.path.exists(path):
            continue
        if kind == 'vit_t':
            from mobile_sam import SamPredictor, sam_model_registry
        else:
            from segment_anything import SamPredictor, sam_model_registry
        print(f'SAM: {kind} из {path}', flush=True)
        sam = sam_model_registry[kind](checkpoint=path)
        sam.eval()
        _predictor, _sam_kind = SamPredictor(sam), kind
        return
    raise SystemExit(
        'Не найдены веса SAM. Положите рядом sam_vit_b.pth или mobile_sam.pt:\n'
        '  ' + '\n  '.join(p for _, p in CHECKPOINTS))


def task_items():
    with open(os.path.join(WORK, 'task.json'), encoding='utf-8') as fh:
        return json.load(fh)['items']


def photo_rgb(item):
    return np.asarray(Image.open(os.path.join(PHOTOS, item['file'])).convert('RGB'))


def prime(item):
    """Посчитать (или поднять из кэша) эмбеддинг фото и зарядить им предиктор."""
    cpath = os.path.join(CACHE, f'{item["id"]}.npz')
    with _lock:
        if os.path.exists(cpath):
            z = np.load(cpath)
            feat, orig, inp = z['feat'], tuple(z['orig']), tuple(z['inp'])
            _predictor.reset_image()
            _predictor.features = torch.from_numpy(feat.astype(np.float32))
            _predictor.original_size = orig
            _predictor.input_size = inp
            _predictor.is_image_set = True
            return 'кэш'
        rgb = photo_rgb(item)
        t0 = time.time()
        _predictor.set_image(rgb)
        os.makedirs(CACHE, exist_ok=True)
        # float16 вдвое легче на диске, а на качество маски не влияет:
        # декодер всё равно работает с этим тензором как с приближением.
        np.savez(cpath,
                 feat=_predictor.features.numpy().astype(np.float16),
                 orig=np.array(_predictor.original_size),
                 inp=np.array(_predictor.input_size))
        return f'{time.time() - t0:.1f} с'


_primed = None


def ensure_primed(item):
    global _primed
    if _primed != item['id']:
        how = prime(item)
        _primed = item['id']
        return how
    return 'уже'


def prefetch(item):
    """Досчитать эмбеддинг следующего фото, пока человек размечает текущее.

    Предиктор один на процесс, поэтому поток берёт тот же замок; заряженным
    остаётся последнее фото, за которым сходили. Чтобы работа человека не
    сбилась, после досчёта возвращаем предиктор на текущее фото.
    """
    cpath = os.path.join(CACHE, f'{item["id"]}.npz')
    if os.path.exists(cpath):
        return
    try:
        with _lock:
            cur = _primed
            prime(item)
            items = {t['id']: t for t in task_items()}
            if cur and cur in items:
                prime(items[cur])
    except Exception as e:  # фоновая работа не должна ронять сервер
        print('префетч не удался:', e, flush=True)


def png_rgba(mask, rgb=(0, 224, 255), alpha=120):
    """Маску — в RGBA-PNG: цвет в RGB, сама маска в альфе.

    Так браузеру не нужно разбирать пиксели, чтобы показать предложение:
    достаточно drawImage поверх фото.
    """
    h, w = mask.shape
    out = np.zeros((h, w, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = rgb
    out[..., 3] = np.where(mask, alpha, 0)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, 'PNG', optimize=False)
    return buf.getvalue()


def predict(item, points, box, variant):
    ensure_primed(item)
    pc = np.array([[p[0], p[1]] for p in points], np.float32) if points else None
    pl = np.array([p[2] for p in points], np.int32) if points else None
    bx = np.array(box, np.float32) if box else None
    with _lock:
        masks, scores, _ = _predictor.predict(
            point_coords=pc, point_labels=pl, box=bx, multimask_output=True)
    order = np.argsort(-scores)  # варианты по убыванию уверенности
    i = order[variant % len(order)]
    return masks[i].astype(bool), float(scores[i]), len(order)


def save_labels(item, idx_png_b64, meta):
    raw = base64.b64decode(idx_png_b64.split(',', 1)[-1])
    a = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))
    idx = a[..., 0]  # номер ногтя лежит в красном канале
    for d in (MASKS, INSTANCES, META):
        os.makedirs(d, exist_ok=True)
    Image.fromarray(np.where(idx > 0, 255, 0).astype(np.uint8)).save(
        os.path.join(MASKS, f'{item["id"]}.png'))
    Image.fromarray(idx).save(os.path.join(INSTANCES, f'{item["id"]}.png'))
    ids = [int(v) for v in np.unique(idx) if v]
    meta = dict(meta or {})
    meta.update({
        'id': item['id'], 'file': item['file'], 'part': item['part'],
        'nails': len(ids),
        'areas': {str(v): int((idx == v).sum()) for v in ids},
        'saved_at': datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds'),
        'sam': _sam_kind,
    })
    with open(os.path.join(META, f'{item["id"]}.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return len(ids)


def status_of(item):
    p = os.path.join(META, f'{item["id"]}.json')
    if not os.path.exists(p):
        return {'done': False, 'nails': None}
    with open(p, encoding='utf-8') as fh:
        m = json.load(fh)
    return {'done': True, 'nails': m.get('nails', 0), 'skipped': m.get('skipped', False)}


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype, cache=False):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'max-age=3600' if cache else 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), 'application/json')

    def _item(self, iid):
        for t in task_items():
            if t['id'] == iid:
                return t
        return None

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'label_tool.html'), 'rb') as fh:
                return self._send(200, fh.read(), 'text/html; charset=utf-8')
        if path == '/api/task':
            items = task_items()
            return self._json({'sam': _sam_kind,
                               'items': [dict(t, **status_of(t)) for t in items]})
        m = re.fullmatch(r'/api/photo/(\d+)', path)
        if m:
            it = self._item(m.group(1))
            if not it:
                return self._json({'error': 'нет такого фото'}, 404)
            with open(os.path.join(PHOTOS, it['file']), 'rb') as fh:
                return self._send(200, fh.read(), 'image/jpeg', cache=True)
        m = re.fullmatch(r'/api/instances/(\d+)', path)
        if m:
            p = os.path.join(INSTANCES, f'{m.group(1)}.png')
            if not os.path.exists(p):
                return self._json({'error': 'нет разметки'}, 404)
            with open(p, 'rb') as fh:
                return self._send(200, fh.read(), 'image/png')
        self._json({'error': 'не найдено'}, 404)

    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0))
        try:
            req = json.loads(self.rfile.read(n) or b'{}')
        except Exception:
            return self._json({'error': 'битый запрос'}, 400)
        path = self.path.split('?', 1)[0]
        it = self._item(str(req.get('id', '')))
        if not it:
            return self._json({'error': 'нет такого фото'}, 404)

        if path == '/api/prime':
            how = ensure_primed(it)
            nxt = req.get('next')
            if nxt:
                n_it = self._item(str(nxt))
                if n_it:
                    threading.Thread(target=prefetch, args=(n_it,), daemon=True).start()
            return self._json({'ok': True, 'how': how})

        if path == '/api/predict':
            pts = req.get('points') or []
            box = req.get('box')
            if not pts and not box:
                return self._json({'error': 'нужен клик или рамка'}, 400)
            try:
                mask, score, total = predict(it, pts, box, int(req.get('variant', 0)))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
            return self._json({
                'png': 'data:image/png;base64,' + base64.b64encode(png_rgba(mask)).decode(),
                'score': round(score, 3), 'variants': total,
                'area': int(mask.sum())})

        if path == '/api/save':
            try:
                nails = save_labels(it, req['png'], req.get('meta'))
            except Exception as e:
                return self._json({'error': f'{type(e).__name__}: {e}'}, 500)
            return self._json({'ok': True, 'nails': nails})

        self._json({'error': 'не найдено'}, 404)


def precompute():
    """Посчитать эмбеддинги всех фото заранее.

    Кодировщик ViT-B на этом процессоре — полторы минуты на снимок. Если
    считать по ходу разметки, человек будет ждать на каждом втором фото.
    Поэтому весь счёт делается один раз заранее, а инструмент потом работает
    из кэша мгновенно.
    """
    items = task_items()
    t0 = time.time()
    for i, it in enumerate(items, 1):
        how = prime(it)
        left = (time.time() - t0) / i * (len(items) - i)
        print(f'[{i}/{len(items)}] {it["file"]} — {how}; осталось ~{left / 60:.0f} мин',
              flush=True)
    print(f'Готово за {(time.time() - t0) / 60:.0f} мин', flush=True)


if __name__ == '__main__':
    if not os.path.exists(os.path.join(WORK, 'task.json')):
        raise SystemExit('Нет labels/task.json — сначала: python make_label_task.py')
    load_sam()
    if '--precompute' in sys.argv:
        precompute()
        raise SystemExit(0)
    items = task_items()
    done = sum(1 for t in items if status_of(t)['done'])
    print(f'Задание: {len(items)} фото, размечено {done}')
    print(f'Открой http://127.0.0.1:{PORT}  (остановить — Ctrl+C)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
