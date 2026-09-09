"""Набрать снимки рук под CC0 — с манифестом, откуда взялся каждый кадр.

Зачем. Обучающих источников у нас два: колода MANI Magic и CC0-набор vpapenko
на 52 кадра. Чужие снимки из открытых источников 8 сентября убраны из обучения
(PROVENANCE.md), и заменить их можно только тем, на что права отданы явно.
Openverse собирает Flickr, Wikimedia, rawpixel и музеи и отдаёт лицензию по
каждому файлу — значит, набор можно собрать так, чтобы на вопрос «откуда это»
был ответ по каждому кадру, а не по набору целиком.

Берём только CC0 и Public Domain Mark. CC-BY тоже разрешает коммерческое
использование, но требует называть автора — при продаже приложения это
обязательство, тянущееся за каждым кадром.

Фильтр по руке обязателен. По запросу «fingernails» под CC0 приходят
пресноводный моллюск Fingernail Clam, музейные напёрстки, айсберг и лягушка:
поиск ищет слово, а не кисть. MediaPipe отбраковывает такое дёшево и молча.

    python fetch_cc0.py --limit 300     # скачать кандидатов и отобрать
    python fetch_cc0.py --filter        # только отбор, ничего не качая

Пишет в папку рядом с проектом (по умолчанию «КОЛОДА ПРИЛ/cc0-руки»):
photos/ — что скачано, отобрано/ — что прошло фильтр, манифест.json и
отобрано.json — источник и лицензия по каждому файлу.
"""
import argparse
import io
import json
import os
import time
import urllib.parse
import urllib.request

from PIL import Image

import make_label_task2 as L2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'cc0-руки'))
API = 'https://api.openverse.org/v1/images/'
UA = 'mani-magic-dataset/1.0 (CC0 collection for nail segmentation training)'
# Больше 20 на страницу API без ключа не отдаёт.
PAGE = 20
# Мелкие кадры не нужны: ноготь на них в несколько пикселей, размечать нечего.
MIN_SIDE = 600
# Хранить в исходном разрешении незачем: обучение всё равно вписывает в 512.
WORK_SIDE = 1600

QUERIES = [
    'fingernails', 'fingernail', 'manicure', 'nail polish', 'natural nails',
    'woman hands', 'hands fingers', 'hand closeup', 'female hand',
    'hands nails', 'painted nails', 'nail art', 'hand skin', 'human palm',
]


def search(q, pages=3, lic='cc0,pdm'):
    out = []
    for page in range(1, pages + 1):
        url = (API + '?q=' + urllib.parse.quote(q) + '&license=' + lic
               + '&page_size=' + str(PAGE) + '&page=' + str(page))
        try:
            req = urllib.request.Request(
                url, headers={'Accept': 'application/json', 'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                res = json.load(r).get('results', [])
        except Exception as e:
            print('  ' + q + ' стр.' + str(page) + ': ' + str(e), flush=True)
            break
        if not res:
            break
        out += res
        time.sleep(0.4)
    return out


def grab(item, photos):
    """Скачать один кадр. Возвращает запись манифеста или None."""
    w, h = item.get('width') or 0, item.get('height') or 0
    if w and h and max(w, h) < MIN_SIDE:
        return None
    name = item['id'][:12] + '.jpg'
    path = os.path.join(photos, name)
    if not os.path.exists(path):
        try:
            req = urllib.request.Request(item['url'], headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            im = Image.open(io.BytesIO(raw)).convert('RGB')
        except Exception as e:
            print('  ' + item['id'][:12] + ': ' + str(e), flush=True)
            return None
        if max(im.size) < MIN_SIDE:
            return None
        k = WORK_SIDE / max(im.size)
        if k < 1:
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        im.save(path, quality=92, subsampling=0)
    return {'file': name, 'id': item['id'], 'title': item.get('title'),
            'creator': item.get('creator'), 'source': item.get('source'),
            'license': item.get('license'),
            'license_version': item.get('license_version'),
            'license_url': item.get('license_url'),
            'page': item.get('foreign_landing_url'), 'url': item.get('url')}


def download(out, limit):
    photos = os.path.join(out, 'photos')
    os.makedirs(photos, exist_ok=True)
    mpath = os.path.join(out, 'манифест.json')
    manifest, seen = [], set()
    if os.path.exists(mpath):
        with open(mpath, encoding='utf-8') as fh:
            manifest = json.load(fh)['кадры']
        seen = set(m['id'] for m in manifest)
    for q in QUERIES:
        if len(manifest) >= limit:
            break
        found = search(q)
        took = 0
        for it in found:
            if len(manifest) >= limit:
                break
            if it['id'] in seen:
                continue
            seen.add(it['id'])
            rec = grab(it, photos)
            if rec:
                rec['запрос'] = q
                manifest.append(rec)
                took += 1
        print('%-16s найдено %3d, взято %3d, всего %d'
              % (q, len(found), took, len(manifest)), flush=True)
    with open(mpath, 'w', encoding='utf-8') as fh:
        json.dump({'источник': 'Openverse (api.openverse.org)',
                   'лицензии': 'CC0-1.0 и Public Domain Mark',
                   'собрано': time.strftime('%Y-%m-%d'),
                   'кадры': manifest}, fh, ensure_ascii=False, indent=1)
    print('\nскачано ' + str(len(manifest)) + ' кадров в ' + photos)


def keep(out):
    """Оставить кадры, где есть кисть заметного размера и нет совпадений с экзаменом."""
    photos = os.path.join(out, 'photos')
    good_dir = os.path.join(out, 'отобрано')
    os.makedirs(good_dir, exist_ok=True)
    with open(os.path.join(out, 'манифест.json'), encoding='utf-8') as fh:
        data = json.load(fh)
    det = L2.make_hand_detector()
    gold = []
    for p in (os.path.join(HERE, 'labels', 'photos'),
              os.path.join(HERE, 'labels3', 'photos')):
        if os.path.isdir(p):
            gold += [L2.ahash(os.path.join(p, f)) for f in os.listdir(p)]
    kept, hashes = [], []
    reasons = {'нет руки': 0, 'рука мелкая': 0, 'совпало с экзаменом': 0, 'дубль': 0}
    for m in data['кадры']:
        p = os.path.join(photos, m['file'])
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert('RGB')
        frac = L2.hand_fraction(det, im)
        if frac <= 0:
            reasons['нет руки'] += 1
            continue
        if frac < L2.HAND_MIN_FRAC:
            reasons['рука мелкая'] += 1
            continue
        h = L2.ahash(im)
        if gold and min(int((h != g).sum()) for g in gold) < L2.HAMMING_GOLD:
            reasons['совпало с экзаменом'] += 1
            continue
        if hashes and min(int((h != g).sum()) for g in hashes) < L2.HAMMING_SELF:
            reasons['дубль'] += 1
            continue
        hashes.append(h)
        m['доля кисти'] = round(float(frac), 3)
        kept.append(m)
        im.save(os.path.join(good_dir, m['file']), quality=92, subsampling=0)
    with open(os.path.join(out, 'отобрано.json'), 'w', encoding='utf-8') as fh:
        json.dump({'из': len(data['кадры']), 'отброшено': reasons, 'кадры': kept},
                  fh, ensure_ascii=False, indent=1)
    print('осталось ' + str(len(kept)) + ' из ' + str(len(data['кадры'])))
    for k, v in reasons.items():
        print('  ' + k + ': ' + str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--limit', type=int, default=300)
    ap.add_argument('--filter', action='store_true',
                    help='только отбор, ничего не скачивая')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.filter:
        download(args.out, args.limit)
    keep(args.out)


if __name__ == '__main__':
    main()
