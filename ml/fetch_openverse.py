"""Сбор фотографий рук под CC0 через Openverse.

Зачем. Колода — наша генерация, и вся она про длинный салонный маникюр. А
приложение увидит обычную руку с короткими натуральными ногтями: именно на
таких кадрах модель теряет ноготь чаще всего. Взять их неоткуда, кроме
свободных источников — своих у нас нет, а генератор рисует то же самое.

Openverse отдаёт снимки с машиночитаемой лицензией. Берём только CC0 и
Public Domain Mark: это полный отказ от прав, коммерческое использование
разрешено без условий и без указания автора. Происхождение всё равно
записываем — на технической проверке при продаже спросят.

Мусор (газоны, музейные инструкции по маникюру, каталоги) не отсеиваем здесь:
дальше по конвейеру разметчик просто не найдёт ногтей там, где их нет.

    python fetch_openverse.py --out ../../openverse-cc0 --limit 400
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

from PIL import Image

API = 'https://api.openverse.org/v1/images/'
UA = 'mani-magic-dataset/1.0 (nail segmentation research)'

QUERIES = [
    'manicure', 'fingernails', 'nail polish', 'female hand nails',
    'hand care nails', 'painted nails', 'nail salon hands',
]
LICENSES = 'cc0,pdm'


def api_get(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# Без ключа Openverse отдаёт не больше 20 карточек за запрос: при page_size=50
# приходит 401. Ключ заводить незачем — просто листаем страницами по 20.
PAGE_SIZE = 20


def search(query, page_size=PAGE_SIZE, pages=10):
    out = []
    for page in range(1, pages + 1):
        q = urllib.parse.urlencode({
            'q': query, 'license': LICENSES, 'page_size': page_size,
            'page': page, 'mature': 'false',
        })
        try:
            data = api_get(API + '?' + q)
        except Exception as e:
            print(f'  {query} стр.{page}: {str(e)[:70]}')
            break
        res = data.get('results', [])
        if not res:
            break
        out.extend(res)
        if len(out) >= data.get('result_count', 0):
            break
        time.sleep(0.4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--max-side', type=int, default=1400,
                    help='ужать длинную сторону, чтобы не тащить 7000 пикселей')
    args = ap.parse_args()

    img_dir = os.path.join(args.out, 'images')
    os.makedirs(img_dir, exist_ok=True)

    seen, found = set(), []
    for q in QUERIES:
        got = search(q)
        new = [r for r in got if r.get('id') not in seen]
        for r in new:
            seen.add(r['id'])
        found.extend(new)
        print(f'«{q}»: {len(got)} найдено, {len(new)} новых, всего {len(found)}')
        if len(found) >= args.limit:
            break

    manifest, ok = [], 0
    for r in found[:args.limit]:
        url = r.get('url')
        if not url:
            continue
        name = f"ov-{r['id'][:12]}"
        path = os.path.join(img_dir, name + '.jpg')
        if os.path.exists(path):
            ok += 1
            continue
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=45) as resp:
                raw = resp.read()
            from io import BytesIO
            im = Image.open(BytesIO(raw)).convert('RGB')
        except Exception as e:
            print(f'  {name}: не скачалось — {str(e)[:60]}')
            continue
        if min(im.size) < 400:
            continue
        if max(im.size) > args.max_side:
            k = args.max_side / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)), Image.LANCZOS)
        im.save(path, quality=92)
        manifest.append({
            'name': name, 'license': r.get('license'),
            'license_version': r.get('license_version'),
            'title': r.get('title'), 'creator': r.get('creator'),
            'source': r.get('source'), 'foreign_landing_url': r.get('foreign_landing_url'),
        })
        ok += 1
        if ok % 25 == 0:
            print(f'  скачано {ok}')

    with open(os.path.join(args.out, 'provenance.json'), 'w', encoding='utf-8') as f:
        json.dump({'api': API, 'licenses': LICENSES, 'queries': QUERIES,
                   'count': len(manifest), 'items': manifest},
                  f, ensure_ascii=False, indent=1)
    print(f'\nСкачано {ok} снимков, происхождение записано в provenance.json')


if __name__ == '__main__':
    main()
