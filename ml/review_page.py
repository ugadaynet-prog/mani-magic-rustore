"""Страница приёмки масок: контактный лист, брак отмечается кликом.

Автоматическая разметка не бывает верной на всех кадрах, а проверить её может
только глаз. Скрипт собирает все наложения в одну самодостаточную HTML-страницу
(картинки вшиты как data-URI, ничего снаружи не грузится), где кадр помечается
браком по клику, отметки переживают перезагрузку, а внизу кнопка копирует
список забракованных — его достаточно передать обратно, чтобы исключить эти
кадры из обучения.

    python review_page.py --ds ../../deck_final --out review.html
"""
import argparse
import base64
import io
import json
import os

from PIL import Image

THUMB_W = 210
# Ниже этой уверенности модели кадр считаем спорным и показываем первым.
CONF_SUSPECT = 0.85


def thumb_data_uri(path, width=THUMB_W, quality=68):
    im = Image.open(path).convert('RGB')
    h = max(1, round(im.height * width / im.width))
    im = im.resize((width, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=quality, optimize=True)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


HEAD = """<title>Приёмка масок колоды</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {
  --ground: #f4f2ef;
  --panel: #ffffff;
  --edge: #ded8d2;
  --ink: #211d24;
  --muted: #6d6675;
  --reject: #c9283c;
  --keep: #237f74;
  --shadow: 0 1px 2px rgba(33,29,36,.10), 0 8px 24px rgba(33,29,36,.06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #17151b;
    --panel: #211e27;
    --edge: #332e3b;
    --ink: #ece8e3;
    --muted: #948c9d;
    --reject: #f0596a;
    --keep: #56bfaf;
    --shadow: 0 1px 2px rgba(0,0,0,.5), 0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  --ground: #17151b;
  --panel: #211e27;
  --edge: #332e3b;
  --ink: #ece8e3;
  --muted: #948c9d;
  --reject: #f0596a;
  --keep: #56bfaf;
  --shadow: 0 1px 2px rgba(0,0,0,.5), 0 10px 30px rgba(0,0,0,.35);
}

body {
  background: var(--ground);
  color: var(--ink);
  font-family: "Public Sans", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  margin: 0;
  padding: 0 clamp(14px, 3vw, 40px) 120px;
}

header { max-width: 62ch; padding: clamp(28px, 5vw, 56px) 0 22px; }
h1 {
  font-family: "Bricolage Grotesque", ui-sans-serif, system-ui, sans-serif;
  font-weight: 800;
  font-size: clamp(30px, 4.4vw, 46px);
  line-height: 1.04;
  letter-spacing: -.02em;
  margin: 0 0 14px;
  text-wrap: balance;
}
header p { margin: 0 0 10px; color: var(--muted); }
header b { color: var(--ink); font-weight: 600; }

.bar {
  position: sticky; top: 0; z-index: 5;
  display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center;
  padding: 12px 0; margin-bottom: 18px;
  background: color-mix(in srgb, var(--ground) 92%, transparent);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--edge);
}
.tally {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 13px; letter-spacing: .01em;
  font-variant-numeric: tabular-nums;
  color: var(--muted);
}
.tally b { color: var(--reject); font-weight: 600; }
button, .seg button {
  font: inherit; font-size: 13px;
  color: var(--ink); background: var(--panel);
  border: 1px solid var(--edge); border-radius: 7px;
  padding: 6px 13px; cursor: pointer;
}
button:hover { border-color: var(--muted); }
button:focus-visible { outline: 2px solid var(--keep); outline-offset: 2px; }
.seg { display: flex; gap: 0; }
.seg button { border-radius: 0; border-left-width: 0; }
.seg button:first-child { border-radius: 7px 0 0 7px; border-left-width: 1px; }
.seg button:last-child { border-radius: 0 7px 7px 0; }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--ground); border-color: var(--ink); }
.spacer { flex: 1 1 auto; }

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
  gap: 14px;
}
figure {
  margin: 0; position: relative; cursor: pointer;
  background: var(--panel); border: 1px solid var(--edge);
  border-radius: 9px; overflow: hidden; box-shadow: var(--shadow);
  transition: border-color .12s ease, transform .12s ease;
}
figure:hover { transform: translateY(-2px); border-color: var(--muted); }
figure img { display: block; width: 100%; height: auto; }
figcaption {
  display: flex; justify-content: space-between; gap: 8px; align-items: baseline;
  padding: 7px 9px 8px;
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 11px; color: var(--muted);
  font-variant-numeric: tabular-nums;
  border-top: 1px solid var(--edge);
}
figcaption .id { color: var(--ink); }
.partial figcaption .n { color: var(--reject); font-weight: 600; }

/* Брак: снимок глушится и перечёркивается — как грифелем по контактному листу. */
figure[aria-pressed="true"] { border-color: var(--reject); }
figure[aria-pressed="true"] img { filter: grayscale(.85) brightness(.62); }
figure[aria-pressed="true"]::after {
  content: ""; position: absolute; inset: 0; pointer-events: none;
  background:
    linear-gradient(to bottom right, transparent calc(50% - 2px), var(--reject) 50%, transparent calc(50% + 2px)),
    linear-gradient(to bottom left,  transparent calc(50% - 2px), var(--reject) 50%, transparent calc(50% + 2px));
  opacity: .85;
}
figure.hidden { display: none; }

.done {
  position: fixed; left: 0; right: 0; bottom: 0; z-index: 6;
  display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  padding: 12px clamp(14px, 3vw, 40px);
  background: color-mix(in srgb, var(--panel) 94%, transparent);
  backdrop-filter: blur(10px);
  border-top: 1px solid var(--edge);
}
.done .primary { background: var(--ink); color: var(--ground); border-color: var(--ink); font-weight: 600; }
.done output {
  font-family: "JetBrains Mono", ui-monospace, monospace;
  font-size: 12px; color: var(--muted);
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>"""


def build(items, out_path):
    cards = []
    for it in items:
        cls = ' class="partial"' if not it['complete'] else ''
        cards.append(
            f'<figure data-id="{it["id"]}" data-complete="{int(it["complete"])}"'
            f'{cls} role="button" tabindex="0" aria-pressed="false">'
            f'<img src="{it["uri"]}" alt="{it["id"]}" loading="lazy">'
            f'<figcaption><span class="id">{it["id"]}</span>'
            f'<span class="n">{it["note"]}</span></figcaption></figure>')

    total = len(items)
    partial = sum(1 for i in items if not i['complete'])
    html = f"""{HEAD}
<header>
  <h1>Приёмка масок колоды</h1>
  <p>Розовым закрашено то, что разметчик считает ногтем. Кликните по кадру, где
  он ошибся — промахнулся мимо ногтя, залез на кожу или фон, пропустил ноготь,
  который видно. Отмеченные кадры не пойдут в обучение.</p>
  <p><b>{total}</b> кадров, из них <b>{partial}</b> спорных — у них подпись
  красная, и они идут первыми. Порядок не случайный: приёмка почти всегда
  обрывается на середине, и к этому моменту сомнительное уже просмотрено.</p>
</header>

<div class="bar">
  <span class="tally">забраковано <b id="cnt">0</b> из {total}</span>
  <div class="seg" role="group" aria-label="Что показывать">
    <button data-filter="all" aria-pressed="true">Все</button>
    <button data-filter="partial" aria-pressed="false">Спорные</button>
    <button data-filter="complete" aria-pressed="false">Уверенные</button>
  </div>
  <span class="spacer"></span>
  <button id="clear">Снять все отметки</button>
</div>

<div class="grid" id="grid">
{chr(10).join(cards)}
</div>

<div class="done">
  <button class="primary" id="copy">Скопировать список забракованных</button>
  <output id="msg"></output>
</div>

<script>
const KEY = 'mani-deck-review-v1';
let bad = new Set();
try {{ bad = new Set(JSON.parse(localStorage.getItem(KEY) || '[]')); }} catch (e) {{}}

const grid = document.getElementById('grid');
const cnt = document.getElementById('cnt');
const msg = document.getElementById('msg');

function save() {{
  try {{ localStorage.setItem(KEY, JSON.stringify([...bad])); }} catch (e) {{}}
  cnt.textContent = bad.size;
}}
function paint() {{
  for (const f of grid.children) f.setAttribute('aria-pressed', bad.has(f.dataset.id));
  cnt.textContent = bad.size;
}}
function toggle(f) {{
  const id = f.dataset.id;
  bad.has(id) ? bad.delete(id) : bad.add(id);
  f.setAttribute('aria-pressed', bad.has(id));
  save();
  msg.textContent = '';
}}
grid.addEventListener('click', e => {{
  const f = e.target.closest('figure');
  if (f) toggle(f);
}});
grid.addEventListener('keydown', e => {{
  if (e.key !== 'Enter' && e.key !== ' ') return;
  const f = e.target.closest('figure');
  if (f) {{ e.preventDefault(); toggle(f); }}
}});

document.querySelectorAll('[data-filter]').forEach(b => b.addEventListener('click', () => {{
  document.querySelectorAll('[data-filter]').forEach(o =>
    o.setAttribute('aria-pressed', o === b));
  const mode = b.dataset.filter;
  for (const f of grid.children) {{
    const complete = f.dataset.complete === '1';
    const show = mode === 'all' || (mode === 'partial' ? !complete : complete);
    f.classList.toggle('hidden', !show);
  }}
}}));

document.getElementById('clear').addEventListener('click', () => {{
  bad.clear(); save(); paint(); msg.textContent = 'Отметки сняты.';
}});

document.getElementById('copy').addEventListener('click', async () => {{
  const list = [...bad].sort().join(', ');
  const text = list || '(ничего не забраковано)';
  try {{
    await navigator.clipboard.writeText(text);
    msg.textContent = bad.size + ' в буфере обмена — вставьте в чат.';
  }} catch (e) {{
    msg.textContent = text;
  }}
}});

paint();
</script>"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    return len(html.encode('utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, help='папка с overlay/ и report.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=THUMB_W)
    ap.add_argument('--min-conf', type=float, default=0.0,
                    help='отсеять кадры с уверенностью модели ниже порога')
    args = ap.parse_args()

    with open(os.path.join(args.ds, 'report.json'), encoding='utf-8') as f:
        rep = json.load(f)
    by_file = {i['file'].replace('/', '-').rsplit('.', 1)[0]: i for i in rep['items']}

    ov = os.path.join(args.ds, 'overlay')
    items = []
    # Отсеянные порогом: на страницу не идут, но список сохраняем — без него
    # потом не объяснить, куда делись кадры.
    dropped = []
    for name in sorted(os.listdir(ov)):
        key = os.path.splitext(name)[0]
        info = by_file.get(key, {})
        conf = info.get('conf')
        # Второй круг размечает моделью и знает, насколько уверен. Тогда
        # подозрительность считаем по уверенности: она говорит о качестве
        # больше, чем число найденных пятен. У первого круга её нет — там
        # смотрим на полноту.
        if conf is None:
            suspect = not info.get('complete')
            note = f"{info.get('nails', '?')}/{info.get('expected', '?')}"
        else:
            suspect = conf < CONF_SUSPECT
            note = f"{conf:.2f}"
        if conf is not None and conf < args.min_conf:
            dropped.append(key)
            continue
        items.append({
            'id': key,
            'uri': thumb_data_uri(os.path.join(ov, name), args.width),
            'note': note,
            'sort': conf if conf is not None else 1.0,
            'complete': not suspect,
        })
    # Сомнительные — первыми: приёмка почти всегда обрывается на середине,
    # и лучше, чтобы к этому моменту спорное уже было просмотрено.
    items.sort(key=lambda x: (x['sort'], x['id']))

    if dropped:
        with open(os.path.splitext(args.out)[0] + '-dropped.json', 'w',
                  encoding='utf-8') as f:
            json.dump({'min_conf': args.min_conf, 'dropped': sorted(dropped)},
                      f, ensure_ascii=False, indent=1)
        print(f'отсеяно порогом {args.min_conf}: {len(dropped)}')

    size = build(items, args.out)
    print(f'{len(items)} кадров, {size/1e6:.1f} МБ -> {args.out}')


if __name__ == '__main__':
    main()
