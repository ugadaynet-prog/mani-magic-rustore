"""Разделить снимки владельца на обучение и экзамен «свои».

Зачем экзамен. Все прежние экзамены собраны из Pinterest и стока, а приложение
будет видеть другое: человек фотографирует свою руку. Замер 10 сентября это
вскрыл — 21 кадр владельца в обучении улучшил набор «без лака» и ухудшил
Pinterest-контроль, и решить, лучше ли стала модель, оказалось не по чему.
Экзамен «свои» — ближайшее к настоящему сценарию, что у нас есть.

Почему по съёмкам. Кадры одного ролика или одной клиентки почти одинаковы;
если один уйдёт в обучение, а соседний в экзамен, модель сдаст экзамен по
памяти. Поэтому делятся группы целиком, а не кадры.

Что в экзамене. Треть кадров, и среди них те же трудные случаи, что и в
обучении: френч, нюд, макро, тёмный лак, белое на белом. Отдать все трудные
кадры в экзамен нельзя — модели не на чем будет им учиться; отдать все в
обучение тоже нельзя — мерить станет нечем.

    python make_own_split.py

Пишет labels8/ — экзамен «свои» в формате задания (фото, маски, номера
ногтей), и own-train/ — обучающую часть для шифрования и отправки в CI.
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
EXAM = os.path.join(HERE, 'labels8')
TRAIN = os.path.join(HERE, 'own-train')

# Съёмки: какие кадры сняты вместе и потому делятся только вместе.
GROUPS = {
    'фиолетовый фон, пять оттенков': [('labels6', '03'), ('labels6', '04'),
                                      ('labels6', '05'), ('labels6', '06')],
    'красный на белом столе': [('labels6', '07'), ('labels6', '08')],
    'серо-голубой, две руки, френч': [('labels6', '09'), ('labels6', '10')],
    'журнал ROSES': [('labels6', '11'), ('labels6', '12'), ('labels6', '13')],
    'сиреневое матовое макро': [('labels6', '02'), ('labels6', '14')],
    'синее матовое макро': [('labels6', '15')],
    'хаки матовое макро': [('labels6', '16')],
    'тауп с кольцами': [('labels6', '01')],
    'нюд миндаль, УФ-лампа и перчатка': [('labels6', '17'), ('labels6', '18')],
    'бордо в перчатке': [('labels6', '19'), ('labels6', '20')],
    'френч на двух руках': [('labels6', '21')],
    'френч с телефоном': [('labels7', '01')],
    'нюд с золотой полосой, макро': [('labels7', '02'), ('labels7', '03')],
    'френч на кожаном кресле': [('labels7', '04')],
    'бордо на белой ткани': [('labels7', '05')],
    'белый френч на белом свитере': [('labels7', '06')],
    'обратный френч, макро': [('labels7', '07')],
    'бледно-розовый с фольгой': [('labels7', '08')],
    'френч на джинсах': [('labels7', '09')],
}
TO_EXAM = [
    'серо-голубой, две руки, френч',
    'бордо в перчатке',
    'хаки матовое макро',
    'нюд с золотой полосой, макро',
    'френч на кожаном кресле',
    'белый френч на белом свитере',
    'френч на джинсах',
]


def copy_frame(work, iid, dst, new_id):
    for sub, ext in (('photos', 'jpg'), ('masks', 'png'), ('instances', 'png'), ('meta', 'json')):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)
        src = os.path.join(HERE, work, sub, f'{iid}.{ext}')
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(dst, sub, f'{new_id}.{ext}'))


def main():
    for d in (EXAM, TRAIN):
        shutil.rmtree(d, ignore_errors=True)
    items = {}
    for w in ('labels6', 'labels7'):
        for t in json.load(open(os.path.join(HERE, w, 'task.json'), encoding='utf-8'))['items']:
            items[(w, t['id'])] = t

    exam_items, train_items = [], []
    for name, frames in GROUPS.items():
        side = TO_EXAM if name in TO_EXAM else None
        for w, iid in frames:
            t = items[(w, iid)]
            new_id = ('a' if w == 'labels6' else 'b') + iid
            if side:
                copy_frame(w, iid, EXAM, new_id)
                exam_items.append(dict(t, id=new_id, file=f'{new_id}.jpg', part='свои',
                                       group=name, src=f'{w}/{iid}'))
            else:
                copy_frame(w, iid, TRAIN, new_id)
                train_items.append(dict(t, id=new_id, file=f'{new_id}.jpg', part='свои',
                                        group=name, src=f'{w}/{iid}'))

    covered = {(w, i) for fr in GROUPS.values() for w, i in fr}
    missing = set(items) - covered
    if missing:
        raise SystemExit(f'кадры без группы: {sorted(missing)}')

    json.dump({'items': exam_items}, open(os.path.join(EXAM, 'task.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    json.dump({'items': train_items}, open(os.path.join(TRAIN, 'task.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'экзамен «свои»: {len(exam_items)} кадров → {os.path.relpath(EXAM, HERE)}')
    print(f'обучение: {len(train_items)} кадров → {os.path.relpath(TRAIN, HERE)}')


if __name__ == '__main__':
    main()
