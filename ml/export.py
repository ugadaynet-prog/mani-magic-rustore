"""Экспорт обученной модели в ONNX — формат, который умеет onnxruntime-web.

Проверяем экспорт тут же: гоняем ту же картинку через PyTorch и через
onnxruntime и сравниваем. Расхождение означает, что в браузере модель поведёт
себя не так, как на обучении, и узнать об этом лучше здесь, а не на телефоне.
"""
import json
import os
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

import train as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'nail-unet.onnx')


def export_checkpoint(checkpoint, out, size=576, overwrite=False):
    """Explicit paths; never replace an existing named candidate implicitly."""
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    T.SIZE = size
    torch.manual_seed(7)
    torch.set_num_threads(2)
    net = T.NailNet(pretrained=False)
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    net.eval()

    dummy = torch.rand(1, 3, size, size)
    torch.onnx.export(
        net, dummy, str(out),
        input_names=['image'], output_names=['mask'],
        # Батч динамический: в браузере иногда удобнее прогнать пачку кропов
        # за один вызов, а размер картинки у нас всегда 256×256.
        dynamic_axes={'image': {0: 'batch'}, 'mask': {0: 'batch'}},
        opset_version=13,   # opset 13 — для onnxruntime-android 1.20.0
        dynamo=False,        # legacy exporter — совместим с onnxruntime-android
    )

    # Новый экспортёр torch кладёт веса ОТДЕЛЬНЫМ файлом .onnx.data, а в самом
    # .onnx оставляет только граф. В браузере такой файл грузится и падает на
    # первом же тензоре: onnxruntime-web не умеет искать внешние данные.
    # Пересобираем в один файл — заодно это единственный формат, который можно
    # просто положить в сборку приложения.
    import onnx
    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    onnx.save_model(model, str(out), save_as_external_data=False)

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(out), sess_options=opts, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        ref = torch.sigmoid(net(dummy)).numpy()
    got = 1 / (1 + np.exp(-sess.run(None, {'image': dummy.numpy()})[0]))
    diff = float(np.abs(ref - got).max())

    info = {
        'файл': out.name,
        'мегабайт': round(out.stat().st_size / 1e6, 2),
        'вход': [1, 3, size, size],
        'расхождение с PyTorch': round(diff, 6),
        'sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    # 1.1 млн параметров это ~4.4 МБ fp32. Файл заметно меньше означает, что
    # веса опять уехали наружу, и в браузере он работать не будет.
    if out.stat().st_size < 3e6:
        raise SystemExit('в .onnx нет весов — они остались во внешнем файле')
    if diff > 1e-3:
        raise SystemExit('ONNX считает иначе, чем PyTorch — в браузер такое отдавать нельзя')
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default=os.path.join(HERE, 'best.pt'))
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--size', type=int, default=T.SIZE)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    export_checkpoint(args.checkpoint, args.out, args.size, args.overwrite)


if __name__ == '__main__':
    main()
