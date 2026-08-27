"""Экспорт обученной модели в ONNX — формат, который умеет onnxruntime-web.

Проверяем экспорт тут же: гоняем ту же картинку через PyTorch и через
onnxruntime и сравниваем. Расхождение означает, что в браузере модель поведёт
себя не так, как на обучении, и узнать об этом лучше здесь, а не на телефоне.
"""
import json
import os

import numpy as np
import torch

from train import NailNet, SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'nail-unet.onnx')


def main():
    net = NailNet()
    net.load_state_dict(torch.load(os.path.join(HERE, 'best.pt'), map_location='cpu'))
    net.eval()

    dummy = torch.rand(1, 3, SIZE, SIZE)
    torch.onnx.export(
        net, dummy, OUT,
        input_names=['image'], output_names=['mask'],
        # Батч динамический: в браузере иногда удобнее прогнать пачку кропов
        # за один вызов, а размер картинки у нас всегда 256×256.
        dynamic_axes={'image': {0: 'batch'}, 'mask': {0: 'batch'}},
        opset_version=17,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(OUT, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        ref = torch.sigmoid(net(dummy)).numpy()
    got = 1 / (1 + np.exp(-sess.run(None, {'image': dummy.numpy()})[0]))
    diff = float(np.abs(ref - got).max())

    info = {
        'файл': os.path.basename(OUT),
        'мегабайт': round(os.path.getsize(OUT) / 1e6, 2),
        'вход': [1, 3, SIZE, SIZE],
        'расхождение с PyTorch': round(diff, 6),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if diff > 1e-3:
        raise SystemExit('ONNX считает иначе, чем PyTorch — в браузер такое отдавать нельзя')


if __name__ == '__main__':
    main()
