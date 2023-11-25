import cv2
import numpy as np
import onnxruntime as rt
import torch
import torchvision


def init_model(model, weight_path, device):
    """Initialize model using pretrained weight"""

    model_dict = model.state_dict()
    if next(model.parameters()).device == torch.device('cpu'):
        pretrained_dict = torch.load(weight_path)
    else:
        pretrained_dict = torch.load(weight_path, map_location=device)
    for k, v in pretrained_dict.items():
        if k not in model_dict or v.shape != model_dict[k].shape:
            raise AttributeError('Model weight mismatch!')
    model.load_state_dict(pretrained_dict)
    model.to(device)
    return model


def run_resnet50():
    model_weight_path = '../model_weights/resnet50-19c8e357.pth'
    onnx_model_path = '../model/resnet50.onnx'
    sample_img_path = '../input/n01491361_tiger_shark.JPEG'
    label_path = '../input/imagenet_classes.txt'
    device = torch.device('cpu')
    test_batch_size = 4

    model = torchvision.models.resnet50(weights=None)
    model = init_model(model, model_weight_path, device)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    input_names = ['input']
    output_names = ['output']
    dynamic_axes_dict = {'input': {0: 'batch'}, 'output': {0: 'batch'}}

    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=12,
        dynamic_axes=dynamic_axes_dict,
    )

    img = cv2.imread(sample_img_path, 0).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img -= mean
    img /= std
    img = np.transpose(img, (2, 0, 1))
    # (3, 224, 224) -> (test_batch_size, 3, 224, 224)
    img = np.repeat(img[np.newaxis, :, :], test_batch_size, axis=0)

    sess = rt.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    onnx_inputs = {sess.get_inputs()[0].name: img}
    pred_onx = sess.run(output_names=output_names, input_feed=onnx_inputs)

    with open(label_path) as file:
        label = [line.rstrip() for line in file]

    pred_inds = np.argmax(pred_onx[0], axis=1)
    print([label[pred_ind] for pred_ind in pred_inds])


if __name__ == '__main__':
    run_resnet50()
