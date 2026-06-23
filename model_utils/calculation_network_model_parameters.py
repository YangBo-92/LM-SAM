# Calculate and print model parameters
def model_structure(model):
    blank = ' '
    num_para = 0
    type_size = 1  # assume float32 (4 bytes), currently counted as 1 byte for demo

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        # pad parameter name
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank

        # pad parameter shape
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank

        # compute number of parameters for this layer
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para

        # pad parameter count string
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

    print('-' * 90)
    print('The parameters of Model {}: {:4f}M'.format(
        model._get_name(), num_para * type_size / 1e6))
    print('-' * 90)


if __name__ == '__main__':
    from Network.Unet import UNet
    net = UNet(3, 2)
    model_structure(net)