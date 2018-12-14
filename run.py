import argparse
import numpy as np
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import google.protobuf as pb
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--deploy', action='store', dest='deploy', help='deploy prototxt')
    parser.add_argument('-m', '--model', action='store', dest='model', help='caffemodel')
    parser.add_argument('-t', '--test', action='store_true', dest='test', help='run test')
    parser.add_argument('--height', type=int, default=-1, help='Used to generate random sample if need to run test')
    parser.add_argument('--width', type=int, default=-1, help='Used to generate random sample if need to run test')
    args = parser.parse_args()
    return args


def load_network_def(model_def):
    """Reads a .prototxt file that defines the network.
    """
    with open(model_def) as f:
        net = caffe_pb2.NetParameter()
        pb.text_format.Parse(f.read(), net)
    return net


def get_layer_by_name(proto, name):
    for i in xrange(len(proto.layer)):
        if proto.layer[i].name == name:
            return proto.layer[i]
    return None


def get_conv_layer_name(proto, name):
    layer = get_layer_by_name(proto, name)
    if not layer:
        return None
    if layer.type == u'Scale':
        bottom_layer = get_layer_by_name(proto, layer.bottom[0])
        if bottom_layer and bottom_layer.type == u'BatchNorm':
            bottom2_layer = get_layer_by_name(proto, bottom_layer.bottom[0])
            if bottom2_layer and bottom2_layer.type == u'Convolution':
                return bottom2_layer.name
    elif layer.type == u'BatchNorm':
        bottom_layer = get_layer_by_name(proto, layer.bottom[0])
        if bottom_layer and bottom_layer.type == u'Convolution':
            return bottom_layer.name
    return None


def dump_weight_to_hist(weight, path, bins=1024):
    _data = np.reshape(weight, [-1])
    plt.clf()
    plt.cla()
    plt.hist(_data, bins)
    plt.savefig(path)


class ConvBnFusioner:

    postfix = '_folded'

    def __init__(self, network, model):
        self.network = network
        self.model = model
        caffe.set_mode_cpu()
        self.network_def = load_network_def(self.network)
        self.orig_net = caffe.Net(network, model, caffe.TEST)
        self.net = self.orig_net

    def get_bn_epsilon(self, conv_layers_bn):
        res = {}
        for layer in self.network_def.layer:
            if layer.name in conv_layers_bn.values():
                res[layer.name] = layer.batch_norm_param.eps
        return res

    def eliminate_bn(self):
        conv_layers_bn = {}
        conv_layers_sc = {}
        proto = caffe_pb2.NetParameter()
        text_format.Merge(open(self.network).read(), proto)

        # change network topology
        for i in xrange(len(proto.layer)):
            layer = proto.layer[i]
            if layer.type == u'BatchNorm' or layer.type == u'Scale':
                continue
            for j in xrange(len(layer.top)):
                conv_layer_name = get_conv_layer_name(proto, layer.top[j])
                if conv_layer_name:
                    layer.top[j] = conv_layer_name + self.postfix
            for j in xrange(len(layer.bottom)):
                conv_layer_name = get_conv_layer_name(proto, layer.bottom[j])
                if conv_layer_name:
                    layer.bottom[j] = conv_layer_name + self.postfix

        # loop again to remove BN and SC
        i = len(proto.layer)
        while i > 0:
            i -= 1
            layer = proto.layer[i]
            if layer.type == u'BatchNorm' or layer.type == u'Scale':
                conv_layer_name = get_conv_layer_name(proto, layer.name)
                if conv_layer_name:
                    if layer.type == u'BatchNorm':
                        conv_layers_bn[conv_layer_name] = layer.name
                        conv_layer = get_layer_by_name(proto, conv_layer_name)
                        conv_layer.convolution_param.bias_term = True
                        for j in xrange(len(conv_layer.top)):
                            if conv_layer.top[j] == conv_layer.name:
                                conv_layer.top[j] += self.postfix
                        conv_layer.name += self.postfix
                    else:
                        conv_layers_sc[conv_layer_name] = layer.name
                    proto.layer.remove(layer)

        outproto = self.network.replace('.prototxt', '{}.prototxt'.format(self.postfix))
        outmodel = self.model.replace('.caffemodel', '{}.caffemodel'.format(self.postfix))

        with open(outproto, 'w') as f:
            f.write(str(proto))

        bn_eps = self.get_bn_epsilon(conv_layers_bn)

        new_w = {}
        new_b = {}

        print('# -----------------Layers to be folded----------------- #')
        for i in conv_layers_bn:
            print('{:<30}    |   {:<30}   |   {:<30}'.format(i, conv_layers_bn[i], conv_layers_sc[i]))
        print('# ----------------------------------------------------- #')

        for layer in conv_layers_bn:
            print('==============================')
            print('Process layer {}'.format(layer))
            print('Value Range')

            old_w = self.orig_net.params[layer][0].data
            old_w = old_w.astype(np.float32)

            # dump_weight_to_hist(old_w.reshape([-1]), os.path.join('vis', '{}_orig.jpg'.format(layer)))

            print('          --orig weight {} {} {}'.format(old_w.shape, np.max(old_w), np.min(old_w)))

            # Get weight and bias
            if len(self.orig_net.params[layer]) > 1:
                old_b = self.orig_net.params[layer][1].data
            else:
                old_b = np.zeros(self.orig_net.params[layer][0].data.shape[0],
                                 self.orig_net.params[layer][0].data.dtype)
            print('          --orig bias {} {} {}'.format(old_b.shape, np.max(old_b) , np.min(old_b)))

            # Get bn scale
            if self.orig_net.params[conv_layers_bn[layer]][2].data[0] != 0:
                s = 1 / self.orig_net.params[conv_layers_bn[layer]][2].data[0]
            else:
                s = 0
            assert(s == 1)

            u = self.orig_net.params[conv_layers_bn[layer]][0].data * s # mean
            v = self.orig_net.params[conv_layers_bn[layer]][1].data * s # variance
            alpha = self.orig_net.params[conv_layers_sc[layer]][0].data # alpha
            beta = self.orig_net.params[conv_layers_sc[layer]][1].data # beta

            print('          --mean {} {} {}'.format(u.shape, np.max(u) , np.min(u)))
            print('          --variance {} {} {}'.format(v.shape, np.max(v) , np.min(v)))
            print('          --gamma {} {} {}'.format(alpha.shape, np.max(alpha) , np.min(alpha)))
            print('          --beta {} {} {}'.format(beta.shape, np.max(beta) , np.min(beta)))

            new_b[layer] = (((old_b - u) * alpha) / (np.sqrt(v + bn_eps[conv_layers_bn[layer]]))) + beta
            scale = (alpha / (np.sqrt(v + bn_eps[conv_layers_bn[layer]])))
            new_w[layer] = old_w * scale[..., np.newaxis, np.newaxis, np.newaxis]

            # dump_weight_to_hist(new_w[layer].reshape([-1]), os.path.join('vis', '{}_new.jpg'.format(layer)))

            print('')
            print('          --new weight {} {} {}'.format(new_w[layer].shape, np.max(new_w[layer]) , np.min(new_w[layer])))
            print('          --new bias {} {} {}'.format(new_b[layer].shape, np.max(new_b[layer]) , np.min(new_b[layer])))

        # Make new net and save model
        self.net = caffe.Net(outproto, self.model, caffe.TEST)
        for layer in new_w:
            self.net.params[layer + self.postfix][0].data[...] = new_w[layer]
            self.net.params[layer + self.postfix][1].data[...] = new_b[layer]
        self.net.save(outmodel)
        self.net = caffe.Net(outproto, outmodel, caffe.TEST)

    def test(self, height, width):
        np.random.seed()
        rand_image = np.random.rand(1, 3, height, width) * 255
        rand_image = (rand_image - 127.5) / 128.0
        self.net.blobs['data'].data[...] = rand_image
        self.orig_net.blobs['data'].data[...] = rand_image

        out = self.net.forward()
        orig_out = self.orig_net.forward()

        for key in orig_out:
            print('=========Output {}'.format(key))
            print('Orig')
            print(orig_out[key].reshape([-1]))
            print('Folded')
            print(out[key].reshape([-1]))
            print('')


if __name__ == "__main__":
    args = parse_args()

    fusioner = ConvBnFusioner(args.deploy, args.model)
    fusioner.eliminate_bn()

    if args.test:
        if args.height == -1 or args.width == -1:
            raise ValueError('If running test, --height and --width need to be set.')
        fusioner.test(args.height, args.width)

