import logging

import ffmpeg
import numpy as np
import tensorflow as tf

from lib.frvsr import generator_F, fnet
from lib.ops import deprocess, upscale_four

SCALE_X = 4
SCALE_Y = 4

logger = logging.getLogger(__name__)

class Upscaler:
    def _init_flags(self):
        Flags = tf.app.flags

        Flags.DEFINE_integer("rand_seed", 1, "random seed")

        # Directories
        Flags.DEFINE_string(
            "input_dir_LR",
            None,
            "The directory of the input resolution input data, " "for inference mode",
        )
        Flags.DEFINE_integer(
            "input_dir_len", -1, "length of the input for inference mode, -1 means all"
        )
        Flags.DEFINE_string(
            "input_dir_HR",
            None,
            "The directory of the input resolution input data, " "for inference mode",
        )
        Flags.DEFINE_string("mode", "inference", "train, or inference")
        Flags.DEFINE_string(
            "output_dir", None, "The output directory of the checkpoint"
        )
        Flags.DEFINE_string(
            "output_pre", "", "The name of the subfolder for the images"
        )
        Flags.DEFINE_string("output_name", "output", "The pre name of the outputs")
        Flags.DEFINE_string(
            "output_ext", "jpg", "The format of the output when evaluating"
        )
        Flags.DEFINE_string("summary_dir", None, "The dirctory to output the summary")

        # Models
        Flags.DEFINE_string(
            "checkpoint",
            "./model/TecoGAN",
            "If provided, the weight will be restored from the " "provided checkpoint",
        )
        Flags.DEFINE_integer(
            "num_resblock", 16, "How many residual blocks are there in the generator"
        )
        # Models for training
        Flags.DEFINE_boolean(
            "pre_trained_model",
            False,
            "If True, the weight of generator will be loaded as "
            "an initial point"
            "If False, continue the training",
        )
        Flags.DEFINE_string("vgg_ckpt", None, "path to checkpoint file for the vgg19")

        # Machine resources
        Flags.DEFINE_string("cudaID", "0", "CUDA devices")
        Flags.DEFINE_integer(
            "queue_thread",
            6,
            "The threads of the queue (More threads can speedup "
            "the training process.",
        )
        Flags.DEFINE_integer(
            "name_video_queue_capacity",
            512,
            "The capacity of the filename queue (suggest large "
            "to ensure"
            "enough random shuffle.",
        )
        Flags.DEFINE_integer(
            "video_queue_capacity",
            256,
            "The capacity of the video queue (suggest large to "
            "ensure"
            "enough random shuffle",
        )
        Flags.DEFINE_integer("video_queue_batch", 2, "shuffle_batch queue capacity")

        # Training details
        # The data preparing operation
        Flags.DEFINE_integer("RNN_N", 10, "The number of the rnn recurrent length")
        Flags.DEFINE_integer("batch_size", 4, "Batch size of the input batch")
        Flags.DEFINE_boolean(
            "flip", True, "Whether random flip data augmentation is applied"
        )
        Flags.DEFINE_boolean("random_crop", True, "Whether perform the random crop")
        Flags.DEFINE_boolean(
            "movingFirstFrame",
            True,
            "Whether use constant moving first frame randomly.",
        )
        Flags.DEFINE_integer("crop_size", 32, "The crop size of the training image")
        # Training data settings
        Flags.DEFINE_string(
            "input_video_dir", "", "The directory of the video input data, for training"
        )
        Flags.DEFINE_string(
            "input_video_pre",
            "scene",
            "The pre of the directory of the video input data",
        )
        Flags.DEFINE_integer(
            "str_dir", 1000, "The starting index of the video directory"
        )
        Flags.DEFINE_integer("end_dir", 2000, "The ending index of the video directory")
        Flags.DEFINE_integer(
            "end_dir_val",
            2050,
            "The ending index for validation of the video " "directory",
        )
        Flags.DEFINE_integer("max_frm", 119, "The ending index of the video directory")
        # The loss parameters
        Flags.DEFINE_float(
            "vgg_scaling",
            -0.002,
            "The scaling factor for the VGG perceptual loss, "
            "disable with negative value",
        )
        Flags.DEFINE_float("warp_scaling", 1.0, "The scaling factor for the warp")
        Flags.DEFINE_boolean("pingpang", False, "use bi-directional recurrent or not")
        Flags.DEFINE_float(
            "pp_scaling",
            1.0,
            "factor of pingpang term, only works when pingpang is " "True",
        )
        # Training parameters
        Flags.DEFINE_float("EPS", 1e-12, "The eps added to prevent nan")
        Flags.DEFINE_float("learning_rate", 0.0001, "The learning rate for the network")
        Flags.DEFINE_integer(
            "decay_step", 500000, "The steps needed to decay the learning rate"
        )
        Flags.DEFINE_float("decay_rate", 0.5, "The decay rate of each decay step")
        Flags.DEFINE_boolean(
            "stair",
            False,
            "Whether perform staircase decay. True => decay in " "discrete interval.",
        )
        Flags.DEFINE_float("beta", 0.9, "The beta1 parameter for the Adam optimizer")
        Flags.DEFINE_float("adameps", 1e-8, "The eps parameter for the Adam optimizer")
        Flags.DEFINE_integer("max_epoch", None, "The max epoch for the training")
        Flags.DEFINE_integer("max_iter", 1000000, "The max iteration of the training")
        Flags.DEFINE_integer(
            "display_freq", 20, "The diplay frequency of the training process"
        )
        Flags.DEFINE_integer("summary_freq", 100, "The frequency of writing summary")
        Flags.DEFINE_integer("save_freq", 10000, "The frequency of saving images")
        # Dst parameters
        Flags.DEFINE_float(
            "ratio", 0.01, "The ratio between content loss and adversarial loss"
        )
        Flags.DEFINE_boolean(
            "Dt_mergeDs", True, "Whether only use a merged Discriminator."
        )
        Flags.DEFINE_float(
            "Dt_ratio_0", 1.0, "The starting ratio for the temporal adversarial loss"
        )
        Flags.DEFINE_float(
            "Dt_ratio_add",
            0.0,
            "The increasing ratio for the temporal adversarial loss",
        )
        Flags.DEFINE_float(
            "Dt_ratio_max", 1.0, "The max ratio for the temporal adversarial loss"
        )
        Flags.DEFINE_float("Dbalance", 0.4, "An adaptive balancing for Discriminators")
        Flags.DEFINE_float(
            "crop_dt", 0.75, "factor of dt crop"
        )  # dt input size = crop_size*crop_dt
        Flags.DEFINE_boolean("D_LAYERLOSS", True, "Whether use layer loss from D")

        FLAGS = Flags.FLAGS

        return FLAGS

    def __init__(self, *shape):
        FLAGS = self._init_flags()  # TODO: do cleanup before loading new one
        input_shape = [1] + list(shape) + [3]
        output_shape = [1, input_shape[1] * 4, input_shape[2] * 4, 3]
        oh = input_shape[1] % 8
        ow = input_shape[2] % 8
        paddings = tf.constant([[0, 0], [0, oh], [0, ow], [0, 0]])
        print("input shape:", input_shape)
        print("output shape:", output_shape)

        self.inputs_raw = tf.placeholder(
            tf.float32, shape=input_shape, name="inputs_raw"
        )

        pre_inputs = tf.Variable(
            tf.zeros(input_shape), trainable=False, name="pre_inputs"
        )
        pre_gen = tf.Variable(tf.zeros(output_shape), trainable=False, name="pre_gen")
        pre_warp = tf.Variable(tf.zeros(output_shape), trainable=False, name="pre_warp")

        transpose_pre = tf.space_to_depth(
            pre_warp, 4
        )  # TODO: hardcoded scale coefficient
        inputs_all = tf.concat((self.inputs_raw, transpose_pre), axis=-1)

        with tf.variable_scope("generator"):
            gen_output = generator_F(inputs_all, 3, reuse=False, FLAGS=FLAGS)
            # Deprocess the images outputed from the model, and assign things
            # for next frame
            with tf.control_dependencies([tf.assign(pre_inputs, self.inputs_raw)]):
                self.outputs = tf.assign(pre_gen, deprocess(gen_output))

        inputs_frames = tf.concat((pre_inputs, self.inputs_raw), axis=-1)
        with tf.variable_scope("fnet"):
            gen_flow_lr = fnet(inputs_frames, reuse=False)
            gen_flow_lr = tf.pad(gen_flow_lr, paddings, "SYMMETRIC")
            gen_flow = upscale_four(gen_flow_lr * SCALE_X)
            gen_flow.set_shape(output_shape[:-1] + [2])
        pre_warp_hi = tf.contrib.image.dense_image_warp(pre_gen, gen_flow)
        self.before_ops = tf.assign(pre_warp, pre_warp_hi)

        logger.info("Finish building the network")

        # In inference time, we only need to restore the weight of the generator
        var_list = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope="generator")
        var_list = var_list + tf.get_collection(
            tf.GraphKeys.MODEL_VARIABLES, scope="fnet"
        )

        weight_initiallizer = tf.train.Saver(var_list)

        # Define the initialization operation
        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        # Load the pretrained model
        self.session.run(init_op)
        self.session.run(local_init_op)

        logger.info("Loading weights from ckpt model")
        weight_initiallizer.restore(self.session, FLAGS.checkpoint)

    def process_frame(self, frame):
        input_im = np.expand_dims(frame.astype(np.float32), axis=0)
        feed_dict = {self.inputs_raw: input_im}

        self.session.run(
            self.before_ops, feed_dict=feed_dict
        )  # Need to run once more time before job start
        output_frame = self.session.run(self.outputs, feed_dict=feed_dict)

        return output_frame


def get_video_size(filename):
    logger.info("Getting video size for {!r}".format(filename))
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width = int(video_info["width"])
    height = int(video_info["height"])
    return width, height


def read_frame(process1, width, height):
    logger.debug("Reading frame")

    # Note: RGB24 == 3 bytes per pixel.
    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
    return frame


def write_frame(process2, frame):
    logger.debug("Writing frame")
    process2.stdin.write(frame.astype(np.uint8).tobytes())

