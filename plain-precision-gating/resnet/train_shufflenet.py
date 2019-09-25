import sys, os, re
import subprocess as sp
import timeit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', type=int, default=0, help="Activation bits")
parser.add_argument('--gpus', type=str, default='0', help="GPUs to use")
parser.add_argument('--t', type=int, default=0, help="Truncate activations to a number of bits")
parser.add_argument('--g', type=float, default=0., help="Trainable threshold target")
parser.add_argument('--bs', type=int, default=512, help="batch size")
parser.add_argument('--pact', action='store_true')
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--data_dir', type=str, default='/dataset/imagenet/', help="location of the dataset")


model_name = 'shufflenetv2-x0.5'

if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()

    print("abits={}, trunc_to_bits={}, gating_target={}, num_gpus={}".format(FLAGS.a, FLAGS.t, FLAGS.g, len(FLAGS.gpus.split(','))))

    file_dir  = os.path.dirname(os.path.realpath(__file__))
    parent_dir = os.path.abspath(os.path.join(file_dir, os.pardir))

    results_dir = 'model_trained/pg_shufflenet/'
    results_dir = os.path.join(parent_dir, results_dir)
    train_dir = os.path.join(results_dir, "shufflenetv2_a{}_t{}_g{}_pact={}".format(FLAGS.a, FLAGS.t, FLAGS.g, FLAGS.pact))
    if not os.path.exists(train_dir):
      os.makedirs(train_dir)

    train_out = os.path.join(train_dir, "train.stdout")
    train_err = os.path.join(train_dir, "train.stderr")

    args = ["--data_dir=%s" % FLAGS.data_dir,
            "--model_dir=%s" % train_dir,
            "--model=%s" % model_name,
            "--epochs_per_eval=2",
            "--train_epochs=120",
            "--batch_size=%d" % FLAGS.bs,
            "--abits=%d" % FLAGS.a,
            "--trunc_to_bits=%d" % FLAGS.t,
            "--gating_target=%f" % FLAGS.g,
            "--num_gpus=%d" % len(FLAGS.gpus.split(',')),
            "--which_gpus=%s" % FLAGS.gpus]
    if FLAGS.pact:
        args.append("--use_pact")
    if FLAGS.test_only:
        args.append("--test_only")

    with open(train_out, 'w') as to:
      with open(train_err, 'w') as te:
        sp.call(["python", file_dir+"/imagenet_shufflenet.py"] + args, stdout=to, stderr=te)
