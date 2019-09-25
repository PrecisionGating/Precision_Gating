import numpy as np
import tensorflow as tf
import itertools, six

def compute_loss(network_fn, inputs, labels, is_training,
                 weight_decay=1e-4,
                 gating_loss_factor=1e-4,
                 delta_T=0.):

  logits = network_fn(inputs, is_training)

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Calculate the gating loss
  gating_loss_list = [tf.nn.l2_loss(delta_T-v) for v in tf.trainable_variables()
                          if 'training_threshold' in v.name]
  if len(gating_loss_list) > 0:
    gating_loss = gating_loss_factor * tf.add_n(gating_loss_list)
  else:
    gating_loss = 0


  # Add weight decay to the loss. We exclude the batch norm variables because
  # doing so leads to a small improvement in accuracy.
  loss = cross_entropy + weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name]) + gating_loss

  tf.add_to_collection(tf.GraphKeys.LOSSES,
                       tf.identity(cross_entropy, name='cross_entropy'))
  tf.add_to_collection(tf.GraphKeys.LOSSES,
                       tf.identity(gating_loss, name='gating_loss'))

  return logits, loss


def sync_grads(tower_gradvars):
  ''' Take the grads computed by each tower and average them
      Inputs:
        tower_gradvars: some kind of list of (grad, var)
      Returns:
        gradvars: list of (grad, var)
  '''
  gradvars = []
  with tf.name_scope('gradient_averaging'):
    all_grads = {}
    for grad, var in itertools.chain(*tower_gradvars):
      if grad is not None:
        all_grads.setdefault(var, []).append(grad)
    for var, grads in six.iteritems(all_grads):
      # Average gradients on the same device as the variables they apply to
      with tf.device(var.device):
        if len(grads) == 1:
          avg_grad = grads[0]
        else:
          avg_grad = tf.multiply(tf.add_n(grads), 1./len(grads))
      gradvars.append((avg_grad, var))

  return gradvars

def build_train_or_eval_graph(network_fn, inputs, labels, optimizer, is_training,
                              weight_decay=1e-4, num_gpus=1, gating_loss_factor=1e-4,
                              delta_T=0.):
  ''' Given a network and inputs, builds the graph to compute the logits,
      loss, as well as the graph to train the network.

      Returns:
        logits
        loss
        train_op
  '''
  #------------#
  # Single GPU #
  #------------#
  if num_gpus == 1:
    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    logits, loss = compute_loss(
            network_fn, inputs, labels, is_training,
            weight_decay=weight_decay,
            gating_loss_factor=gating_loss_factor,
            delta_T=delta_T)

    if is_training:
      global_step = tf.train.get_or_create_global_step()

      # Batch norm requires update ops to be added as a dependency to the train_op
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)
    else:
      train_op = None

  #------------#
  # Multi-GPU  #
  #------------#
  else:
    inputs_split = tf.split(inputs, num_gpus, axis=0)
    labels_split = tf.split(labels, num_gpus, axis=0)
    tower_losses = []
    tower_gradvars = []
    eval_logits = []

    ''' Distribute logit and gradient computation onto each GPU '''
    for i in range(num_gpus):
      with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device('/gpu:%d' % i):

            # Calculate loss, which includes softmax cross entropy and L2 regularization.
            # RZ: we can pass model_params into compute_loss to filter variables for training
            model_params = tf.trainable_variables()
            logits, loss = compute_loss(network_fn, inputs_split[i], labels_split[i], is_training,
                                        weight_decay=weight_decay,
                                        gating_loss_factor=gating_loss_factor,
                                        delta_T=delta_T)

            grads = tf.gradients(loss, model_params)
            gradvars = zip(grads, model_params)

            #tf.get_variable_scope().reuse_variables()
            tower_losses.append(loss)
            tower_gradvars.append(gradvars)
            eval_logits.append(logits)

            # Batch norm requires update ops to be added as a dependency to the train_op
            # Only trigger batch norm on the 1st tower
            if i == 0:
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

    ''' Average loss and gradients '''
    with tf.device('/cpu:0'):
      loss = tf.reduce_mean(tower_losses, name='loss')
      tf.identity(loss, name='avg_loss')
      tf.summary.scalar('avg_loss', loss)

      if is_training:
        # Synchronization point for all towers
        gradvars = sync_grads(tower_gradvars)

        global_step = tf.train.get_or_create_global_step()

        # Batch norm requires update_ops to be added as a train_op dependency.
        with tf.control_dependencies(update_ops):
          train_op = [ optimizer.apply_gradients(gradvars, global_step) ]
          train_op = tf.group(*train_op)
      else:
        train_op = None

    ''' Collect logits '''
    logits = tf.concat(eval_logits, axis=0)

  return logits, loss, train_op
  
