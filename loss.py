'''
loss implementation of lambdaRank Factorization Machines (LambdaFM) as described in:

@author:
Fan Li
@references:
https://github.com/tensorflow/ranking
'''

import abc

import tensorflow as tf

import utils


_EPSILON = 1e-10


class RankingLossKey(object):
  """Ranking loss key strings."""
  # Names for the ranking based loss functions.
  PAIRWISE_LOGISTIC_LOSS = 'pairwise_logistic_loss'


def _compute_ranks(logits, is_valid):
  """Computes ranks by sorting valid logits.

  Args:
    logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
    is_valid: A `Tensor` of the same shape as `logits` representing validity of
      each entry.

  Returns:
    The `ranks` Tensor.
  """
  _check_tensor_shapes([logits, is_valid])
  # Only sort entries with is_valid = True.
  scores = tf.where(
      is_valid, logits, -1e-6 * tf.ones_like(logits) +
      tf.reduce_min(input_tensor=logits, axis=1, keepdims=True))
  return utils.sorted_ranks(scores)


def _pairwise_comparison(labels, logits):
  r"""Returns pairwise comparison `Tensor`s.

  Given a list of n items, the labels of graded relevance l_i and the logits
  s_i, we form n^2 pairs. For each pair, we have the following:

                        /
                        | 1   if l_i > l_j for valid l_i and l_j.
  * `pairwise_labels` = |
                        | 0   otherwise
                        \
  * `pairwise_logits` = s_i - s_j

  Args:
    labels: A `Tensor` with shape [batch_size, list_size].
    logits: A `Tensor` with shape [batch_size, list_size].

  Returns:
    A tuple of (pairwise_labels, pairwise_logits) with each having the shape
    [batch_size, list_size, list_size].
  """
  # Compute the difference for all pairs in a list. The output is a Tensor with
  # shape [batch_size, list_size, list_size] where the entry [-1, i, j] stores
  # the information for pair (i, j).
  pairwise_label_diff = _apply_pairwise_op(tf.subtract, labels)
  pairwise_logits = _apply_pairwise_op(tf.subtract, logits)
  # Only keep the case when l_i > l_j.
  pairwise_labels = tf.cast(
      tf.greater(pairwise_label_diff, 0), dtype=tf.float32)
  is_valid = utils.is_label_valid(labels)
  valid_pair = _apply_pairwise_op(tf.logical_and, is_valid)
  pairwise_labels *= tf.cast(valid_pair, dtype=tf.float32)
  return pairwise_labels, pairwise_logits


def _check_tensor_shapes(tensors):
  """Checks the tensor shapes to be compatible."""
  for tensor in tensors:
    tensor = tf.convert_to_tensor(value=tensor)
    tensor.get_shape().assert_has_rank(2)
    tensor.get_shape().assert_is_compatible_with(
        tf.convert_to_tensor(value=tensors[0]).get_shape())


def _apply_pairwise_op(op, tensor):
  """Applies the op on tensor in the pairwise manner."""
  _check_tensor_shapes([tensor])
  return op(tf.expand_dims(tensor, 2), tf.expand_dims(tensor, 1))


def _get_valid_pairs_and_clean_labels(labels):
  """Returns a boolean Tensor for valid pairs and cleaned labels."""
  labels = tf.convert_to_tensor(value=labels)
  labels.get_shape().assert_has_rank(2)
  # 筛选大于等于0的值是可用的
  is_valid = utils.is_label_valid(labels)
  # pair对两元素必须都>=0
  valid_pairs = _apply_pairwise_op(tf.logical_and, is_valid)
  # > 0 保留原有label值 否则为0
  labels = tf.where(is_valid, labels, tf.zeros_like(labels))
  return valid_pairs, labels


class _LambdaWeight(object):
  """Interface for ranking metric optimization.

  This class wraps weights used in the LambdaLoss framework for ranking metric
  optimization (https://ai.google/research/pubs/pub47258). Such an interface is
  to be instantiated by concrete lambda weight models. The instance is used
  together with standard loss such as logistic loss and softmax loss.
  """

  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def pair_weights(self, labels, ranks):
    """Returns the weight adjustment `Tensor` for example pairs.

    Args:
      labels: A dense `Tensor` of labels with shape [batch_size, list_size].
      ranks: A dense `Tensor` of ranks with the same shape as `labels` that are
        sorted by logits.

    Returns:
      A `Tensor` that can weight example pairs.
    """
    raise NotImplementedError('Calling an abstract method.')

  def individual_weights(self, labels, ranks):
    """Returns the weight `Tensor` for individual examples.

    Args:
      labels: A dense `Tensor` of labels with shape [batch_size, list_size].
      ranks: A dense `Tensor` of ranks with the same shape as `labels` that are
        sorted by logits.

    Returns:
      A `Tensor` that can weight individual examples.
    """
    del ranks
    return labels


class DCGLambdaWeight(_LambdaWeight):
  """LambdaWeight for Discounted Cumulative Gain metric."""

  def __init__(self,
               topn=None,
               gain_fn=lambda label: label,
               rank_discount_fn=lambda rank: 1. / rank,
               normalized=False,
               smooth_fraction=0.):
    """Constructor.

    Ranks are 1-based, not 0-based. Given rank i and j, there are two types of
    pair weights:
      u = |rank_discount_fn(|i-j|) - rank_discount_fn(|i-j| + 1)|
      v = |rank_discount_fn(i) - rank_discount_fn(j)|
    where u is the newly introduced one in LambdaLoss paper
    (https://ai.google/research/pubs/pub47258) and v is the original one in the
    LambdaMART paper "From RankNet to LambdaRank to LambdaMART: An Overview".
    The final pair weight contribution of ranks is
      (1-smooth_fraction) * u + smooth_fraction * v.

    Args:
      topn: (int) The topn for the DCG metric.
      gain_fn: (function) Transforms labels.
      rank_discount_fn: (function) The rank discount function.
      normalized: (bool) If True, normalize weight by the max DCG.
      smooth_fraction: (float) parameter to control the contribution from
        LambdaMART.
    """
    self._topn = topn
    self._gain_fn = gain_fn
    self._rank_discount_fn = rank_discount_fn
    self._normalized = normalized
    assert 0. <= smooth_fraction and smooth_fraction <= 1., (
        'smooth_fraction %s should be in range [0, 1].' % smooth_fraction)
    self._smooth_fraction = smooth_fraction

  def pair_weights(self, labels, ranks):
    # label为本身的labels， ranks为logits排序后
    """See `_LambdaWeight`."""
    with tf.name_scope(name='dcg_lambda_weight'):
      _check_tensor_shapes([labels, ranks])
      # 可用的pair对，处理小于0 = 0后的label值
      valid_pair, labels = _get_valid_pairs_and_clean_labels(labels)
      gain = self._gain_fn(labels)
      if self._normalized:
        # batch * 1 => batch * fea_num
        gain *= utils.inverse_max_dcg(
            labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
      pair_gain = _apply_pairwise_op(tf.subtract, gain)
      pair_gain *= tf.cast(valid_pair, dtype=tf.float32)

      list_size = tf.shape(input=labels)[1]
      topn = self._topn or list_size

      def _discount_for_relative_rank_diff():
        """Rank-based discount in the LambdaLoss paper."""
        # The LambdaLoss is not well defined when topn is active and topn <
        # list_size. We cap the rank of examples to topn + 1 so that the rank
        # differene is capped to topn. This is just a convenient upperbound
        # when topn is active. We need to revisit this.
        # 排序大于topn的全部置为topn+1, 否则保持原样
        capped_rank = tf.where(
            tf.greater(ranks, topn),
            tf.ones_like(ranks) * (topn + 1), ranks)
        # 这样的话相同的值被置为0
        rank_diff = tf.cast(
            tf.abs(_apply_pairwise_op(tf.subtract, capped_rank)),
            dtype=tf.float32)
        pair_discount = tf.where(
            tf.greater(rank_diff, 0),
            tf.abs(
                self._rank_discount_fn(rank_diff) -
                self._rank_discount_fn(rank_diff + 1)),
            tf.zeros_like(rank_diff))

        # |(1/(rank_i - rank_j)) - ((1/(rank_i - rank_j + 1))
        return pair_discount

      def _discount_for_absolute_rank():
        """Standard discount in the LambdaMART paper."""
        # When the rank discount is (1 / rank) for example, the discount is
        # |1 / r_i - 1 / r_j|. When i or j > topn, the discount becomes 0.
        # rank > topn的置为0，否则为1/rank
        rank_discount = tf.where(
            tf.greater(ranks, topn),
            tf.zeros_like(tf.cast(ranks, dtype=tf.float32)),
            self._rank_discount_fn(tf.cast(ranks, dtype=tf.float32)))
        # |1/rank_i - 1/rank_j|
        pair_discount = tf.abs(_apply_pairwise_op(tf.subtract, rank_discount))
        return pair_discount

      u = _discount_for_relative_rank_diff()
      v = _discount_for_absolute_rank()
      pair_discount = (1. -
                       self._smooth_fraction) * u + self._smooth_fraction * v
      pair_weight = tf.abs(pair_gain) * pair_discount
      if self._topn is None:
        return pair_weight
      pair_mask = _apply_pairwise_op(tf.logical_or,
                                     tf.less_equal(ranks, self._topn))
      return pair_weight * tf.cast(pair_mask, dtype=tf.float32)

  def individual_weights(self, labels, ranks):
    """See `_LambdaWeight`."""
    with tf.name_scope(name='dcg_lambda_weight'):
      _check_tensor_shapes([labels, ranks])
      labels = tf.convert_to_tensor(value=labels)
      labels = tf.where(
          utils.is_label_valid(labels), labels, tf.zeros_like(labels))
      gain = self._gain_fn(labels)
      if self._normalized:
        gain *= utils.inverse_max_dcg(
            labels,
            gain_fn=self._gain_fn,
            rank_discount_fn=self._rank_discount_fn,
            topn=self._topn)
      rank_discount = self._rank_discount_fn(tf.cast(ranks, dtype=tf.float32))
      return gain * rank_discount


def create_ndcg_lambda_weight(topn=None, smooth_fraction=0.):
  """Creates _LambdaWeight for NDCG metric."""
  return DCGLambdaWeight(
      topn,
      gain_fn=lambda labels: tf.pow(2.0, labels) - 1.,
      rank_discount_fn=lambda rank: 1. / tf.math.log1p(rank),
      normalized=True,
      smooth_fraction=smooth_fraction)


class _RankingLoss(object):
  """Interface for ranking loss."""

  __metaclass__ = abc.ABCMeta

  @abc.abstractproperty
  def name(self):
    """The loss name."""
    raise NotImplementedError('Calling an abstract method.')

  @abc.abstractmethod
  def compute_unreduced_loss(self, labels, logits):
    """Computes the unreduced loss.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.

    Returns:
      A tuple(losses, loss_weights) that have the same shape.
    """
    raise NotImplementedError('Calling an abstract method.')

  def normalize_weights(self, labels, weights):
    """Normalizes weights needed for tf.estimator (not tf.keras).

    This is needed for `tf.estimator` given that the reduction may be
    `SUM_OVER_NONZERO_WEIGHTS`. This function is not needed after we migrate
    from the deprecated reduction to `SUM` or `SUM_OVER_BATCH_SIZE`.

    Args:
      labels: A `Tensor` of shape [batch_size, list_size] representing graded
        relevance.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.

    Returns:
      The normalized weights.
    """
    del labels
    return 1.0 if weights is None else weights

  def compute(self, labels, logits, weights=None, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):
    """Computes the reduced loss for tf.estimator (not tf.keras).

    Note that this function is not compatible with keras.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.
      reduction: One of `tf.losses.Reduction` except `NONE`. Describes how to
        reduce training loss over batch.

    Returns:
      Reduced loss for training and eval.
    """
    losses, loss_weights = self.compute_unreduced_loss(labels, logits)
    weights = tf.multiply(self.normalize_weights(labels, weights), loss_weights)
    return tf.losses.compute_weighted_loss(
        losses, weights, reduction=reduction)

  def eval_metric(self, labels, logits, weights):
    """Computes the eval metric for the loss in tf.estimator (not tf.keras).

    Note that this function is not compatible with keras.

    Args:
      labels: A `Tensor` of the same shape as `logits` representing graded
        relevance.
      logits: A `Tensor` with shape [batch_size, list_size]. Each value is the
        ranking score of the corresponding item.
      weights: A scalar, a `Tensor` with shape [batch_size, 1] for list-wise
        weights, or a `Tensor` with shape [batch_size, list_size] for item-wise
        weights.

    Returns:
      A metric op.
    """
    losses, loss_weights = self.compute_unreduced_loss(labels, logits)
    weights = tf.multiply(self.normalize_weights(labels, weights), loss_weights)
    return tf.metrics.mean(losses, weights)


class _PairwiseLoss(_RankingLoss):
  """Interface for pairwise ranking loss."""

  __metaclass__ = abc.ABCMeta

  def __init__(self, name, lambda_weight=None, params=None):
    """Constructor.

    Args:
      name: A string used as the name for this loss.
      lambda_weight: A `_LambdaWeight` object.
      params: A dict for params used in loss computation.
    """
    self._name = name
    self._lambda_weight = lambda_weight
    self._params = params or {}

  @property
  def name(self):
    """The loss name."""
    return self._name

  @abc.abstractmethod
  def _pairwise_loss(self, pairwise_logits):
    """The loss of pairwise logits with l_i > l_j."""
    raise NotImplementedError('Calling an abstract method.')

  def compute_unreduced_loss(self, labels, logits):
    """See `_RankingLoss`."""
    # 检测样本是否可用，过滤填充样本 < 0的样本False
    is_valid = utils.is_label_valid(labels)
    # 对于true保留logits，false 为极小浮点数
    ranks = _compute_ranks(logits, is_valid)
    pairwise_labels, pairwise_logits = _pairwise_comparison(labels, logits)
    pairwise_weights = pairwise_labels
    if self._lambda_weight is not None:
      pairwise_weights *= self._lambda_weight.pair_weights(labels, ranks)
      # For LambdaLoss with relative rank difference, the scale of loss becomes
      # much smaller when applying LambdaWeight. This affects the training can
      # make the optimal learning rate become much larger. We use a heuristic to
      # scale it up to the same magnitude as standard pairwise loss.
      pairwise_weights *= tf.cast(tf.shape(input=labels)[1], dtype=tf.float32)

    # 不进行梯度更新
    pairwise_weights = tf.stop_gradient(
        pairwise_weights, name='weights_stop_gradient')
    return self._pairwise_loss(pairwise_logits), pairwise_weights

  def normalize_weights(self, labels, weights):
    """See _RankingLoss."""
    # The `weights` is item-wise and is applied non-symmetrically to update
    # pairwise_weights as
    #   pairwise_weights(i, j) = w_i * pairwise_weights(i, j).
    # This effectively applies to all pairs with l_i > l_j. Note that it is
    # actually symmetric when `weights` are constant per list, i.e., listwise
    # weights.
    if weights is None:
      weights = 1.
    weights = tf.where(
        utils.is_label_valid(labels),
        tf.ones_like(labels) * weights, tf.zeros_like(labels))
    return tf.expand_dims(weights, axis=2)


class PairwiseLogisticLoss(_PairwiseLoss):
  """Implements pairwise logistic loss."""

  def _pairwise_loss(self, pairwise_logits):
    """See `_PairwiseLoss`."""
    # The following is the same as log(1 + exp(-pairwise_logits)).
    return tf.nn.relu(-pairwise_logits) + tf.math.log1p(
        tf.exp(-tf.abs(pairwise_logits)))
