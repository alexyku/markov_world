"""PCSW dataset."""

import functools
import jax
import jax.numpy as jnp
from . import markov


def jit_partial(f, **kwargs):
  return jax.jit(functools.partial(f, **kwargs))


def vmap_partial(f, **kwargs):
  return jax.vmap(functools.partial(f, **kwargs))


class PCSW:
  """PCSW dataset."""

  def __init__(
      self,
      rng,
      num_worlds=5,
      num_contexts=5,
      num_hidden=100,
      num_vocab=50,
      num_permutations=100,
      alpha=0.1,
      identity_prior=0.9,
      emission_mode='hidden',
  ):
    """PCSW dataset constructor.

    Args:
      rng: Random key.
      num_worlds: Number of splits for training and evaluation.
      num_contexts: Number of contexts. Transition dynamics between contexts are
        different between worlds.
      num_hidden: Number of hidden states. Transition dynamics between hidden
        states are shared between worlds.
      num_vocab: Vocabulary size.
      num_permutations: Number of permutations sampling transition matrices.
      alpha: Dirichlet alpha for sampling entity and property matrices.
      identity_prior: Identity prior for entity matrix.
      emission_mode: Whether the emission distribution is conditioned on solely
        on 'hidden' or both 'hidden_and_context'. When emission mode is
        'hidden', the model is a true hierarchical hidden Markov model.
    """
    context_rng, state_rng, emission_rng = jax.random.split(rng, 3)

    self.num_worlds = num_worlds
    self.num_contexts = num_contexts
    self.num_hidden = num_hidden
    self.num_vocab = num_vocab
    self.alpha = alpha
    self.identity_prior = identity_prior

    # Uniform initial distributions.
    # init: [num_worlds, num_contexts * num_hidden]
    self.uniform_init = jnp.ones([num_contexts * num_hidden], dtype=jnp.float32)
    self.uniform_init /= self.uniform_init.sum()

    # Context dynamics are unqiue between world.
    # context_matrices: [num_worlds, num_contexts, num_contexts]
    context_matrices_without_prior = vmap_partial(
        markov.sample_convex_combination_of_permutation_matrices,
        size=num_contexts,
        num_permutations=num_permutations,
        alpha=alpha,
    )(jax.random.split(context_rng, num_worlds))
    self.context_matrices = jax.vmap(
        lambda context_matrix: identity_prior * jnp.eye(num_contexts)
        + (1 - identity_prior) * context_matrix
    )(context_matrices_without_prior)

    # Hidden state dynamics are shared between worlds.
    # hidden_state_matrices: [num_contexts, num_hidden, num_hidden]
    self.hidden_state_matrices = vmap_partial(
        markov.sample_convex_combination_of_permutation_matrices,
        size=num_hidden,
        num_permutations=num_permutations,
        alpha=alpha,
    )(jax.random.split(state_rng, num_contexts))

    # Construct hierarchical transition matrix for each world.
    # transition_matrices: [num_worlds, num_contexts * num_hidden,
    # num_contexts * num_hidden]
    self.world_matrices = vmap_partial(
        markov.compute_hierarchical_matrix, inners=self.hidden_state_matrices
    )(outer=self.context_matrices)

    # Emission distributions are shared between worlds.
    if emission_mode == 'hidden':
      # Emissions only depend on hidden states.
      vocab_lut = jnp.tile(
          jax.random.randint(
              emission_rng,
              minval=0,
              maxval=num_vocab,
              shape=[num_hidden],
          ),
          num_contexts,
      )
    elif emission_mode == 'hidden_and_context':
      # Emissions depend on hidden states and context.
      vocab_lut = jax.random.randint(
          emission_rng,
          minval=0,
          maxval=num_vocab,
          shape=[num_contexts * num_hidden],
      )
    else:
      raise ValueError(f'Unknown emission mode: {emission_mode}')
    # vocab_categorical: [num_contexts * num_hidden, num_vocab]
    self.vocab_categorical = jax.nn.one_hot(vocab_lut, num_vocab)

  def _sample_sequence(self, rng, world, sequence_length):
    """Sample a training sequences."""
    contextual_states, emissions = (
        markov.sample_categorical_hidden_markov_model(
            rng,
            init=self.uniform_init,
            matrix=self.world_matrices[world],
            categorical=self.vocab_categorical,
            num_steps=sequence_length,
        )
    )
    contexts, hidden_states = jnp.divmod(contextual_states, self.num_contexts)
    return contexts, hidden_states, emissions

  def generate_sequences(self, rng, world, sequence_length):
    """Generate training sequences."""
    sample_fn = jit_partial(
        self._sample_sequence, world=world, sequence_length=sequence_length
    )
    while True:
      rng, sequence_rng = jax.random.split(rng)
      contexts, hidden_states, emissions = sample_fn(sequence_rng)
      yield {
          'contexts': contexts,
          'hidden_states': hidden_states,
          'emissions': emissions,
      }
