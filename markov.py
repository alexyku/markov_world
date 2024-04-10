"""Markov utils."""

import functools
import jax
import jax.numpy as jnp


def vmap_partial(f, **kwargs):
  return jax.vmap(functools.partial(f, **kwargs))


def sample_dirichlet_weights(rng, size: int, alpha):
  """Sample dirichlet weights."""
  return jax.random.dirichlet(rng, alpha=jnp.full(size, alpha))


def sample_dirichlet_matrix(rng, size, alpha):
  """Sample dirichlet matrix."""
  return jax.random.dirichlet(rng, alpha=jnp.full(size, alpha), shape=[size])


def sample_permutation_matrix(rng, size):
  """Sample permutation matrix."""
  return jnp.eye(size)[jax.random.permutation(rng, size)]


def sample_convex_combination_of_permutation_matrices(
    rng, size, num_permutations, alpha
):
  """Sample convex combination of permutation matrices."""
  perm_rng, combo_rng = jax.random.split(rng)
  perm_matrices = vmap_partial(sample_permutation_matrix, size=size)(
      jax.random.split(perm_rng, num_permutations)
  )
  combo_weights = sample_dirichlet_weights(combo_rng, num_permutations, alpha)
  return jnp.sum(combo_weights[:, None, None] * perm_matrices, axis=0)


def compute_cartesian_product_of_matrices(*matrices):
  """Compute cartesian product of matrices."""
  outer, inner, *recurse = matrices
  product = jnp.repeat(
      jnp.repeat(outer, inner.shape[0], axis=0), inner.shape[0], axis=1
  ) * jnp.tile(inner, [outer.shape[0], outer.shape[0]])
  return (
      compute_cartesian_product_of_matrices(product, *recurse)
      if recurse
      else product
  )


def compute_hierarchical_matrix(outer, inners):
  """Compute hierarchical matrix."""
  assert outer.shape[0] == inners.shape[0]
  assert outer.ndim == 2  # [m, m]
  assert inners.ndim == 3  # [m, n, n]
  return jnp.repeat(
      jnp.repeat(outer, inners.shape[1], axis=0), inners.shape[1], axis=1
  ) * jnp.tile(jnp.concatenate(inners, axis=0), [1, outer.shape[0]])


def compute_steady_state(matrix):
  """Compute steady state."""
  eigenvalues, eigenvectors = jnp.linalg.eig(matrix.T)
  idx = jnp.argmin(jnp.abs(eigenvalues - 1))
  steady_state = jnp.real(eigenvectors[:, idx])
  steady_state /= jnp.sum(steady_state)
  is_aperiodic = ~jnp.any(
      jnp.isclose(jnp.abs(eigenvalues), 1) & jnp.isreal(eigenvalues)
  )
  is_irreducible = jnp.all(jnp.linalg.matrix_power(matrix, matrix.shape[0]) > 0)
  return steady_state, is_aperiodic, is_irreducible


def compute_noise_ceiling(matrix):
  """Compute noise ceiling."""
  steady_state, _, _ = compute_steady_state(matrix)  # Expected occupation.
  mode = jnp.max(matrix, axis=-1)  # Expected accuracy from predicting the mode.
  return jnp.sum(steady_state * mode)


def random_walk(rng, state, matrix, num_steps):
  """Random walk.

  Args:
    rng: Random key.
    state: Integer between [0, num_states). Initial state.
    matrix: Float array with shape [num_states, num_states] and values between
      [0, 1]. Transition matrix.
    num_steps: Integer. Number of steps.

  Returns:
    walk: Integer array with shape [num_steps] and values between [0,
    num_states).
  """

  def body_fun(i, val):
    rng, state, walk = val
    next_rng, state_rng = jax.random.split(rng)
    state_logits = jnp.log(matrix[state])
    next_state = jax.random.categorical(state_rng, logits=state_logits)
    next_walk = jax.lax.dynamic_update_index_in_dim(walk, state, i, axis=0)
    return next_rng, next_state, next_walk

  walk = jnp.zeros([num_steps], dtype=jnp.int32)
  _, _, walk = jax.lax.fori_loop(0, num_steps, body_fun, (rng, state, walk))
  return walk


def random_walk_with_init(rng, init, matrix, num_steps):
  """Random walk with initial state distribution.

  Args:
    rng: Random key.
    init: Float array with shape [num_states] and values between [0, 1].
      Distribution over initial states.
    matrix: Float array with shape [num_states, num_states] and values between
      [0, 1]. Transition matrix.
    num_steps: Integer. Number of steps.

  Returns:
    walk: Integer array with shape [num_steps] and values between [0,
    num_states).
  """
  init_rng, random_walk_rng = jax.random.split(rng)
  init_state = jax.random.categorical(init_rng, logits=jnp.log(init))
  return random_walk(random_walk_rng, init_state, matrix, num_steps)


def sample_categorical_hidden_markov_model(
    rng, init, matrix, categorical, num_steps
):
  """Sample categorical hidden Markov model (HMM).

  Args:
    rng: Random key.
    init: Float array with shape [num_states] and values between [0, 1].
      Distribution over initial states.
    matrix: Float array with shape [num_states, num_states] and values between
      [0, 1]. Transition matrix.
    categorical: Float array with shape [num_states, num_vocab] and values
      between [0, 1]. Emission distribution.
    num_steps: Integer. Number of steps.

  Returns:
    states: Integer array with shape [num_states] and values between [0,
    num_states). Hidden states.
    emissions: Integer array with shape [num_states] and values between
    [0, num_vocab). Emissions.
  """
  random_walk_rng, emission_rng = jax.random.split(rng)
  states = random_walk_with_init(random_walk_rng, init, matrix, num_steps)
  emission_logits = jnp.log(categorical[states])
  emissions = jax.random.categorical(emission_rng, logits=emission_logits)
  return states, emissions
