"""GINC dataset."""

import functools
import jax
import jax.numpy as jnp
from . import markov


def jit_partial(f, **kwargs):
  return jax.jit(functools.partial(f, **kwargs))


def vmap_partial(f, **kwargs):
  return jax.vmap(functools.partial(f, **kwargs))


def sample_concept_matrix(
    rng, num_entities, num_properties, num_permutations, alpha, identity_prior
):
  """Sample GINC concept matrix.

  Args:
    rng: Random key.
    num_entities: Integer. Number of entities.
    num_properties: Integer. Number of properties.
    num_permutations: Integer. Number of permutations sampling for entity and
      property matrices.
    alpha: Float. Dirichlet alpha for sampling entity and property matrices.
    identity_prior: Float. Identity prior for entity matrix.

  Returns:
    matrix: Float array with shape [num_entities * num_properties, num_entities
    * num_properties] and values between [0, 1]. Transition matrix between
    entity-property states.
  """
  entity_rng, property_rng = jax.random.split(rng)
  entity_matrix = markov.sample_convex_combination_of_permutation_matrices(
      rng=entity_rng,
      size=num_entities,
      num_permutations=num_permutations,
      alpha=alpha,
  )
  entity_matrix = (
      identity_prior * jnp.eye(num_entities)
      + (1 - identity_prior) * entity_matrix
  )
  property_matrix = markov.sample_convex_combination_of_permutation_matrices(
      rng=property_rng,
      size=num_properties,
      num_permutations=num_permutations,
      alpha=alpha,
  )
  return markov.compute_cartesian_product_of_matrices(
      entity_matrix, property_matrix
  )


class GINC:
  """GINC dataset."""

  def __init__(
      self,
      rng,
      num_concepts=5,
      num_permutations=100,
      num_entities=10,
      num_properties=10,
      num_vocab=50,
      alpha=0.1,
      identity_prior=0.9,
      emission_mode='default',
  ):
    """GINC dataset constructor.

    Args:
      rng: Random key.
      num_concepts: Integer. Number of concepts.
      num_permutations: Integer. Number of permutations sampling for entity and
        property matrices.
      num_entities: Integer. Number of entities.
      num_properties: Integer. Number of properties.
      num_vocab: Integer. Number of vocabulary.
      alpha: Float. Dirichlet alpha for sampling entity and property matrices.
      identity_prior: Float. Identity prior for entity matrix.
      emission_mode: String. Emission mode: 'default', 'aliased'
    """
    init_rng, matrix_rng, emission_rng = jax.random.split(rng, 3)

    self.num_concepts = num_concepts
    self.num_permutations = num_permutations
    self.num_entities = num_entities
    self.num_properties = num_properties
    self.num_vocab = num_vocab
    self.alpha = alpha
    self.identity_prior = identity_prior

    # Initial distributions are unique between concepts.
    # concept_inits: [num_concepts, num_entities * num_properties]
    self.concept_inits = jax.random.dirichlet(
        init_rng,
        alpha=jnp.full(num_entities * num_properties, alpha),
        shape=[num_concepts],
    )

    # Transition matrices are unique between concepts.
    # concept_matrices: [num_concepts, num_entities * num_properties,
    # num_entities * num_properties]
    self.concept_matrices = vmap_partial(
        sample_concept_matrix,
        num_entities=num_entities,
        num_properties=num_properties,
        num_permutations=num_permutations,
        alpha=alpha,
        identity_prior=identity_prior,
    )(jax.random.split(matrix_rng, num_concepts))

    # Emission distribution is shared between concepts.
    # vocab_categorical: [num_entities * num_properties, num_vocab]
    if emission_mode == 'default':
      # Vocabulary lookup table mapping entity-property state to tokens.
      vocab_lut = jax.random.randint(
          emission_rng,
          minval=1,  # Let 0 be the delimiter token.
          maxval=num_vocab,
          shape=[num_entities * num_properties],
      )
    elif emission_mode == 'aliased':
      # In the aliased emission setting, the vocabulary is non-overlapping
      # between entities. This means that the entity is no longer latent, since
      # the token fully determines the entity.
      assert (num_vocab - 1) % num_entities == 0  # Don't waste tokens!
      sub_vocab = (num_vocab - 1) // num_entities
      vocab_lut = jax.random.randint(
          emission_rng,
          minval=0,
          maxval=sub_vocab,
          shape=[num_entities, num_properties],
      )
      # Shift vocabulary to be non-overlapping.
      # Add one to let 0 be the delimiter token.
      vocab_lut += jnp.arange(num_entities)[:, None] * sub_vocab + 1
      vocab_lut = jnp.reshape(vocab_lut, [num_entities * num_properties])
    else:
      raise ValueError(f'Unknown emission mode: {emission_mode}')
    self.vocab_categorical = jax.nn.one_hot(vocab_lut, num_vocab)

  def _sample_document(self, rng, document_length):
    """Sample a training document."""
    concept_rng, document_rng = jax.random.split(rng)
    concept = jax.random.choice(concept_rng, self.num_concepts)
    states, document = markov.sample_categorical_hidden_markov_model(
        document_rng,
        init=self.concept_inits[concept],
        matrix=self.concept_matrices[concept],
        categorical=self.vocab_categorical,
        num_steps=document_length,
    )
    entities, properties = jnp.divmod(states, self.num_properties)
    return concept, entities, properties, document

  def generate_documents(self, rng, document_length):
    """Generate training documents."""
    sample_fn = jit_partial(
        self._sample_document, document_length=document_length
    )
    while True:
      rng, document_rng = jax.random.split(rng)
      concept, entities, properties, document = sample_fn(document_rng)
      yield {
          'concept': concept,
          'entities': entities,
          'properties': properties,
          'document': document,
      }

  def _sample_example(self, rng, concept, init_prop, example_length):
    """Sample a few-shot learning example."""
    concept_rng, example_rng = jax.random.split(rng)
    entity = jax.random.choice(concept_rng, self.num_entities)
    init = jax.nn.one_hot(
        entity * self.num_properties + init_prop,
        self.num_entities * self.num_properties,
    )
    states, example = markov.sample_categorical_hidden_markov_model(
        example_rng,
        init=init,
        matrix=self.concept_matrices[concept],
        categorical=self.vocab_categorical,
        num_steps=example_length,
    )
    entities, properties = jnp.divmod(states, self.num_properties)
    return entities, properties, example

  def _sample_prompt(self, rng, num_examples, example_length):
    """Sample a few-shot learning prompt."""
    concept_rng, property_rng, prompt_rng = jax.random.split(rng, 3)
    concept = jax.random.choice(concept_rng, self.num_concepts)
    init_prop = jax.random.choice(property_rng, self.num_properties)
    entities, properties, examples = vmap_partial(
        self._sample_example,
        concept=concept,
        init_prop=init_prop,
        example_length=example_length,
    )(jax.random.split(prompt_rng, num_examples))
    # Append delimiter and then concatenate the examples.
    prompt = jnp.reshape(
        jnp.pad(examples, [[0, 0], [0, 1]], constant_values=0), [-1]
    )
    return concept, entities, properties, examples, prompt

  def generate_prompts(self, rng, num_examples, example_length):
    """Generate few-shot learning prompts."""
    sample_fn = jit_partial(
        self._sample_prompt,
        num_examples=num_examples,
        example_length=example_length,
    )
    while True:
      rng, prompt_rng = jax.random.split(rng)
      concept, entities, properties, examples, prompt = sample_fn(prompt_rng)
      yield {
          'concept': concept,
          'entities': entities,
          'properties': properties,
          'examples': examples,
          'prompt': prompt,
      }
