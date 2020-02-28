# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


# =============================================================================
# Constants
# =============================================================================

# Map the rank type string identifier
# to the rank type integer identifier
RANK_TYPE_MAPPING = {
    "random": RANDOM,
    "top": TOP
}
