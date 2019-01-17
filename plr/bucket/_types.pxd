# =============================================================================
# Imports
# =============================================================================

# PLR
from .._types cimport SIZE_t

# C++
from libcpp.vector cimport vector

# =============================================================================
# Types
# =============================================================================
ctypedef vector[SIZE_t]   BUCKET_t  # Type for a single bucket
ctypedef vector[BUCKET_t] BUCKETS_t # Type for buckets
    
# =============================================================================
# Enums
# =============================================================================

# Optimal Bucket Order Problem algorithms
ctypedef enum OBOP_ALGORITHM: BPA_ORIGINAL_SG,  \
                              BPA_ORIGINAL_MP,  \
                              BPA_ORIGINAL_MP2, \
                              BPA_LIA_SG,       \
                              BPA_LIA_MP,       \
                              BPA_LIA_MP2
