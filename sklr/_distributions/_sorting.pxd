from sklearn.utils._typedefs cimport float64_t, intp_t


cdef void rank_by_scores(
    const float64_t[::1] scores,
    intp_t[::1] ranking,
) noexcept nogil
