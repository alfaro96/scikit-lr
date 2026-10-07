from sklearn.utils._typedefs cimport float64_t


cdef void borda_scores(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    float64_t[::1] scores,
) noexcept nogil
