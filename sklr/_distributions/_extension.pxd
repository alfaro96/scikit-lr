from sklearn.utils._typedefs cimport float64_t, intp_t


cdef void complete_ranking(
    const float64_t[::1] y,
    const intp_t[::1] center,
    float64_t[::1] completed,
    intp_t[::1] ranked_at,
    intp_t[::1] gaps,
) noexcept nogil
