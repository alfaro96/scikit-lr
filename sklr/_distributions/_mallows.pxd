from sklearn.utils._typedefs cimport float64_t, intp_t


cdef bint estimate_center(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    intp_t max_iter,
    intp_t[::1] center,
    intp_t* n_iter,
    float64_t[:, ::1] completed,
    float64_t[::1] scores,
    intp_t[:, ::1] work,
) noexcept nogil
