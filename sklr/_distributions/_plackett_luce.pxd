from sklearn.utils._typedefs cimport float64_t, intp_t


cdef bint estimate_parameters(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    float64_t tol,
    intp_t max_iter,
    float64_t[::1] parameters,
    intp_t* n_iter,
    intp_t[:, ::1] order,
    intp_t[::1] n_ranked,
    float64_t[:, ::1] work,
) noexcept nogil
