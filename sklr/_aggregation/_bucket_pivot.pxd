from sklearn.utils._typedefs cimport float64_t, intp_t


cdef void pair_order_matrix(
    const float64_t[:, ::1] precedes,
    const float64_t[:, ::1] ties,
    float64_t[:, ::1] matrix,
) noexcept nogil


cdef intp_t bucket_order(
    const float64_t[:, ::1] matrix,
    intp_t[::1] items,
    intp_t[::1] ranking,
    float64_t[:, ::1] work,
    intp_t[:, ::1] index_work,
) noexcept nogil


cdef intp_t aggregate(
    const float64_t[:, ::1] y,
    const float64_t[::1] sample_weight,
    intp_t[::1] ranking,
    float64_t[:, :, ::1] matrices,
    float64_t[:, ::1] work,
    intp_t[:, ::1] index_work,
) noexcept nogil
