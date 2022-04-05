import cupy as cp


def _unit_norm(signal, *, axis=-1, eps=1e-4, eps_style="plus", ord=None):
    """Unit normalization.
    Args:
        signal: STFT signal with shape (..., T, D).
        eps_style: in ['plus', 'max']
    Returns:
        Normalized STFT signal with same shape.
    >>> signal = cp.array([[1, 1], [1e-20, 1e-20], [0, 0]])
    >>> _unit_norm(signal, eps_style='plus')
    array([[7.07056785e-01, 7.07056785e-01],
           [1.00000000e-16, 1.00000000e-16],
           [0.00000000e+00, 0.00000000e+00]])
    >>> _unit_norm(signal, eps_style='max')
    array([[7.07106781e-01, 7.07106781e-01],
           [1.00000000e-16, 1.00000000e-16],
           [0.00000000e+00, 0.00000000e+00]])
    >>> _unit_norm(signal, eps_style='where')  # eps has no effect
    array([[0.70710678, 0.70710678],
           [0.70710678, 0.70710678],
           [0.        , 0.        ]])
    """
    norm = cp.linalg.norm(signal, ord=ord, axis=axis, keepdims=True)
    if eps_style == "plus":
        norm = norm + eps
    elif eps_style == "max":
        norm = cp.maximum(norm, eps)
    elif eps_style == "where":
        norm = cp.where(norm == 0, eps, norm)
    else:
        assert False, eps_style
    return signal / norm


def force_hermitian(matrix):
    """
    >>> A = cp.array([[1+2j, 3+5j], [7+11j, 13+17j]])
    >>> force_hermitian(A)
    array([[ 1.+0.j,  5.-3.j],
           [ 5.+3.j, 13.+0.j]])
    >>> force_hermitian(force_hermitian(A))
    array([[ 1.+0.j,  5.-3.j],
           [ 5.+3.j, 13.+0.j]])
    """
    return (matrix + cp.swapaxes(matrix.conj(), -1, -2)) / 2


def estimate_mixture_weight(
    affiliation,
    saliency=None,
    weight_constant_axis=-1,
):
    """
    Estimates the mixture weight of a mixture model.
    The simplest version (without saliency and prior):
        return cp.mean(affiliation, axis=weight_constant_axis, keepdims=True)
    Args:
        affiliation: Shape: (..., K, T)
        saliency: Shape: (..., K, T)
        weight_constant_axis: int
    Returns:
        mixture weight with the same shape as affiliation, except for the
        weight_constant_axis that is a singleton:
            e.g. for weight_constant_axis == -1: (..., K, 1)
        When the weight_constant_axis is -2 or the positive counterpart,
        then the returned shape is always (K, 1) and the value if 1/K.
    >>> affiliation = [[0.4, 1, 0.4], [0.6, 0, 0.6]]
    >>> estimate_mixture_weight(affiliation)
    array([[0.6],
           [0.4]])
    >>> estimate_mixture_weight(affiliation, weight_constant_axis=-2)
    array([[0.5],
           [0.5]])
    >>> estimate_mixture_weight([affiliation, affiliation])
    array([[[0.6],
            [0.4]],
    <BLANKLINE>
           [[0.6],
            [0.4]]])
    >>> estimate_mixture_weight([affiliation, affiliation], weight_constant_axis=-2)
    array([[0.5],
           [0.5]])
    >>> estimate_mixture_weight([affiliation, affiliation], weight_constant_axis=-3)
    array([[[0.4, 1. , 0.4],
            [0.6, 0. , 0.6]]])
    """
    affiliation = cp.asarray(affiliation)

    if (
        isinstance(weight_constant_axis, int)
        and weight_constant_axis % affiliation.ndim - affiliation.ndim == -2
    ):
        K = affiliation.shape[-2]
        return cp.full([K, 1], 1 / K)
    elif isinstance(weight_constant_axis, list):
        weight_constant_axis = tuple(weight_constant_axis)

    if saliency is None:
        weight = cp.mean(affiliation, axis=weight_constant_axis, keepdims=True)
    else:
        masked_affiliation = affiliation * saliency[..., None, :]
        weight = _unit_norm(
            cp.sum(masked_affiliation, axis=weight_constant_axis, keepdims=True),
            ord=1,
            axis=-2,
            eps=1e-10,
            eps_style="where",
        )

    return weight


def log_pdf_to_affiliation(
    weight,
    log_pdf,
    source_activity_mask=None,
    affiliation_eps=0.0,
):
    """
    Args:
        weight: Needs to be broadcast compatible (i.e. unsqueezed).
        log_pdf: Shape (..., K, N)
        source_activity_mask: Shape (..., K, N)
        affiliation_eps:
    Returns:
    """
    # Only check broadcast compatibility
    if source_activity_mask is None:
        _ = cp.broadcast_arrays(weight, log_pdf)
    else:
        _ = cp.broadcast_arrays(weight, log_pdf, source_activity_mask)

    # The value of affiliation max may exceed float64 range.
    # Scaling (add in log domain) does not change the final affiliation.
    affiliation = log_pdf - cp.amax(log_pdf, axis=-2, keepdims=True)

    cp.exp(affiliation, out=affiliation)

    # Weight multiplied not in log domain to avoid logarithm of zero.
    affiliation *= weight

    if source_activity_mask is not None:
        assert source_activity_mask.dtype == cp.bool, source_activity_mask.dtype  # noqa
        affiliation *= source_activity_mask

    denominator = cp.maximum(
        cp.sum(affiliation, axis=-2, keepdims=True),
        cp.finfo(affiliation.dtype).tiny,
    )
    affiliation /= denominator

    # Strictly, you need re-normalization after clipping. We skip that here.
    if affiliation_eps != 0:
        affiliation = cp.clip(
            affiliation,
            affiliation_eps,
            1 - affiliation_eps,
        )

    return affiliation


def is_broadcast_compatible(*shapes):
    if len(shapes) < 2:
        return True
    else:
        for dim in zip(*[shape[::-1] for shape in shapes]):
            if len(set(dim).union({1})) <= 2:
                pass
            else:
                return False
        return True


def normalize_observation(observation):
    """
    Attention: swap D and T dim
    The dimensions are swapped, because some calculations (e.g. covariance) do
    a reduction over the sample (time) dimension. Having the time dimension on
    the last axis improves the execution time.
    Args:
        observation: (..., T, D)
    Returns:
        normalized observation (..., D, T)
    """
    observation = _unit_norm(
        observation,
        axis=-1,
        eps=cp.finfo(observation.dtype).tiny,
        eps_style="where",
    )
    return cp.ascontiguousarray(cp.swapaxes(observation, -2, -1))
