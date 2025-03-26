import inspect
from jax import vmap


def _get_vmap_tuple(fn, argnames, axes):
    """Get a tuple of axes for `vmap` from argument names.
    The tuple has axes[i] for argnames[i] and None for other arguments.
    """
    if not isinstance(argnames, (list, tuple)):
        argnames = [argnames]

    if axes is None:
        axes = [0] * len(argnames)
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    sig = inspect.signature(fn)

    fn_argnames_without_args_kwargs = [
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]

    output_tuple = [None] * len(fn_argnames_without_args_kwargs)
    for argname, axis in zip(argnames, axes):
        argpos = fn_argnames_without_args_kwargs.index(argname)
        output_tuple[argpos] = axis

    return tuple(output_tuple)


def named_vmap(function, in_axes, out_axes=None):
    """vmap with named arguments.

    Parameters
    ----------
    function : callable
        Function to be transformed by `vmap`.
    in_axes : list or tuple
        List of argument names to be transformed by `vmap`.
    out_axes : list or tuple
        The axes to map over for each argument. Defaults to all zeros.

    Returns
    -------
    callable
        A function that applies `vmap` to the specified arguments.
    """
    in_axes = _get_vmap_tuple(function, in_axes, out_axes)
    return vmap(function, in_axes=in_axes)
