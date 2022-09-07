import itertools


def unique_everseen(iterable, key=None):
    """
    List unique elements, preserving order. Remember all elements ever seen [1]_.

    Examples
    --------
    >>> list(unique_everseen("AAAABBBCCDAABBB"))
    ["A", "B", "C", "D"]
    >>> list(unique_everseen("ABBCcAD", str.lower))
    ["A", "B", "C", "D"]

    References
    ----------
    .. [1] https://docs.python.org/3/library/itertools.html
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element
