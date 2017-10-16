def pretty_log_batch_size(n):
    if n < 10:
        return 1000000   # do not log individually
    if n < 100:
        return 10
    if n < 1000:
        return 100
    if n < 10000:
        return 1000
    if n < 100000:
        return 5000
    if n < 1000000:
        return 50000
    return 100000


def _locate_features(source, target):
    """
    If any item in `target` is not found in `source`,
    an exception is raised by design.
    """
    return list(source.index(t) for t in target)

