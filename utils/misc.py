from resource import RLIMIT_NOFILE, getrlimit, setrlimit


def set_resource_limit(n: int):
    """Sets a new soft resource limits for the current process."""
    setrlimit(RLIMIT_NOFILE, (n, getrlimit(RLIMIT_NOFILE)[1]))


def true(*_, **__) -> bool:
    """Always returns true."""
    return True
