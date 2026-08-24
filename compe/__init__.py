from __future__ import annotations


def register_t12():
    from compe.t1 import register_t1
    from compe.t2 import register_t2

    register_t1()
    register_t2()


__all__ = ["register_t12"]
