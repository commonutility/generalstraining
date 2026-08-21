"""Environment seam over the vendored generals-bots simulator.

The simulator lives in ``vendor/generals-bots`` (installed as the ``generals``
package). Import the env through this module so wrappers or env variants can
be added here without touching training code.
"""

from generals.core.env import GeneralsEnv

__all__ = ["GeneralsEnv"]
