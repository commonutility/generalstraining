"""Fail fast unless JAX can execute on an NVIDIA GPU."""

import jax
import jax.numpy as jnp


def main() -> None:
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError(f"Expected a GPU device, got {devices}")
    result = jnp.arange(8, dtype=jnp.float32).sum()
    print(f"GPU computation result: {result.item()}")


if __name__ == "__main__":
    main()
