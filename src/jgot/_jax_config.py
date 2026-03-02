"""Configure package-wide JAX defaults.

Importing this module enables x64 mode for JAX within ``jgot``.
"""

import jax

jax.config.update("jax_enable_x64", True)
