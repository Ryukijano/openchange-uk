"""Human-run compute-node checks; no datasets, weights, installation or downloads."""

from openchange.gpu_readiness import main

if __name__ == "__main__":
    raise SystemExit(main())
