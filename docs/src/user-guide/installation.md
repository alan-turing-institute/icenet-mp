# Installation

## Prerequisites

You will need [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to manage the Python environment.

!!! note
    On an HPC system, `uv` installs to `~/.local/bin`, so make sure your home directory has enough free space.

## Installing IceNet-MP

??? warning "Isambard-AI: additional prerequisites required"
    Isambard-AI uses ARM processors and there is currently no `aarch64` wheel for `cf-units`.
    Installing `cf-units` requires a compatible `udunits` installation.
    The IceNet-MP dev team have compiled `udunits` on Isambard-AI and it can be used by setting the following environment variables:

    ```bash
    export UDUNITS2_XML_PATH=/projects/u6iz/public/shared/udunits/share/udunits/udunits2.xml
    export UDUNITS2_INCDIR=/projects/u6iz/public/shared/udunits/include/
    export UDUNITS2_LIBDIR=/projects/u6iz/public/shared/udunits/lib/
    ```

Clone the repository and install with `uv`:

```bash
git clone git@github.com:alan-turing-institute/icenet-mp.git
cd icenet-mp
uv sync --managed-python
```
