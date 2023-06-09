# CGManifest Files
This directory contains CGManifest (cgmanifest.json) files.
See [here](https://docs.opensource.microsoft.com/tools/cg/cgmanifest.html) for details.

## `cgmanifests/generated/cgmanifest.json`
This file contains generated CGManifest entries.

It covers these dependencies:
- git submodules
- dependencies from the Dockerfile `tools/ci_build/github/linux/docker/Dockerfile.manylinux2014_cuda11`
- the entries in [../cmake/deps.txt](../cmake/deps.txt)

If any of these dependencies change, this file should be updated.
**When updating, please regenerate instead of editing manually.**

### How to Generate
1. Change to the repository root directory.
2. Ensure the git submodules are checked out and up to date. For example, with:
    ```
    $ git submodule update --init --recursive
    ```
3. Run the generator script:
    ```
    $ python cgmanifests/generate_cgmanifest.py --username <xxx> --token <your_access_token>
    ```

Please supply your github username and access token to the script. If you don't have a token, you can generate one at https://github.com/settings/tokens. This is for authenticating with Github REST API so that you would not hit the rate limit.

## `cgmanifests/cgmanifest.json`
This file contains non-generated CGManifest entries. Please edit directly as needed.
