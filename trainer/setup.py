#!/usr/bin/env python
# coding: utf-8

try:
    from IPython import get_ipython
    from IPython.utils.capture import capture_output
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

import subprocess

if HAS_IPYTHON:
    ip = get_ipython()
    if ip is not None:
        # with capture_output():
        subprocess.run(
            'bash -l -c "module load cmake && module load cuda"',
            shell=True
        )
        # subprocess.run('module load cmake && module load cuda', shell=True, executable="/bin/bash")
        
        # ip.run_line_magic('pip', 'install -q poetry')
        # poetry = '~/.local/bin/poetry'
        # ip.system(f'{poetry} config virtualenvs.create false')
        # ip.run_line_magic('cd', '../pcx')
        # ip.system(f'{poetry} install')
        # ip.system(f'{poetry} source add jax    https://storage.googleapis.com/jax-releases/jax_releases.html    --priority=explicit')
        # ip.system(f'{poetry} remove jax')
        # ip.system(f'{poetry} add "jax[cuda12]" --source jax')
        # ip.run_line_magic('pip', 'install -q -f https://storage.googleapis.com/jax-releases/jax_releases.html    "equinox>=0.11.7,<0.12.0" "jax[cuda12]>=0.5.3,<0.6.0" torch optax tqdm ripser')
        
        # ip.run_line_magic('cd', '../ripser-plusplus')
        # ip.run_line_magic('pip', 'install .')
        ip.run_line_magic('cd', '..')


import sys
sys.path.append('../pcx')
sys.path.append('../ripser-plusplus')
