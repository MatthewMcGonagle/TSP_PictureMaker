'''
Helper function for doing annealing job with annealer.
'''

import numpy as np

############################
#### Helper Functions
############################

def do_annealing(annealer, n_jobs):
    '''
    Do the annealing jobs (or annealing runs) for a particular annealer. This will perform a
    warm restart of the annealer between each job.

    After each job, it collects information on energy and prints information on the job.

    Parameters
    ----------
    annealer : An annealing iterator class
        The annealer to do the job.

    n_jobs : Int
        The number of jobs to run.

    Returns
    -------
        The energies (or length) of the path at the end of each job.
    '''

    energies = [annealer.get_energy()]

    for i in np.arange(n_jobs):
        print('Annealing Job ', i)

        annealer.do_warm_restart()
        for _ in annealer:
            continue

        energy = annealer.get_energy()
        energies.append(energy)

        print(annealer.get_info_string())

    energies = np.array(energies)

    return energies
