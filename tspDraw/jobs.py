import numpy as np

############################
#### Helper Functions
############################
def doAnnealing(annealer, nJobs):
    '''
    Do the annealing jobs (or annealing runs) for a particular annealer. This will perform a
    warm restart of the annealer between each job.

    After each job, it collects information on energy and prints information on the job. 

    Parameters
    ----------
    annealer : An annealing iterator class
        The annealer to do the job.

    nJobs : Int
        The number of jobs to run.

    Returns
    -------
        The energies (or length) of the path at the end of each job.
    '''

    energies = [annealer.getEnergy()]
    
    for i in np.arange(nJobs):
        print('Annealing Job ', i)
       
        annealer.doWarmRestart() 
        for step in annealer:
            continue
   
        energy = annealer.getEnergy() 
        energies.append(energy)

        print(annealer.getInfoString())
    
    energies = np.array(energies)

    return energies


