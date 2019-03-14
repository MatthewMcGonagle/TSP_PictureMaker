'''
Allow an interactive session for doing annealing.
'''

import numpy as np
import matplotlib.pyplot as plt
import keyboard

import tsp_draw.size_scale
import tsp_draw.neighbors
import tsp_draw.size_neighbors
import tsp_draw.user_input

class SessionState:
    '''
    Keeps track of different states of an interactive session.
    '''

    def __init__(self):
        '''
        Initiate the state as not running, not doing jobs, do graph energies,
        and do print stats.
        '''
        self.running = False
        self.doing_jobs = False
        self.graphing_energies = True
        self.printing_stats = True

class Session:
    '''
    Runs an interactive session.
    '''

    def __init__(self, vertices, n_jobs_between_inquiry = 5, n_steps_per_job = 300,
                 settings = None):
        '''
        Parameters
        ----------
        vertices : Numpy array of shape (nPoints, 2)
            The vertices to do TSP on.
        '''

        self.vertices = vertices
        self.annealer = None
        self.n_jobs_between_inquiry = n_jobs_between_inquiry
        self.n_steps_per_job = n_steps_per_job

        self.state = SessionState()

        if settings is None:

            settings = tsp_draw.size_scale.guess_settings(vertices, n_steps_per_job,
                                                         n_jobs_between_inquiry * 10)
            settings['size_cool'] = 1.0

        self.annealer = tsp_draw.size_scale.Annealer(self.n_steps_per_job, self.vertices, **settings)

        self.energies = np.zeros(n_jobs_between_inquiry * 10)

    def run(self):
        '''
        Run the interactive session. Will loop over running annealers, updating graphs of
        results, and getting user input. Runs until the user tells the session to stop.
        '''
        self.state.running = True
        self.state.doing_jobs = True
        print(self.annealer.get_info_string())
        while self.state.running:

            self._run_state()
            print("Press m for menu")
            if keyboard.is_pressed('m') or not self.state.doing_jobs:
                #command = self._getNextCommand()
                command = tsp_draw.user_input.get_main_menu_choice()
                self._process_command(command)
                #self._setState()

    def _do_annealing_job(self):

        self.annealer.do_warm_restart()
        #new_energies = startEnergy + np.array(list(self.annealer))

    def _process_command(self, command):
        if command == "stop":
            self.state.running = False

        elif command == "continue":
            self.state.doing_jobs = True

        elif command == "graph energies":
            self.state.graphing_energies = True

        elif command == "print stats":
            self.state.printing_stats = True

        elif command == "change temperature":
            self.annealer.temperature = tsp_draw.user_input.get_float("Temperature")
            self.state.doing_jobs = False

        elif command == "change scale":
            self._change_scale()
            self.state.doing_jobs = False

        elif command == "change annealer":
            self._change_annealer()
            self.state.doing_jobs = False

        elif command == "graph result":
            print("\nClose Graph Window to return to the menu.\n")
            self._graph_cycle()
            self.state.doing_jobs = False

    def _change_scale(self):

        if isinstance(self.annealer, tsp_draw.neighbors.Annealer):
            print("ANNEALER DOESN'T HAVE SIZE SCALE")
            self.state.doing_jobs = False
            return

        self.annealer.size_scale = tsp_draw.user_input.get_float("Size Scale")

        try:
            self.annealer.do_warm_restart()
        except tsp_draw.exception.VertexPoolTooSmall as inst:
            message = ("**************\n" +
                       inst.message +
                       "\nTry lowering the size scale." +
                       "\n***************")
            print(message)
            self.state.doing_jobs = False

    def _graph_cycle(self):

        cycle = self.annealer.get_cycle()
        plt.clf()
        plt.plot(cycle[:, 0], cycle[:, 1])
        plt.show()
        plt.clf()

    def _change_annealer(self):

        new_annealer = tsp_draw.user_input.get_annealer_choice()

        #if type(self.annealer) is tsp_draw.size_scale.Annealer:
        print("Changing to ", new_annealer)
        settings = {'temperature' : self.annealer.temperature,
                    'temp_cool' : self.annealer.temp_cool
                   }
        self.vertices = self.annealer.vertices.copy()

        if new_annealer == "neighbors":
            settings.update({'k_nbrs' : 30,
                             'nbrs_cool' : 1
                            })
            self.annealer = tsp_draw.neighbors.Annealer(self.n_steps_per_job, self.vertices,
                                                       **settings)

        elif new_annealer == "size_scale":
            settings.update(tsp_draw.size_scale.guess_settings(self.vertices, self.n_steps_per_job,
                                                              self.n_jobs_between_inquiry * 10))
            settings['size_cool'] = 1.0
            self.annealer = tsp_draw.size_scale.Annealer(self.n_steps_per_job,
                                                        self.vertices, **settings)

        elif new_annealer == "size_neighbors":
            settings.update(tsp_draw.size_scale.guess_settings(self.vertices, self.n_steps_per_job,
                                                              self.n_jobs_between_inquiry * 10))
            settings['size_cool'] = 1.0
            settings.update({'k_nbrs' : 30,
                             'nbrs_cool' : 1
                            })
            self.annealer = tsp_draw.size_neighbors.Annealer(self.n_steps_per_job,
                                                            self.vertices, **settings)

    def _run_state(self):

        if self.state.doing_jobs:

            new_energies = []
            try:
                self.annealer.do_warm_restart()
            except tsp_draw.exception.VertexPoolTooSmall as inst:
                message = ("**************\n" +
                           inst.message +
                           "\nTry lowering the size scale." +
                           "\n**************")
                print(message)
                self.state.doing_jobs = False
                return

            for _ in range(self.n_steps_per_job):
                #self._do_annealing_job()
                next(self.annealer)
            new_energies.append(self.annealer.get_energy())
            new_energies = np.array(new_energies)
            self._append_energies(new_energies)

        if self.state.printing_stats:

            print("\n", self.annealer.get_info_string())

        if self.state.graphing_energies:

            plt.cla()
            plt.plot(self.energies)
            plt.pause(0.001)

    def _append_energies(self, new_energies):
        '''
        Update the most recent energy levels tracked by the session. This is used to give
        feedback to the user on the recent trends in the energy levels, e.g. graphs of
        recent energy levels

        Parameters
        ----------
        new_energies : Numpy array of Float
            The most recent energies.
        '''
        num_new_energies = len(new_energies)
        self.energies = np.concatenate([self.energies[num_new_energies:], new_energies], axis = 0)
