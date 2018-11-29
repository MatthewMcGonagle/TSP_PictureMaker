'''
Allow an interactive session for doing annealing.
'''

import tspDraw.sizeScale
import numpy as np
import matplotlib.pyplot as plt

_commandShortcuts = { "stop" : "s"
                    , "continue" : "c"
                    , "graph energies" : "g"
                    , "graph cycle" : "c"
                    , "print stats" : "p"
                    , "change annealer" : "a"
                    , "change temperature" : "t"
                    , "change scale" : "e"
                    , "change cooling" : "l"
                    } 

class InputTranslation:

    def __init__(self, inputString):

        self.valid = False
        self.result = None
       
        for cmd, shortcut in _commandShortcuts.items():
            if inputString == cmd or inputString == shortcut:
                self.result = cmd
                self.valid = True
                return
    
class Session:

    def __init__(self, vertices, nJobsBetweenInquiry = 30, nStepsPerJob = 10**2, settings = None): 
        '''
        Parameters
        ----------
        vertices : Numpy array of shape (nPoints, 2)
            The vertices to do TSP on.
        '''

        self.vertices = vertices 
        self.annealer = None
        self.nJobsBetweenInquiry = nJobsBetweenInquiry 
        self.nStepsPerJob = nStepsPerJob

        self.running = False
        self.doingJobs = False
        self.graphingEnergies = False
        self.printingStats = True
        self.changingTemperature = False
        self.changingScale = False

        self.settings = settings
        self.command = None

        if self.settings == None:

            settings = tspDraw.sizeScale.guessSettings(vertices, nStepsPerJob, nJobsBetweenInquiry * 10)

        print(settings)
        self.annealer = tspDraw.sizeScale.Annealer(self.nStepsPerJob, self.vertices, **settings)
        self.energies = np.zeros(nJobsBetweenInquiry * 10)

    def run(self):
        self.running = True
        self.doingJobs = True
        print(self.annealer.getInfoString())
        while(self.running): 

            self._runState()
            self.doingJobs = False
            self._getNextCommand()
            self._setState()

        return 

    def _doAnnealingJob(self):

        self.annealer.doWarmRestart()
        startEnergy = self.annealer.getEnergy()
        newEnergies = startEnergy + np.array(list(self.annealer))
   
    def _printCommands(self):

        print("Commands")
        for i, cmd in enumerate(_commandShortcuts.keys()):
            print(cmd, " (", _commandShortcuts[cmd], end = " )\t\t")
            if i % 3 == 2:
                print(" ")
        print(" ")


    def _commandMatches(self, validCommand):

        return (self.command == validCommand) | (self.command == _commandShortcuts[validCommand])

    def _getNextCommand(self):

        haveCommand = False

        while not haveCommand:

            self._printCommands()
            command = input("What do you want to do next?")

            translation = InputTranslation(command)

            if not translation.valid:

                print("Command " + command + " unrecognized.\n")
                haveCommand = False

            else:
                haveCommand = True

            self.command = translation.result
               
    def _setState(self):
 
            if self.command == "stop":
                self.running = False

            elif self.command == "continue":
                self.doingJobs = True 

            elif self.command == "graph energies":
                self.graphingEnergies = True 

            elif self.command == "print stats":
                self.printingStats = True

            elif self.command == "change temperature":
                self.changingTemperature = True

            elif self.command == "change scale":
                self.changingScale = True
                
            elif self.command == "change annealer":
                pass 

    def _runState(self):

        if self.changingTemperature:

            newTemp = input("New Temperature")
            newTemp = float(newTemp)
            self.annealer.temperature = newTemp
            self.changingTemperature = False

        if self.changingScale:
        
            newScale = input("New Scale")
            newScale = float(newScale)
            self.annealer.sizeScale = newScale
            self.changingScale = False

        if self.doingJobs:

            newEnergies = []
            for i in range(self.nJobsBetweenInquiry):
                self._doAnnealingJob()
                newEnergies.append(self.annealer.getEnergy())
            newEnergies = np.array(newEnergies)
            self._appendEnergies(newEnergies)

        if self.printingStats:

            print("\n", self.annealer.getInfoString())

        if self.graphingEnergies:

            plt.plot(self.energies)
            plt.show() 

    def _appendEnergies(self, newEnergies):

        numNewEnergies = len(newEnergies)

        self.energies = np.concatenate([self.energies[numNewEnergies:], newEnergies], axis = 0)
