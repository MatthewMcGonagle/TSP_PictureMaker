'''
Allow an interactive session for doing annealing.
'''

import tspDraw.sizeScale
import numpy as np
import matplotlib.pyplot as plt
import keyboard

_commandShortcuts = { "stop" : "s"
                    , "continue" : "c"
                    , "graph energies" : "g"
                    , "graph cycle" : "c"
                    , "graph result" : "r"
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

    def __init__(self, vertices, nJobsBetweenInquiry = 5, nStepsPerJob = 300, settings = None): 
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
        self.graphingEnergies = True 
        self.printingStats = True
        self.changingTemperature = False
        self.changingScale = False

        self.settings = settings
        self.command = None

        if self.settings == None:

            settings = tspDraw.sizeScale.guessSettings(vertices, nStepsPerJob, nJobsBetweenInquiry * 10)
            settings['sizeCool'] = 1.0

        print(settings)
        self.annealer = tspDraw.sizeScale.Annealer(self.nStepsPerJob, self.vertices, **settings)
        self.energies = np.zeros(nJobsBetweenInquiry * 10)

    def run(self):
        self.running = True
        self.doingJobs = True
        print(self.annealer.getInfoString())
        while(self.running): 

            self._runState()
            print("Press m for menu")
            if keyboard.is_pressed('m') or not self.doingJobs:
                self._getNextCommand()
                self._setState()

        return 

    def _doAnnealingJob(self):

        self.annealer.doWarmRestart()
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
        
            elif self.command == "graph result":
                cycle = self.annealer.getCycle()
                plt.clf()
                plt.plot(cycle[:, 0], cycle[:, 1])
                plt.show() 
                plt.clf()

                self.doingJobs = False

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
            try:
                self.annealer.doWarmRestart()
            except:
                print("ERROR TRYING TO CREATE ANNEALER \n TRY DIFFERENT SETTINGS")
                self.doingJobs = False

            self.changingScale = False

        if self.doingJobs:

            newEnergies = []
            try:
                self.annealer.doWarmRestart()
            except:
                print("ERROR TRYING TO CREATE ANNEALER \n TRY DIFFERENT SETTINGS")
                self.doingJobs = False 

            for i in range(self.nStepsPerJob):
                #self._doAnnealingJob()
                next(self.annealer)
            newEnergies.append(self.annealer.getEnergy())
            newEnergies = np.array(newEnergies)
            self._appendEnergies(newEnergies)

        if self.printingStats:

            print("\n", self.annealer.getInfoString())

        if self.graphingEnergies:

            plt.cla()
            plt.plot(self.energies)
            plt.pause(0.01) 

    def _appendEnergies(self, newEnergies):

        numNewEnergies = len(newEnergies)

        self.energies = np.concatenate([self.energies[numNewEnergies:], newEnergies], axis = 0)
