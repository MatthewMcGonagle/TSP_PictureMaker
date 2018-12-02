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

        self.settings = settings

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
                command = self._getNextCommand()
                self._processCommand(command)
                #self._setState()

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

            command = translation.result
    
        return command
              
    def _processCommand(self, command):
        
            if command == "stop":
                self.running = False

            elif command == "continue":
                self.doingJobs = True 

            elif command == "graph energies":
                self.graphingEnergies = True 

            elif command == "print stats":
                self.printingStats = True

            elif command == "change temperature":
                self._changeTemperature() 
                self.doingJobs = False

            elif command == "change scale":
                self._changeScale()
                self.doingJobs = False
                
            elif command == "change annealer":
                self._changeAnnealer()
                self.doingJobs = False
        
            elif command == "graph result":
                self._graphResult()
                self.doingJobs = False

    def _changeTemperature(self):

        validInput = False 
        while not validInput:

            newTemp = input("New Temperature")
            try:
                newTemp = float(newTemp)
                validInput = True
            except:
                print("Invalid floating point number")
                
        self.annealer.temperature = newTemp

    def _changeScale(self):

         validInput = False

         while not validInput:
        
            newScale = input("New Scale")

            try:
                newScale = float(newScale)
                validInput = True
            except:
                print("Invalid floating point number")

         self.annealer.sizeScale = newScale

         try:
             self.annealer.doWarmRestart()
         except:
             print("ERROR TRYING TO CREATE ANNEALER, BAD SIZE SCALE \nTRY DIFFERENT SETTINGS")
             self.doingJobs = False
      
    def _graphResult(self):

        cycle = self.annealer.getCycle()
        plt.clf()
        plt.plot(cycle[:, 0], cycle[:, 1])
        plt.show() 
        plt.clf()

    def _changeAnnealer(self):

        print("Inside _changeAnnealer()")
        if type(self.annealer) is tspDraw.sizeScale.Annealer:

            print("Changing annealer")
            self.vertices = self.annealer.getCycle()[:-1]
            settings = { 'temperature' : self.annealer.temperature,
                         'cooling' : self.annealer.tempCool,
                         'kNbrs' : 30,
                         'nbrsCooling' : 1
                        }
            self.annealer = tspDraw.neighbors.Annealer(self.nStepsPerJob, self.vertices, **settings) 
        

    def _runState(self):

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
            plt.pause(0.001) 

    def _appendEnergies(self, newEnergies):

        numNewEnergies = len(newEnergies)

        self.energies = np.concatenate([self.energies[numNewEnergies:], newEnergies], axis = 0)
