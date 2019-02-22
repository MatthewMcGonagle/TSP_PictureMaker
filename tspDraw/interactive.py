'''
Allow an interactive session for doing annealing.
'''

import tspDraw.sizeScale
import tspDraw.neighbors
import tspDraw.sizeNeighbors
import numpy as np
import matplotlib.pyplot as plt
import keyboard

# _commandShortcuts = { "stop" : "s"
#                     , "continue" : "c"
#                     , "graph energies" : "g"
#                     , "graph cycle" : "c"
#                     , "print stats" : "p"
#                     , "change annealer" : "a"
#                     , "change temperature" : "t"
#                     , "change scale" : "e"
#                     , "change cooling" : "l"
#                     } 

class InputTranslation:

    def __init__(self, inputString, shortcuts):

        self.valid = False
        self.result = None
       
        for cmd, shortcut in shortcuts.items():
            if inputString == cmd or inputString == shortcut:
                self.result = cmd
                self.valid = True
                return


class InputManager:
    '''
    Responsible for getting input from the user and turning the input into a uniform format (user has multiple options
    to input commands, such as using shortcuts). 
    '''

    def __init__(self):
        self._mainShortcuts = { "stop" : "s"
                              , "continue" : "c"
                              , "graph energies" : "g"
                              , "graph result" : "r"
                              , "print stats" : "p"
                              , "change annealer" : "a"
                              , "change temperature" : "t"
                              , "change scale" : "e"
                              , "change cooling" : "l"
                              } 

        self._annealerShortcuts = { "sizeScale" : "s" 
                                  , "neighbors" : "n" 
                                  , "sizeNeighbors" : "i"
                                  }

    def getMainMenuChoice(self):

        return self._getShortcutMenu(self._mainShortcuts, "What do you want to do next?")

    def getFloat(self, name):
        validInput = False 

        while not validInput:
            newValue = input("New " + name + "? ") 
            try:
                newValue = float(newValue)
                validInput = True
            except:
                print("Invalid floating point number")
        return newValue
 
    def getAnnealerChoice(self):

        return self._getShortcutMenu(self._annealerShortcuts, "Which annealer do you want?")
        
    def _printCommands(self, shortcuts):

        print("Commands")
        for i, cmd in enumerate(shortcuts.keys()):
            print(cmd, " (", shortcuts[cmd], end = " )\t\t")
            if i % 3 == 2:
                print(" ")
        print(" ")

    def _getShortcutMenu(self, shortcuts, message):

        haveCommand = False

        while not haveCommand:

            self._printCommands(shortcuts)
            command = input(message)

            translation = InputTranslation(command, shortcuts)

            if not translation.valid:

                print("Command " + command + " unrecognized.\n")
                haveCommand = False

            else:
                haveCommand = True

            command = translation.result
    
        return command

     
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

        settings = settings

        if settings == None:

            settings = tspDraw.sizeScale.guessSettings(vertices, nStepsPerJob, nJobsBetweenInquiry * 10)
            settings['sizeCool'] = 1.0
        self.settings = settings

        print(self.settings)
        self.annealer = tspDraw.sizeScale.Annealer(self.nStepsPerJob, self.vertices, **settings)

        self.inputManager = InputManager()
        self.energies = np.zeros(nJobsBetweenInquiry * 10)

    def run(self):
        self.running = True
        self.doingJobs = True
        print(self.annealer.getInfoString())
        while(self.running): 

            self._runState()
            print("Press m for menu")
            if keyboard.is_pressed('m') or not self.doingJobs:
                #command = self._getNextCommand()
                command = self.inputManager.getMainMenuChoice()
                self._processCommand(command)
                #self._setState()

        return 

    def _doAnnealingJob(self):

        self.annealer.doWarmRestart()
        newEnergies = startEnergy + np.array(list(self.annealer))
   
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
                self.annealer.temperature = self.inputManager.getFloat("Temperature") 
                self.doingJobs = False

            elif command == "change scale":
                self._changeScale()
                self.doingJobs = False
                
            elif command == "change annealer":
                self._changeAnnealer()
                self.doingJobs = False
        
            elif command == "graph result":
                self._graphCycle()
                self.doingJobs = False

    def _changeScale(self):

         if type(self.annealer) is tspDraw.neighbors.Annealer:
             print("ANNEALER DOESN'T HAVE SIZE SCALE")
             self.doingJobs = False
             return
 
         self.annealer.sizeScale = self.inputManager.getFloat("Size Scale") 

         try:
             self.annealer.doWarmRestart()
         except:
             print("ERROR TRYING TO CREATE ANNEALER, NOT ENOUGH VERTICES IN POOL, TRY DECREASING THE SIZE SCALE") 
             self.doingJobs = False
      
    def _graphCycle(self):

        cycle = self.annealer.getCycle()
        plt.clf()
        plt.plot(cycle[:, 0], cycle[:, 1])
        plt.show() 
        plt.clf()

    def _changeAnnealer(self):

        newAnnealer = self.inputManager.getAnnealerChoice()

        #if type(self.annealer) is tspDraw.sizeScale.Annealer:
        print("Changing to ", newAnnealer)
        settings = { 'temperature' : self.annealer.temperature,
                     'tempCool' : self.annealer.tempCool
                   }
        self.vertices = self.annealer.vertices.copy()

        if newAnnealer == "neighbors": 
            settings.update( { 'kNbrs' : 30,
                               'nbrsCool' : 1
                             } )
            self.annealer = tspDraw.neighbors.Annealer(self.nStepsPerJob, self.vertices, **settings) 

        elif newAnnealer == "sizeScale":
            settings.update(  tspDraw.sizeScale.guessSettings(self.vertices, self.nStepsPerJob, self.nJobsBetweenInquiry * 10))
            settings['sizeCool'] = 1.0
            self.annealer = tspDraw.sizeScale.Annealer(self.nStepsPerJob, self.vertices, **settings) 

        elif newAnnealer == "sizeNeighbors":
             settings.update(  tspDraw.sizeScale.guessSettings(self.vertices, self.nStepsPerJob, self.nJobsBetweenInquiry * 10))
             settings['sizeCool'] = 1.0
             settings.update( { 'kNbrs' : 30,
                                'nbrsCool' : 1
                              } )
             self.annealer = tspDraw.sizeNeighbors.Annealer(self.nStepsPerJob, self.vertices, **settings)

    def _runState(self):

        if self.doingJobs:

            newEnergies = []
            try:
                self.annealer.doWarmRestart()
            except:
                print("ERROR TRYING TO CREATE ANNEALER \n TRY DIFFERENT SETTINGS")
                self.doingJobs = False 
                return

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
