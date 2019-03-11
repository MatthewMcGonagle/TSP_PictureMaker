'''
Responsible for getting input from the user and turning the input into a uniform format (user has
multiple options to input commands, such as using shortcuts).
'''

class InputTranslation:
    '''
    Translation of user input; input may be a full command
    or a shortcut.

    Members
    -------
    result : String
        The uniform version of the command.

    valid : Bool
        Whether the input was valid.
    '''

    def __init__(self, input_string, shortcuts):
        '''
        Use a dictionary of shortcuts to translate input to uniform format.

        Parameters
        ----------
        input_string : String
            The user input.

        shortcuts : Dictionary
            Keys are the uniform format strings. Values are the shortcut strings.
        '''

        self.valid = False
        self.result = None

        for cmd, shortcut in shortcuts.items():
            if input_string in (cmd, shortcut):
                self.result = cmd
                self.valid = True
                return

def get_main_menu_choice():
    '''
    Get a choice from the user from the main menu.

    Returns
    -------
    command : String
        Translation of user choice to a standard format.
    '''
    main_shortcuts = {"stop" : "s",
                      "continue" : "c",
                      "graph energies" : "g",
                      "graph result" : "r",
                      "print stats" : "p",
                      "change annealer" : "a",
                      "change temperature" : "t",
                      "change scale" : "e",
                      "change cooling" : "l"
                     }

    return _get_shortcut_menu(main_shortcuts, "What do you want to do next?")

def get_float(name):
    '''
    Have the user enter a floating point number. Will continue to ask the user for a value
    until they enter a valid floating point number.

    Parameters
    ----------
    name : String
        What to call the floating point value when asking the user to enter a value.

    Returns
    -------
    value : Float
        The correct value entered by the user.
    '''
    valid_input = False

    while not valid_input:
        new_value = input("New " + name + "? ")
        try:
            new_value = float(new_value)
            valid_input = True
        except:
            print("Invalid floating point number")
    return new_value

def get_annealer_choice():
    '''
    Have the user enter a choice of annealer from the annealer menu.

    Returns
    -------
    Annealer : String
        The choice of the user translated to a standard format.
    '''
    annealer_shortcuts = {"size_scale" : "s",
                          "neighbors" : "n",
                          "size_neighbors" : "i"
                         }

    return _get_shortcut_menu(annealer_shortcuts, "Which annealer do you want?")

def _print_commands(shortcuts):

    print("Commands")
    for i, cmd in enumerate(shortcuts.keys()):
        print(cmd, " (", shortcuts[cmd], end = " )\t\t")
        if i % 3 == 2:
            print(" ")
    print(" ")

def _get_shortcut_menu(shortcuts, message):

    have_command = False

    while not have_command:

        _print_commands(shortcuts)
        command = input(message)

        translation = InputTranslation(command, shortcuts)

        if not translation.valid:

            print("Command " + command + " unrecognized.\n")
            have_command = False

        else:
            have_command = True

        command = translation.result

    return command
