'''
Allows us to fake random state for doing deterministic and predictable unit tests
for functions that have random behavior.
'''

class State:

    def __init__(self, uniform_stack):
        self.uniform_stack = uniform_stack.copy()

    def uniform(self):
        return self.uniform_stack.pop()
