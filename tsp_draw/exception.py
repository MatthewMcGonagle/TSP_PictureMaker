'''
Custom exceptions for tsp_draw.
'''

class VertexPoolTooSmall(Exception):
    '''
    When there are too few vertices in a pool of vertices, e.g.
    in a pool of candidate vertices.
    '''
    def __init__(self, pool, message):
        Exception.__init__(self)
        self.message = message
        self.pool = pool
