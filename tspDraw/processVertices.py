import numpy as np

############################
### Greedy Guesser
############################

class GreedyGuesser3:
    '''
    Make an initial greedy guess for a solution to the Traveling Salesman Problem. If there is an odd number of vertices, then
    this method will drop the last vertex to make an even number of vertices; the idea being that we wish to work with
    such a large number of vertices that one single vertex won't make much difference.

    The guess is made by initially pairing up vertices to their closest neighbors. Then iteratively start choosing path segment
    pairs to connect (from either end) based on minimizing the distance added. We try to keep the legnths of the path segments
    uniform, but we don't necessarily do this. We connect segments in order, i.e. we try connecting a given segment to the
    segments that occur after its position in our list of curves. Odd number of curve segments will force segments to
    not be the same size, but this nonuniformity is minimized. 

    Heuristically, this uniformity in size seems like a good idea, because it should allow a good portion of each segment to
    be locally correct because its construction is relatively independent of the other curves.

    We are finished when there is only one segment left. Note that we don't have any guarantee that the beginning of the segment
    is close to the end of the segment.

    Members
    -------
    curves : List of Numpy Arrays, each of shape (segmentVertexCount, 2)
        The list of current path segments. Each one is a numpy array of shape (segmentVertexCount, 2); so the xy-coordinates of the
        vertices in the segment in the order they appear in the segment. Note that the different segments don't have
        the same number of vertices. 
    
    endPts : Dictionary with Values in Numpy Arrays of Shape (SegmentCount, 2)  
        This keeps track of the beginning vertex of each segment in self.curves and the end vertex in self.curves. This is for
        quick numpy calculations for finding the shortest links when connecting existing segments. 
        endPts['begin'] = The xy-coordinates of the beginning vertex of each path segment in self.curves.
        endPts['end'] = The xy-coordinates of the ending vertex of each path segment in self.curves.

    vertices :
        The xy-coordinates of the vertices in their original order. Note that we force this to be an even number of vertices,
        dropping the last vertex if we are in the case of an odd number of vertices.

    nVertices : Int
        The number of vertices we are working with. We force this to be even.
    '''

    def __init__(self):
        '''
        Initialize all of the members to None.
        '''

        self.curves = None
        self.endPts = None
        self.vertices = None
        self.nVertices = None

    def makeGuess(self, vertices):
        '''
        We make a greedy guess for a list of vertices.

        Parameters
        ----------
        vertices : Numpy array of shape (nVertices, 2)
            The list of xy-coordinates of the vertices we wish to make a greedy guess of the solution to TSP for; however 
            we require that vertices only have an even number of members. If it has an odd number of vertices, then the 
            last vertex will be dropped.

        Returns
        -------
        Numpy array of shape (newNVertices, 2)
            The xy-coordinates in order of our greedy guess. Note that if we passed an odd number of vertices then
            the number of vertices by one; in the case that we passed an even number of vertices, the number of
            vertices stays the same.
        '''

        self.vertices = vertices
        self.nVertices = len(vertices)
        self.curves = []
   
        # Drop the last vertex if there is an odd number of vertices.
 
        if self.nVertices % 2 == 1:

            self.vertices = self.vertices[:-1]
            self.nVertices -= 1

        # Do the initial greedy pairing to create our initial curve segments. 

        self.__initializeCurves()       

        # While we have more than one curve segment, greedily connect segments.

        while len(self.curves) > 1:

            self.__connectCurves()

        # When we are left with only one curve segment, it is our greedy guess.

        return self.curves[0]

    def __initializeCurves(self):
        '''
        Do the initial greedy pairing. 
        '''

        nProcessed = 0

        beginPts = []
        endPts = []

        # For each vertex not added to a pair so far, find its nearest neighbor out of
        # the vertices that haven't been put in a pair so far.
        # For ease of determining which ones haven't been paired so far, we switch the paired
        # vertices to occur at the beginning of the array, so that all of the vertices to
        # the right of our current position are exactly the vertices that haven't been
        # processed so far.

        while nProcessed < self.nVertices:

            # Get the unpaired vertices to the right of our current position.

            candidateStart = nProcessed + 1
            candidateVerts = self.vertices[candidateStart :]

            # Find the candidate that gives the shortest connection.

            distances = np.linalg.norm(self.vertices[nProcessed] - candidateVerts, axis = -1)
            partnerI = np.argmin(distances, axis = 0) + candidateStart 

            # Swap the chosen candidate to be directly after the current position.

            self.__swapVertices(nProcessed+1, partnerI)

            # The pair gives a new curve; update the list of curves, list of beginning points,
            # and the list of end points.

            newCurve = np.array([self.vertices[nProcessed], self.vertices[nProcessed + 1]])
            self.curves.append(newCurve)
            beginPts.append(self.vertices[nProcessed])
            endPts.append(self.vertices[nProcessed + 1])

            # We processed a pair so we need to advance by 2 (recall that the next position is
            # now the chosen candidate).

            nProcessed += 2

        self.endPts = {'begin' : np.array(beginPts),
                       'end' : np.array(endPts) }

    def __connectCurves(self):
        '''
        Go through the curves in order and greedily connect them to another curve after their position in such 
        a way that gives the shortest connection. We minimize the nonuniformity in the segment size, but when there
        is an odd number of segments there will be segments of different sizes.
        '''

        nCurves = len(self.curves)
        curveI = 0

        newCurves = []


        # For each curve we find the shortest connection for all curves in a position
        # of our list of curves that occurs after the current curve. We do this to
        # try to keep the curves all of the same size (although they won't necessarily be).
        # We iterate until curveI == nCurves - 2, because we want to make sure that
        # there are atleast two curves left to join. 

        while curveI < nCurves - 1:

            sourceEndPt, destinationI, destEndPt = self.__findShortestConnection(curveI)
            self.__connectSource(sourceEndPt, curveI, destinationI, destEndPt)
            self.__removeCurve(destinationI)

            curveI += 1 
            nCurves -= 1

    def __connectSource(self, sourceEndPt, sourceI, destinationI, destEndPt):
        '''
        The connection is made so that the new connected curve always starts at one of the endpoints of the source
        and ends at one of the endpoints of the destination.

        The new curve replaces the curve at self.curves[sourceI]. This function will not delete/remove the other curve;
        it will need to be removed using self.__removeCurve(). 

        Parameters
        ----------
        sourceEndPt : String
            Should be either 'begin' or 'end' to indicate which side of the source curve the connection should
            be made.

        sourceI : Int
            The index of the source curve for the connection.

        destinationI : Int
            The index of the destination curve for the connection.

        destEndPt : String
            Should be either 'begin' or 'end' to indicate which side of the destination curve the connection
            should be made.
        '''

        # When the we connect to the beginning of the source curve, then we need to flip its order.

        if sourceEndPt == 'end':

            sourceCurve = self.curves[sourceI]

        else:

            sourceCurve = np.flip(self.curves[sourceI], axis  = 0)
            self.endPts['begin'][sourceI] = self.endPts['end'][sourceI]

        # When we connect to the end of the destination curve, then we need to flip its order.

        if destEndPt == 'begin':

            destCurve = self.curves[destinationI]
            self.endPts['end'][sourceI] = self.endPts['end'][destinationI]

        else:

            destCurve = np.flip(self.curves[destinationI], axis = 0)
            self.endPts['end'][sourceI] = self.endPts['begin'][destinationI]

        newCurve = np.concatenate([sourceCurve, destCurve], axis = 0)
        self.curves[sourceI] = newCurve

    def __removeCurve(self, curveI):
        '''
        Remove a curve, updating the list of curves and list of endpoints.

        Parameters
        ----------
        curveI : Int
            The index of the curve to remove.

        '''

        del self.curves[curveI]

        for endPt in ['begin', 'end']:

            self.endPts[endPt] = np.delete(self.endPts[endPt], curveI, axis = 0)

 
    def __findShortestConnection(self, sourceI): 
        '''
        Find the curve after the given curve at index sourceI that will give the shortest
        connection to the curve at self.curves[sourceI]. Note, the connection only
        considers connecting endpoints.

        Parameters
        ----------
        sourceI : Int
            The index of the curve to consider connecting to other curves.

        Returns
        -------
        bestSourceEndPt : String
            bestSourceEndPt is either 'begin' or 'end' to indicate which source endpoint
            should be used for the connection.

        bestDestinationI : Int
            bestDestinationI is the index of the curve that we should connect the source curve to.

        bestDestEndPt : String
            Either 'begin' or 'end' to indicate which endpoint of the destination curve to connect to. 
        '''


        # A negative bestDistance indicates that we haven't found any distances yet.

        bestDist = -1 

        bestSourceEndPt = '' 
        bestDestinationI = -1
        bestDestEndPt = ''

        # Loop over beginning and ending for the endpoints of source and destination.

        for source in self.endPts.keys():

            sourceVertex = self.endPts[source][sourceI]

            for destination in self.endPts.keys():

                # To try to keep curve size uniform, we only consider connecting
                # to curves after the source curve (which shouldn't have been connected yet
                # in this pass). 

                candidates = self.endPts[destination][sourceI + 1 :, : ]
                distances = np.linalg.norm(sourceVertex - candidates, axis = -1)

                minDistI = np.argmin(distances) 
                dist = distances[minDistI]

                # If we have a new best distance then record needed info.

                if bestDist < 0 or dist < bestDist:
                    bestDist = dist
                    bestSourceEndPt = source
                    
                    # Make sure to account for the fact that indices of candidates is offset
                    # from indices in self.curves.

                    bestDestinationI = minDistI + sourceI + 1
                    bestDestEndPt = destination

        return bestSourceEndPt, bestDestinationI, bestDestEndPt

    def __swapVertices(self, i, j):
        '''
        Swap vertices at indices i and j inside self.vertices.

        Parameters
        ----------
        i : Int
        
        j : Int
        '''

        temp = self.vertices[i, :].copy()
        self.vertices[i, :] = self.vertices[j, :].copy()
        self.vertices[j, :] = temp.copy()
     
#############################
### Helper Functions
#############################

def preprocess(vertices):
    '''
    Normalize vertices and make our intial greedy guess as to the solution of the Traveling Salesman Problem.
    If there is an odd number of vertices, then our greedy guess will drop the last vertex.

    Parameters
    ----------
    vertices : Numpy array of shape (nVertices, 2)
        The xy-coordinates of the vertices.

    Returns
    -------
    Numpy array of shape (newNVertices, 2)
        The xy-coordinates of our processed vertices.
    '''

    vertices = normalizeVertices(vertices)
    guesser = GreedyGuesser3()
    vertices = guesser.makeGuess(vertices)
    vertices = np.roll(vertices, 100, axis = 0)

    return vertices

def normalizeVertices(vertices):
    '''
    Normalize the xy-coordinates by scaling by the reciprocal of the largest y-coordinate.

    Parameters
    ----------
    vertices : Numpy array of shape (nVertices, 2)
        The xy-coordinates of the vertices.

    Returns
    -------
    Numpy array of shape (nVertices, 2)
        The normalized xy-coordinates of the vertices. 
    '''

    scale = np.amax(vertices[:, 1])
    vertices = vertices.astype('float')
    vertices[:, 0] = vertices[:, 0] / scale
    vertices[:, 1] = vertices[:, 1] / scale

    return vertices


