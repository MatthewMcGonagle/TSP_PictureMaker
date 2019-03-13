'''
Pre-processing of vertices before applying annealers. In particular, make a greedy guess.
'''

import numpy as np

############################
### Greedy Guesser
############################

class GreedyGuesser3:
    '''
    Make an initial greedy guess for a solution to the Traveling Salesman Problem. If there is an
    odd number of vertices, then this method will drop the last vertex to make an even number of
    vertices; the idea being that we wish to work with such a large number of vertices that one
    single vertex won't make much difference.

    The guess is made by initially pairing up vertices to their closest neighbors. Then iteratively
    start choosing path segment pairs to connect (from either end) based on minimizing the distance
    added. We try to keep the legnths of the path segments uniform, but we don't necessarily do
    this. We connect segments in order, i.e. we try connecting a given segment to the segments that
    occur after its position in our list of curves. Odd number of curve segments will force
    segments to not be the same size, but this nonuniformity is minimized.

    Heuristically, this uniformity in size seems like a good idea, because it should allow a good
    portion of each segment to be locally correct because its construction is relatively
    independent of the other curves.

    We are finished when there is only one segment left. Note that we don't have any guarantee that
    the beginning of the segment is close to the end of the segment.

    Members
    -------
    curves : List of Numpy Arrays, each of shape (segment_vertex_count, 2)
        The list of current path segments. Each one is a numpy array of shape
        (segment_vertex_count, 2); so the xy-coordinates of the vertices in the segment in the order
        they appear in the segment. Note that the different segments don't have the same number of
        vertices.

    end_pts : Dictionary with Values in Numpy Arrays of Shape (SegmentCount, 2)
        This keeps track of the beginning vertex of each segment in self.curves and the end vertex
        in self.curves. This is for quick numpy calculations for finding the shortest links when
        connecting existing segments.
        end_pts['begin'] = The xy-coordinates of the beginning vertex of each path segment in
            self.curves.
        end_pts['end'] = The xy-coordinates of the ending vertex of each path segment in
            self.curves.

    vertices :
        The xy-coordinates of the vertices in their original order. Note that we force this to be an
        even number of vertices, dropping the last vertex if we are in the case of an odd number of
        vertices.

    n_vertices : Int
        The number of vertices we are working with. We force this to be even.
    '''

    def __init__(self):
        '''
        Initialize all of the members to None.
        '''

        self.curves = None
        self.end_pts = None
        self.vertices = None
        self.n_vertices = None

    def make_guess(self, vertices):
        '''
        We make a greedy guess for a list of vertices.

        Parameters
        ----------
        vertices : Numpy array of shape (n_vertices, 2)
            The list of xy-coordinates of the vertices we wish to make a greedy guess of the
            solution to TSP for; however we require that vertices only have an even number of
            members. If it has an odd number of vertices, then the last vertex will be dropped.

        Returns
        -------
        Numpy array of shape (newNVertices, 2)
            The xy-coordinates in order of our greedy guess. Note that if we passed an odd number of
            vertices then the number of vertices by one; in the case that we passed an even number
            of vertices, the number of vertices stays the same.
        '''

        self.vertices = vertices
        self.n_vertices = len(vertices)
        self.curves = []

        # Drop the last vertex if there is an odd number of vertices.

        if self.n_vertices % 2 == 1:

            self.vertices = self.vertices[:-1]
            self.n_vertices -= 1

        # Do the initial greedy pairing to create our initial curve segments.

        self._initialize_curves()

        # While we have more than one curve segment, greedily connect segments.

        while len(self.curves) > 1:

            self._connect_curves()

        # When we are left with only one curve segment, it is our greedy guess.

        return self.curves[0]

    def _initialize_curves(self):
        '''
        Do the initial greedy pairing.
        '''

        n_processed = 0

        begin_pts = []
        end_pts = []

        # For each vertex not added to a pair so far, find its nearest neighbor out of
        # the vertices that haven't been put in a pair so far.
        # For ease of determining which ones haven't been paired so far, we switch the paired
        # vertices to occur at the beginning of the array, so that all of the vertices to
        # the right of our current position are exactly the vertices that haven't been
        # processed so far.

        while n_processed < self.n_vertices:

            # Get the unpaired vertices to the right of our current position.

            candidate_start = n_processed + 1
            candidate_verts = self.vertices[candidate_start :]

            # Find the candidate that gives the shortest connection.

            distances = np.linalg.norm(self.vertices[n_processed] - candidate_verts, axis = -1)
            partner_i = np.argmin(distances, axis = 0) + candidate_start

            # Swap the chosen candidate to be directly after the current position.

            self._swap_vertices(n_processed+1, partner_i)

            # The pair gives a new curve; update the list of curves, list of beginning points,
            # and the list of end points.

            new_curve = np.array([self.vertices[n_processed], self.vertices[n_processed + 1]])
            self.curves.append(new_curve)
            begin_pts.append(self.vertices[n_processed])
            end_pts.append(self.vertices[n_processed + 1])

            # We processed a pair so we need to advance by 2 (recall that the next position is
            # now the chosen candidate).

            n_processed += 2

        self.end_pts = {'begin' : np.array(begin_pts),
                        'end' : np.array(end_pts)}

    def _connect_curves(self):
        '''
        Go through the curves in order and greedily connect them to another curve after their
        position in such a way that gives the shortest connection. We minimize the nonuniformity in
        the segment size, but when there is an odd number of segments there will be segments of
        different sizes.
        '''

        n_curves = len(self.curves)
        curve_i = 0

        # For each curve we find the shortest connection for all curves in a position
        # of our list of curves that occurs after the current curve. We do this to
        # try to keep the curves all of the same size (although they won't necessarily be).
        # We iterate until curve_i == n_curves - 2, because we want to make sure that
        # there are atleast two curves left to join.

        while curve_i < n_curves - 1:

            source_end_pt, destination_i, dest_end_pt = self._find_shortest_connection(curve_i)
            self._connect_source(source_end_pt, curve_i, destination_i, dest_end_pt)
            self._remove_curve(destination_i)

            curve_i += 1
            n_curves -= 1

    def _connect_source(self, source_end_pt, source_i, destination_i, dest_end_pt):
        '''
        The connection is made so that the new connected curve always starts at one of the endpoints
        of the source and ends at one of the endpoints of the destination.

        The new curve replaces the curve at self.curves[source_i]. This function will not
        delete/remove the other curve; it will need to be removed using self._remove_curve().

        Parameters
        ----------
        source_end_pt : String
            Should be either 'begin' or 'end' to indicate which side of the source curve the
            connection should be made.

        source_i : Int
            The index of the source curve for the connection.

        destination_i : Int
            The index of the destination curve for the connection.

        dest_end_pt : String
            Should be either 'begin' or 'end' to indicate which side of the destination curve the
            connection should be made.
        '''

        # When the we connect to the beginning of the source curve, then we need to flip its order.

        if source_end_pt == 'end':

            source_curve = self.curves[source_i]

        else:

            source_curve = np.flip(self.curves[source_i], axis  = 0)
            self.end_pts['begin'][source_i] = self.end_pts['end'][source_i]

        # When we connect to the end of the destination curve, then we need to flip its order.

        if dest_end_pt == 'begin':

            dest_curve = self.curves[destination_i]
            self.end_pts['end'][source_i] = self.end_pts['end'][destination_i]

        else:

            dest_curve = np.flip(self.curves[destination_i], axis = 0)
            self.end_pts['end'][source_i] = self.end_pts['begin'][destination_i]

        new_curve = np.concatenate([source_curve, dest_curve], axis = 0)
        self.curves[source_i] = new_curve

    def _remove_curve(self, curve_i):
        '''
        Remove a curve, updating the list of curves and list of endpoints.

        Parameters
        ----------
        curve_i : Int
            The index of the curve to remove.

        '''

        del self.curves[curve_i]

        for end_pt in ['begin', 'end']:

            self.end_pts[end_pt] = np.delete(self.end_pts[end_pt], curve_i, axis = 0)

    def _find_shortest_connection(self, source_i):
        '''
        Find the curve after the given curve at index source_i that will give the shortest
        connection to the curve at self.curves[source_i]. Note, the connection only
        considers connecting endpoints.

        Parameters
        ----------
        source_i : Int
            The index of the curve to consider connecting to other curves.

        Returns
        -------
        best_source_end_pt : String
            best_source_end_pt is either 'begin' or 'end' to indicate which source endpoint
            should be used for the connection.

        best_destination_i : Int
            best_destination_i is the index of the curve that we should connect the source curve to.

        bestDestEndPt : String
            Either 'begin' or 'end' to indicate which endpoint of the destination curve to connect
            to.
        '''


        # A negative best_distance indicates that we haven't found any distances yet.

        best_dist = -1

        best_source_end_pt = ''
        best_destination_i = -1
        best_dest_end_pt = ''

        # Loop over beginning and ending for the endpoints of source and destination.

        for source in self.end_pts:

            source_vertex = self.end_pts[source][source_i]

            for destination in self.end_pts:

                # To try to keep curve size uniform, we only consider connecting
                # to curves after the source curve (which shouldn't have been connected yet
                # in this pass).

                candidates = self.end_pts[destination][source_i + 1 :, : ]
                distances = np.linalg.norm(source_vertex - candidates, axis = -1)

                min_dist_i = np.argmin(distances)
                dist = distances[min_dist_i]

                # If we have a new best distance then record needed info.

                if best_dist < 0 or dist < best_dist:
                    best_dist = dist
                    best_source_end_pt = source

                    # Make sure to account for the fact that indices of candidates is offset
                    # from indices in self.curves.

                    best_destination_i = min_dist_i + source_i + 1
                    best_dest_end_pt = destination

        return best_source_end_pt, best_destination_i, best_dest_end_pt

    def _swap_vertices(self, i, j):
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
    Normalize vertices and make our intial greedy guess as to the solution of the Traveling
    Salesman Problem. If there is an odd number of vertices, then our greedy guess will
    drop the last vertex.

    Parameters
    ----------
    vertices : Numpy array of shape (n_vertices, 2)
        The xy-coordinates of the vertices.

    Returns
    -------
    Numpy array of shape (newNVertices, 2)
        The xy-coordinates of our processed vertices.
    '''

    vertices = normalize_vertices(vertices)
    guesser = GreedyGuesser3()
    vertices = guesser.make_guess(vertices)
    vertices = np.roll(vertices, 100, axis = 0)

    return vertices

def normalize_vertices(vertices):
    '''
    Normalize the xy-coordinates by scaling by the reciprocal of the largest y-coordinate.

    Parameters
    ----------
    vertices : Numpy array of shape (n_vertices, 2)
        The xy-coordinates of the vertices.

    Returns
    -------
    Numpy array of shape (n_vertices, 2)
        The normalized xy-coordinates of the vertices.
    '''

    scale = np.amax(vertices[:, 1])
    vertices = vertices.astype('float')
    vertices[:, 0] = vertices[:, 0] / scale
    vertices[:, 1] = vertices[:, 1] / scale

    return vertices
