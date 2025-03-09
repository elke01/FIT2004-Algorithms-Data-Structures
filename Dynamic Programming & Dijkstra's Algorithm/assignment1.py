import heapq

def fuse(fitmons):
    """
    Function description: This function returns a integer representing the cuteness score of the final fitmon. This function utilises Dynamic Programming to save overlapping subproblems 
    and reuse it to find the best combination(highest cuteness score) of fused fitmons.

    Approach description: We will first initialise a 2 dimensional array with respect to the number of fitmons from the input for the memoization. The diagonal entry of the memory will have the 
    cuteness score of each input fitmons. Then we will move to the upper diagonal entry and fill it with the cuteness score result of fusing 2 fitmons, those 2 fitmons will be the fitmons from the left and bottom of
    the location we're about to fill. After all the diagonal entry is filled, we move to the diagonal entry above the upper diagonal entry, and do the same, but since the number of fitmons fused will be >2, there will be several
    combinations to consider from all the fitmons at the left and below the location to fill, and out of all the possible combinations there, we choose the one that give the highest cuteness score to fill thsi location with, 
    we can do this because the left and right affinity score of any of the resulting combinations will be the same, so the highest cuteness score will benefit us more. We repeat this process until we reach the top right corner of the memory,
    and this will have the value of the highest cuteness score of all combinations of fusing.
    
    Input: 
        fitmons: a nested list, with inner list representing a fitmon.
        
    Output: An integer, representing the highest cuteness score after fusing.
        
    Time complexity: O(N^3), where N is the length of input "fitmons" (number of fitmons).

    Time complexity analysis :  Given N is the length of input "fitmons" (number of fitmons).
                                Making "memory" will cost O(N^2), and initialising the diagonal entry with each fitmons cost O(N).
                                Finding the best combination will need nested for loop. The outer for loop will only run until N+1, so it is O(N).
                                The middle for loop will also run with a number bounded by N, so also O(N).
                                And the most inner for loop will run until j, where j depends on the outer and middle current looping, but the number will not exceed N, so also O(N).
                                Therefore O(N^2) + O(N) + O(N)*O(N)*O(N) = O(N^3)

    Auxilary Space complexity: O(N^2), where N is the length of input "fitmons" (number of fitmons)

    Auxilary Space complexity analysis: Given that N is the length of input "fitmons" (number of fitmons).
                                            Variable "memory" which is used for the memoization of the dynamic preogramming is a nested list with length of N and
                                            each inner list also have a length of N.
                                        Therefore O(N*N) = O(N^2).
    """
    N = len(fitmons)
    # Initialise the memory to save overlapping problems
    memory = [[float("-inf")] * N for _ in range(N)] #O(N^2) memory


    # Save each fitmon cuteness score as base case in the diagonal position of the memory
    for i in range(N): #O(N)
        memory[i][i] = fitmons[i][1]

    # Finding the best combination of fitmons
    for fitmon_no in range(2, N + 1): # O(N), fitmon_no is the total number of fitmons fused in the memory location we are about to fill(minimum 2)
        for i in range(N - fitmon_no + 1): # O(N), i is the starting index of the memory location we are about to fill
            j = i + fitmon_no - 1 # j is the ending index of the memory location we are baout to fill
            for k in range(i, j): # O(N), This will run for the number of combinations we can have in location memory[i][j]
                # Calculate the new fitmon cuteness score
                new_cuteness = int(fitmons[k][2] * memory[i][k] + memory[k + 1][j] * fitmons[k+1][0])

                # If the new fitmon have larger cuteness score than the current fitmon in memory, update the memory
                if new_cuteness > memory[i][j]:
                    memory[i][j] = new_cuteness


    # The best final fitmon with the best fusing combination will be at the top right position of the memory (since we fill the memory diagonally until the top right corner)
    return memory[0][len(fitmons) - 1]
        
#==========================================================

class TreeMap:
    def __init__(self, roads, solulus):
        """
        Function description: This function initialise 2 list that represents a graph given the input "roads" that represents the edges, one list/graph with normal directed edges
                                and the other with opposite direction directed edges.

        Approach description: Given the edges (roads) from the input, this function will run through the edges and find the maximum vertex (tree id) to know how large the graph list should be initialised to.
                                Then it will run through the edges (roads) again to fill 2 graphs with the edges, where each index of the graph list represents the vertex (tree id)
                                and in that index position is a list of tuples representing the outgoing edges from that vertex. The 2 graphs to fill will be one for normal directed edges as what is given in the 
                                input (roads) and other is where the directed edges are on opposite directions, with an additional 1 vertex with no outgoing edges.

        
        Input: 
            roads: an list of tuples, representing edges of a graph.
            solulus: a list of tuples, representing special edges (solulu tree) of a graph.
            
        Output: None.
            
        Time complexity: O(|T| + |R|), where R is the number of tuples in input list "roads" and T is an integer of the largest vertex id in input list "roads".

        Time complexity analysis :  Given R is the number of tuples in input list "roads" and T is an integer of the largest vertex id in input list "roads".
                                    Since we will loop through the input list (roads) to find the largest vertex id and fill the graph, then the complexity will be O(|R|) 
                                    and since we are initialising a nested list of size number of vertices/trees for the graph, the complexity is O(|T|). 
                                    Therefore O(|R|) + O(|T|) + O(|T|) + O(|R|) = O(|T| + |R|).

        Auxilary Space complexity: O(|T| + |R|),  where R is the number of tuples in input list "roads" and T is an integer of the largest vertex id in input list "roads".

        Auxilary Space complexity analysis: Since we need to create 2 list of size T, so O(|T|) and the total number of tuples inside the lists is equals to the number of tuples
                                            from the input list "roads", then O(|R|).
                                            Therefore O(2|T|) + O(2|R|) = O(|T| + |R|).
        """
        # Variables to record the maximum tree id and solulu trees
        self.max_tree_id = 0
        self.solulus = solulus

        #Finding the largest tree id
        for road in roads: # O(R)
            v1, v2, w = road
            max_id = max(v1, v2)
            if max_id > self.max_tree_id:
                self.max_tree_id = max_id

        # Initialising the normal graph and inverted graph lists
        self.graph = [None]*(self.max_tree_id + 1) #O(T+R) memory
        self.inverted_graph = [None]*(self.max_tree_id + 2) #O(T+R) memory

        # Initialise all roads to the normal graph and inverted graph
        for road in roads: #O(R)
            v1, v2, w = road
            if self.graph[v1] == None:
                self.graph[v1] = []
                self.graph[v1].append((v2, w))
            else:
                self.graph[v1].append((v2, w))

            if self.inverted_graph[v2] == None:
                self.inverted_graph[v2] = []
                self.inverted_graph[v2].append((v1, w))
            else:
                self.inverted_graph[v2].append((v1, w))


    def escape(self, start, exits): 
        """
        Function description: This function returns a tuple containing an integer and list representing the time taken from start tree to exit tree while destroying solulu tree at least and at most 1 time and the trees visited, or None when no path exist. 
                                This function uses djikstra algorithm to find the nearest exit from a starting tree in the TreeMap while destroying at least and at most 1 solulu tree along the way.

        Approach description: This function uses dijkstra algorithm to find the shortest route from the starting tree "start" to any solulu tree in the normal graph. Then it adds a new vertex (special vertex)
                                to the inverted graph and give outgoing edges from the special vertex to every exit vertex with weight 0. It then uses dijkstra again to find the fastest route from this special vertex to any of the
                                destination vertex of the solulu tree teleportation.
                                At the end it will find the best of solulu tree to be taken considering the fastest route from the staring vertex to the solulu tree, the time taken to destroy the solulu tree and
                                the fastest route from any of the exits(via special vertex) to the destination tree of solulu tree teleportation.
                                After deciding on the best solulu tree, it will traverse through the "parent" array returned by the dijsktra algorithm to track which trees are visited, then this function
                                will return a tuple with integer of the time taken from starting tree to the exit and a list of all visited trees in order.
                                If there is no route that involves a destruction of solulu tree, then this function returns None.

        
        Input: 
            start: an integer indicating the starting tree/vertex 
            exits: a list of integers, indicating the exit trees/vertices.
            
        Output: A tuple containing an integer and list or None.
            
        Time complexity: O(|R|*log(|T|)), where R is the number of tuples in global variable "graph" representing the number of Roads in the TreeMap and T is the length of the global variable "graph"
                                        representing the number of Trees in the TreeMap.

        Time complexity analysis :  Given R is the number of tuples in  global variable "graph" which represents the number of Roads and T is the length of the global variable "graph"
                                        representing the number of Trees in the TreeMap.
                                    We run dijkstra twice with complexity O(|R|*log(|T|)), then we loop through the exits and solulu arrays
                                    which both will at worst have a length of T (all trees are exits and solulu trees) and we also run through an array of size T
                                    from the output of the dijkstra algorithm 2 times and reverse it 1 time to find the visited Trees which at worst will be O(|T|) (all Trees are visited)
                                    Therefore O(|R|*log(|T|) + |T| + |R|*log(|T|) + |T| + |T| + |T|) = O(|R|*log(|T|))

        Auxilary Space complexity: O(|T| + |R|),  where R is the number of tuples in global variable "graph" representing the number of Roads in the TreeMap and T is the length fo the global variable "graph"
                                        representing the number of Trees in the TreeMap.

        Auxilary Space complexity analysis: Djiktra algorithm/function we use have an aux space complexity of O(|T| + |R|) and this escape function have an array "best_path" and 4 variables (p1,d1,p2,d2) which
                                            stores the returned value of dijkstra where all of them have a length bounded by T.
                                            Therefore O(|T| + |R| + |T| + |T| + |T| + |T| + |T|) = O(|T| + |R|).
        """
        # Get the fastest time from starting tree to all tree, specifically all solulu trees.
        p1, d1 = self.dijkstra(start, self.graph, len(self.graph))

        #Create a new tree for the inverted graph and give outgoing edges with weight 0 from the new tree to all exit trees.
        for exit in exits:
            if self.inverted_graph[self.max_tree_id+1] == None:
                self.inverted_graph[self.max_tree_id+1] = []
            self.inverted_graph[self.max_tree_id+1].append((exit, 0))

        # Get the fastest time from the newly added tree in the inverted graph to all other trees, specifically solulu tree destination tree.
        p2, d2 = self.dijkstra(self.max_tree_id+1, self.inverted_graph, len(self.inverted_graph))

        # Initialise the special vertex back into having no outgoing edges for the next escape function call
        self.inverted_graph[self.max_tree_id+1] = None

        # Variable to save overall best path
        shortest_time = float("inf")
        solulu_to_take = None

        # For every solulu trees, check their fastest time to starting tree and fastest time of the solulu destination tree to any of the exit
        # Then accumulate the time obtained above + the teleportation time, do this until the fastest time for any solulu tree is obtained
        for solulu in self.solulus: # O(V) at worst all when all trees are solulu trees.
            v1, w, v2 = solulu
            if shortest_time > d1[v1] + w + d2[v2]:
                shortest_time = d1[v1] + w + d2[v2]
                solulu_to_take = solulu

        # If there is a route including the destruction of solulu tree.
        if solulu_to_take != None:
            start_solulu, w, destination = solulu_to_take
            v1 = start_solulu
            v2 = destination
            best_path = []

            # Find the visited trees from the solulu tree taken to the starting tree
            while v1 != start: #-> O(V) at worst, visit all nodes
                best_path.append(v1)
                v1 = p1[v1]

            # Append the starting tree to the list of trees visited
            best_path.append(start)
            # Reverse the list, since we are adding from the last tree, not the first tree to be visited.
            best_path = list(reversed(best_path)) #-> O(V) at worst, all nodes visited
            # If the solulu tree teleport us back to the same tree, remove the solulu tree from the record of visited trees (avoid duplicates)
            if start_solulu == destination:
                best_path.pop()


            # Find the visited tree from the newly added vertex to the destination solulu tree taken
            while v2 != self.max_tree_id+1: #-> O(V) at worst, visit all nodes
                best_path.append(v2)
                v2 = p2[v2]

            return (shortest_time, best_path)
                
        # If there is no route including the destruction of solulu tree then return None
        return None


    def dijkstra(self, start, graph, n):
        """
        Function description: This function returns a tuple containing 2 lists and each list contain integers. These 2 lists contain the time taken to go from start vertex to all vertices and refrence of the parent
                                of each vertices at which the shortest time is obtained. This function utilises MinHeap to find the shortest distance
                                from a starting vertex "start" from the input to all other vertices. This is a dijkstra algorithm.

        Approach description: We will first initialise 2 array of size "n" from input to record the shortest distance from starting vertex to all other vertices "distances", the parent of 
                                each vertices at which the shortest distance is obtained "parent".
                                Then we will initialise the MinHeap and insert the first item which is the starting vertex and the minimum distance from the starting vertex itself
                                which is zero, we also initialise the "distance" array for this start vertex which will also be zero.
                                Then as long as there is an item in the minheap, we will serve the item(vertex) from the min heap with the lowest distance to the starting vertex. 
                                If the served vertex item have the same record we have in "distances" array (Not outdated/the shortest path to that vertex) we will see all outgoing edges from the served vertex, 
                                then we should check if the weight of the edge + the distance where the edge is going from (parent) 
                                is shorter than the one we have in our record "distances", if yes then we update the value in the "distances" with this new shorter value then push this new shorter distance to the vertex.
                                After the minheap is empty, we will obtain the shortest distance from the "start" vertex to all other vertices and the parent for each vertices, so we return these list.

        
        Input: 
            start: an integer (indicating the starting vertex)
            graph: a list of lists that contains tuples (representing the graph)
            n: an integer (the number of vertices)
            
        Output: A tuple containing 2 lists.
            
        Time complexity: O(|R|*log(|T|)), where T is the integer input "n" representing the number of Tree in the TreeMap and R is the number of tuples in input "graph"
                                        representing number of Roads in the TreeMap.

        Time complexity analysis :  Given R is the number of tuples in input array "graph" (number of Roads) and T is the integer "n" (number of Tree).
                                    The while loop will end only when all Roads(R) which is edges has been checked, and for each checking we will pop an item
                                    from the MinHeap which have a complexity of O(log(|R|)) at worst since the number if items in the MinHeap will be bounded by
                                    the number of edges(R). This part will be O(|R|*log(|R|)).
                                    Then we have a for loop inside the while loop, but since we have a pre-condition before running the for loop which is the current popped vertex is not outdated, therefore the inner for loop which is O(T)
                                    will not run for R times(the number of times the while loop runs) but it will only run for T times. And we might push an item to the Minheap which is O(log(R)) in the for loop.
                                    This part will be O(|T|^2*log(|R|)).
                                    Note: In simple dense graph number of edges(R) is at most Vertex^2(|T|^2), so log(|R|) = log(|T|^2) = 2log(|T|), meaning O(2log(|T|)) = O(log(|T|)).
                                    Therefore, given R = T^2(bounded by), O(|R|*log(|T|)) + O(|T|^2*log(|T|)) = O(|R|*log(|T|)) + O(|R|*log(|T|)) = O(|R|*log(|T|))

        Auxilary Space complexity: O(|T| + |R|),  where T is the integer input "n" representing the number of Tree in the TreeMap and R is the number of tuples in input "graph"
                                                representing number of Roads in the TreeMap.

        Auxilary Space complexity analysis:  "distances" and "parent" array have a length bounded by T which represents the number of Trees and "Minheap" have a length bounded by R which is the 
                                                number of tuples in input "graph" representing the number of Roads.
                                            Therefore O(|T|+|T|+|R|) = O(|T| + |R|).
        """
        # Variables to record the distances, parent and Minheap arrays
        distances = [float("inf")] * n # O(T) SPACE
        parent = [None] * n # O(T) SPACE
        Minheap = [(0, start)] # O(R) SPACE at max

        # Initialise the distance and parent of starting tree
        distances[start] = 0
        parent[start] = start

        # While there is item in the minheap
        while Minheap: # O(R)
            # Pop the item(tree) with least distance from the starting tree
            dist, vertex = heapq.heappop(Minheap) # O(log(T))

            if graph[vertex] != None and distances[vertex] == dist: # If the vertex we currently on (we popped) have outgoing edges and not outdated(We have visited the tree)
                for v, w in graph[vertex]: # O(T), Relaxed all outgoing edges 
                    if distances[v] > distances[vertex] + w: # If the total time needed by the parent tree + new road is shorter than the recorded time, update and push
                        distances[v] = distances[vertex] + w
                        parent[v] = vertex
                        heapq.heappush(Minheap, (distances[vertex] + w, v)) # O(log(T)), Push new shorter time to heap

        # Return the recorded parent and distances list
        return(parent, distances)
