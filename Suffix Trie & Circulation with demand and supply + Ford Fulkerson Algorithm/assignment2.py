import math
from collections import deque


class OrfFinder:
    """
    Problem Approach description:
    This problem utilises Suffix Trie to have a fast retrieval of substrings by the idea of substring = prefix of a suffix and character indexing.
    Create a suffix Trie from the genome string, and in each node in the trie that correspond to a specific character in the string, we will save the index + 1 where the character
    is located in the string inside a list (therefore this list will at worst have a length of len(genome)). 

    When looking for a substring of genome with prefix 'start' and suffix 'end':
    - look for the 'start' in the suffix trie then get the indexes list from the Node at the end of the searching (this indexes correspond to the index + 1 of the where the last character of 'start' 
        is located in the genome). If 'start' is not found, then immediately return empty string, meaning no such substring is found.
    - look for the 'end' in the suffix trie then get the indexes list from the Node at the end of the searching (this indexes correspond to the index + 1 of the where the last character of 'end' 
        is located in the genome). If 'end' is not found, then immediately return empty string, meaning no such substring is found.
    - if both 'start' and 'end' is found, then filter all the index of 'start' that comes before the last located 'end' and all index of 'end' that comes after the 
        first located 'start'. (Since all 'end' before the first 'start' and all 'start' after the last 'end' will definitely not be a part of the output)
    - For every filtered 'Start' we will get the substring up to the every 'end' that does not overlap or does not comes before the filtered 'start', once there is an overlap, 
        we will stop finding the combination of 'start' and 'end' for that specific 'start' and move to the next 'start'. (This ensure we do not try to match 'start' 'end' combinations
        that will not be a part of the output)

    """
    def __init__(self, genome) -> None:
        """
            Function description: This function initialise a suffix trie of the input string 'genome'. 

            Approach description: This function adds special character '$' to denote the end of the input string 'genome' and initialise the root for a suffix trie,
                                    it then calls the 'build_suffix' function to automatically build a suffix trie out of the input string 'genome'.  

            Input: 
                genome: a string representing a genome containing only uppercase character from A to D.
                
            Output: -
            
            Time complexity: O(N^2), where N is the length of the input string 'genome'.

            Time complexity analysis : Given that N is the length of the input string 'genome'.
                                        Adding a special character to the input string will be O(N) because it includes making a new string with length of N + 1.
                                        Then, the creation of object 'Node' for the root of Trie is O(1) and the function 'build_suffix' which is O(N^2).
                                        Therefore, O(N) + O(1) + O(N^2) = O(N^2)
            
                
            Aux complexity: O(N^2), where N is the length of the input string 'genome'.

            Aux complexity analysis: Given that N is the length of the input string 'genome'.
                                        And the function 'build_suffix' takes up O(N^2) space.
                                        Therefore O(N) + O(N^2) = O(N^2)		
	    """
        self.genome = genome + "$" #O(N)
        self.root = Node()

        self.build_suffix(self.genome) #O(N^2)

    def find(self, start, end):
        """
            Function description: Return a list of all substring in class variable 'self.genome' that have a prefix 'start' and suffix 'end' given that 'start' and 'end' doesn't overlap,
                                    empty list if there are no such substring. 

            Approach description: It will look for the substring 'start' and 'end' in the suffix trie, if any of them is not found, then immediately return empty list. 
                                    If both is found, it will filter all 'start' that comes before the last located 'end' and filter all 'end' that comes after the first located 'start',
                                    If after filtering, if filtered 'start' or 'end' does not exist, then immediately return empty list.
                                    If both filtered 'start' and 'end' exist, meaning we have an output, so find the combination of 'start' and 'end' that does not overlap
                                    and output the list of substrings.
                                    Since the combination searching starts from the first occurence of 'start' with the last occurence of 'end', if at any point of finding combination there is an
                                    overlap, immediately stop finding combination for that particular 'start' and move to the next located 'start'. This will ensure the time complexity does not exceeds O(V)
                                    , meaning we don't check for combinations that will not be in the output.

            Input: 
                start: A string, indicating the prefix of the substrings of 'self.genome'.
                end: A string, indicating the suffix of the substrings of 'self.genome'.
                
            Output: 
                output: A list containing all the substrings of 'self.genome' with prefix 'start' and suffix 'end' that doesn't overlap. Empty if there is no such substring.

            Justification of len(output) < V: Given V is the number of characters in the output list.
                                                Since 'start' and 'end' should not overlap and are non-empty, hence the length of each substring is at minimum 2 when len(start) and len(end) are both 1.
                                                By this fact, it is justified that len(output) will never be greater than or equal to V and is also in a way is independent of V(when V > 0, V can be short or infinitely long
                                                but the len(output) can be 1 or any number).
            
            Time complexity:  O(T + U + V), where T is the length of input 'start', U is the length of input 'end', and V is the number of characters in the output list.
                                Note: O(T + U + V) can be O(T + U + N^2) where N is the length of 'genome' in the case when all possible substrings of 'genome' is a part of the output (V = N^2).

            Time complexity analysis : Given that T is the length of input 'start', U is the length of input 'end', and V is the number of characters in the output list.
                                        Since 'get_indexes' function depends on the input parameter it is given, the first call using 'start' will have complexity of O(T),
                                        and the second call with 'end' will have a complexity of O(U).
                                        First and second loop: will iterate by a number bounded by V which is len(output) at worst (will be smaller than V).
                                        Triple nested for loop: 1. The outer and middle loop will together(combined) only iterate len(output) times which is smaller than V.
                                                                2. The inner most loop will iterate with a number of times == length of the current substring to be inserted to the output lets call it K. So, O(K).
                                                                    (K can change in every iteration of point 1 with K being from 0 to len('genome')).
                                                                3. Join the 'intermediate' list complexity is the same as point 2 above since the list have a length of K.  So, O(K).
                                                                4. Since for each iteration of 1st point which result to 1 of the output substring , we will iterate and use join with complexity of the length of the 
                                                                    current output substring (K). 
                                                                
                                                                Given in total we will have len(output)*K where K is the length of current substring to be isnerted to the output.
                                                                Therefore the triple nested for loop will have a complexity of O(len(output) * (K + K)) = O(V + V) = O(2V)
                                                                Note: 0 <= K <= len('genome')
                                                                
                                        Therefore, O(T) + O(U) + O(2V)  = O(T + U + V)                               
            
                
            Aux Space complexity: O(T + U + V), where T is the length of input 'start', U is the length of input 'end', and V is the number of characters in the output list.

            Aux Space complexity analysis: Given that T is the length of input 'start', U is the length of input 'end', and V is the number of characters in the output list.
                                        - 'beginning' and 'ending' are just referencing a list form a Node object, so O(1).
                                        - 'end_needed' and 'begin_needed' list will take up at max O(len(output)), will be smaller than V.
                                        - 'intermediate' list will have a length of maximum K, where K is the length of the substring to be insserted to the output list and K can vary from 0 to len('genome').
                                        - 'output' list will have a length that is less than V and independent of V, but the total number of character in each substring will be equal to V.
                                            So, O(K) * len(output) = O(V) (For the nested loop)
                                        Therefore, O(T) + O(U) + O(V) = O(T + U + V)		
	    """
        beginning = self.get_indexes(start) # Get all the end indexes + 1 where 'start' is found.
        ending = self.get_indexes(end) # Get all the end indexes + 1 where 'end' is found.

        if len(beginning) == 0 or len(ending) == 0: # If the 'start' or 'end' is not found, immediately no output, return empty list.
            return []
        
        # The first 'start' found in 'self.genome'
        first_start = beginning[0]
        end_needed = [] # At max space O(len(output)), less than V

        # Take all end indexes + 1 of 'end' that will be in the output
        # Starting from the very last possible 'end' to the first (once 'end' is overlapping with the first 'start', stop iterating) --> time is O(len(output)), less than V
        for i in range(len(ending) - 1, -1, -1):
            begin_of_end = ending[i] - (len(end)) # Change from end indexes + 1 of 'end' to starting index of 'end'

            # If not overlapping, then append to list of needed 'end'
            if first_start <= begin_of_end: 
                end_needed.append(begin_of_end)
                
            else: # Once overlapping, stop the iteration
                break

        if len(end_needed) == 0: # If there is no 'end' that can be a part of the output, immediately no output, return empty list
            return []
        

        # The last 'end' found in 'self.genome'
        last_end = end_needed[0]
        begin_needed = []
        # Take all start indexes + 1 of 'start' that will be in the output.
        # Starting from the first possible 'start' to the last (once 'start' is overlapping with the last 'end', stop iterating) --> time is O(len(output)), less than V
        for b in beginning:

            # If not overlapping, then append to list of needed 'start'
            if last_end >= b: 
                begin_needed.append(b)
                
            else: # Once overlapping, stop the iteration
                break

        if len(begin_needed) == 0: # If there is no 'start' that can be a part of the output, immediately no output, return empty list
            return []
        

        output = [] # space O(len(output)), less than V

        # If all checking above is passed, meaning there is an output, look for output and return it.
        # This double for loop combined will have a time complexity of exactly O(len(output)). It doesn't iterate over combinations that is not a part of the output with the use of break.
        for b in begin_needed: # for every possible 'start' from the first.
            for e in end_needed: # for every possible 'end' from the last.

                if b <= e: # if possible 'start' is not overlapping with possible 'end'.
                    intermediate = [] # Space is bounded by V

                    for i in range(b - len(start), e + len(end)): # O(V) at worst, when there is only 1 substring in the output.
                        intermediate.append(self.genome[i])

                    output.append("".join(intermediate)) # O(V) at worst, when there is only 1 substring in the output.
                    
                else: # Once the possible 'start' and 'end' overlap, break (ensure doesn't continue to iterate over combinations of 'start' and 'end' that will not be a part of the output)
                    break

        # Return all the substrings in a list (output).
        return output 

    def build_suffix(self, string):
        """
            Function description: This function acts as an intermediate to obtain all the suffix of the input string 'string' and insert it to the Suffix Trie. 

            Approach description: The function get all the starting index of the suffixes of the input string 'string'
                                    and call the 'insert_suffix' in each of the starting index to insert the suffix string(from starting index to end of string) to 
                                    the Suffix Trie. 

            Input: 
                string: A string to get the suffixes of.
                
            Output: -
            
            Time complexity:  O(N^2), where N is the length of the input string 'string'.

            Time complexity analysis : Given that N is the length of the input string 'string'.
                                        The function will iterate through the input string which will be O(N) and for each iteration, it will call function 
                                        'insert_suffix' which have a complexity of O(N).
                                        Therefore, O(N * N) = O(N^2).
                
            Aux Space complexity: O(N^2), where N is the length the of input string 'string'.

            Aux Space complexity analysis: Given that N is the length of the input string 'string'.
                                            Since the for loop iterate N times and for every iteration it will call 'insert_suffix' which takes up O(N), hence it will be O(N^2)
                                            Therefore, O(N) + O(N^2) = O(N^2)
		
	    """
        for i in range(len(string)): #O(N)
            self.insert_suffix(i, string) #O(N)


    def insert_suffix(self, start, string):
        """
            Function description: Insert a suffix string to the Suffix Trie. 

            Approach description: Iterate through the suffix Trie depending on the current character in the string to be inserted, if 
                                    the current Node doesn't have an existing Node in the location for the current character, it will create a new Node in
                                    that location. For each insertion of a suffix, it will insert characters starting from the index 'start' of the input string 'string' to the
                                    last index of the input string 'string'. 

            Input: 
                start: An integer indicating the starting index of the suffix in the input string 'string'.
                string: A string to iterate on when inserting the suffix string.
                
            Output: -
            
            Time complexity:  O(N), where N is the length the of input string 'string'.

            Time complexity analysis : Given that N is the length the of input string 'string'.
                                        For each substring of input string 'string' from index 'start' to the end of the input string 'string', it will iterate
                                        through the suffix trie to insert the substring to the suffix trie by each character. At worst, 'start' will be 0 
                                        when we initially insert the whole string to the suffix trie, which means there will be a total of N character to iterate through.
                                        Therefore, O(N).
            
                
            Aux Space complexity: O(N), where N is the length the of input string 'string'.

            Aux Space complexity analysis: Given that N is the length the of input string 'string'.
                                            At worst, the substring to be inserted to the Trie is the whole input string 'string', thie means we have to create N number of Nodes
                                            for each of the characters in 'string'. Since each Node can at worst have a total space complexity of O(N), it may seem like the total space complexity
                                            here will be O(N^2), but each Node don't and will not all use up O(N) space, since each insertion of suffix will be shorter than the previous,
                                            this fact means that the Node nearest to the leaves will always take up a space that is less than or equal to the Node above it, hence this makes the space complextity
                                            still bounded by O(N).
                                            Therefore, O(N).
		
	    """
        curr = self.root
        for j in range(start, len(string)): #O(N) at worst
            char = string[j]  
            # get index to insert character to the current Node's 'links'
            index = ord(char) - ord("A") + 1
            if char == "$":
                index = 0

            # If Node in the index of 'links' doesn't exist, create Node
            if curr.links[index] is None:
                curr.links[index] = Node() #O(N) space at worst (rarely happen)

            # Save the index where the char is located
            curr.end_indexes.append(j)
            # Move to next char/node
            curr = curr.links[index]


    def get_indexes(self, to_search):
        """
            Function description: Given a input string 'to_search', this function will try to get the list of all the index + 1 of the last character index where substring 'to_search' 
                                    is located in the input string of the OrfFinder Class which is 'genome'. If the substring 'to_search' is not found, then this function will just return
                                    empty list.

            Approach description: Starting from the root of the suffix trie, we will traverse through the Node/branch that corresponds to the current character of string 'to_search' 
                                    to be looked for. Once we reach the Node of last character, we return the Node's 'end_indexes' class variable. If at any time, a character from 
                                    the string 'to_search' is not found it its correspond branch/location, meaning the string 'to_search' is not in the Trie, hence we will return empty list.

            Input: 
                to_search = A substring to search in the suffix trie.
            
            Output: A list of integers indicating the indexes of each character in the OrfFinder Class input stirng 'genome' + "$" , therefore only contain integers from 0 to len(genome) + 1.
            
            Time complexity:  O(Y), where Y is the length of teh input string 'to_search'.

            Time complexity analysis : Given that Y is the length of the input string 'to_search'.
                                        At case when the substring (prefix of suffix) is found in the suffix trie, we need to traverse through the suffix trie until the 
                                        end of the input string 'to_search' with the length of Y.
                                        Therefore, O(Y)
            
                
            Total Space complexity: O(Y), where Y is the length of the input string 'to_search'.

            Total Space complexity analysis: Given that Y is the length of the input string 'to_search'.
                                                Since the function have O(1) auxilary space and the input have a length of Y.
                                                Therefore, O(1) + O(Y) = O(Y)

            Auxilary Space Complexity: Since this function only iterate through the Suffix Trie and return a class variable of 'Node',
                                        it doesn't consume any auxilary space.
                                        Therefore, O(1).
		
	    """
        # Starting from the root, look for the substring in the Suffix Trie
        curr = self.root
        for char in to_search:
            # get index to move to the current Node's 'links'
            index = ord(char) - ord("A") + 1
            if curr.links[index] is None: # If the current character is not found in the 'links', then the substring does not exist in the Suffix Trie, return empty list
                return []
            
            curr = curr.links[index]

        return curr.end_indexes # If the substrign exist in the Suffix trie, return the saved indexes list from the Node after the last character of the substring.

class Node:
    def __init__(self) -> None:
        """
            Function description: Initialise the 'Node' object with list class variable 'links' of length 5 to store characters from 'A' to 'D' and '$' for the suffix trie 
                                    and initialise 'end_indexes' list to save needed data for the problem. 

            Approach description: Initialise the 'Node' object with class variable 'links' of length 5 to store characters for the suffix trie and initialise 
                                    'end_indexes' to save needed data for the problem. 

            Input: -
            
            Output: -
            
            Time complexity:  O(1).

            Time complexity analysis : Since the the creation of 'self.links' will always have a length of 5, meaning it is constant.
                                        Therefore, O(1).
                

            Possible Total Space complexity: O(N), where N is the length of the OrdFinder input string parameter 'genome'.

            Possible Total Space complexity analysis: Given that N is the length of the OrdFinder input string parameter 'genome'.
                                                        Since 'self.links' always have constant length of 5, meaning O(1), and 'self.end_indexes' will at worst save every index of the 
                                                        OrdFinder input string 'genome' with length N, meaning O(N).
                                                        Therefore, O(1) + O(N) = O(N).

            Auxilary Space Complexity: For every insertion of a substring in the trie, 'self.end_indexes' may grow up to N
                                        where N is the length of the OrdFinder input string parameter 'genome', but it is initially empty
                                        , therefore it does not takes up any space in the initial initialization.
                                        Therefore O(1).
	    """
        self.end_indexes = [] # Bounded by length of input variable 'genome'

        # INDEX 0 IS LEAF ("$")
        self.links = [None] * 5


#=============================================================================================================

def allocate(preferences, officers_per_org, min_shifts, max_shifts):
    """
        Function description: This funtion utilises Ford Fulkerson and Circulation with demand and lower bound to determine whether an allocation for security officers(N) to a group of 
                                companies(M) in a certain month with 30 days where each day have 3 shifts that meets the shift preference of each officer,
                                the number officer needed by each company in each shifts, and the minimum and maximum number of days an officer are allowed to work for
                                can all be satisfied or not given that each officer can only work 1 shift a day. If all can be satisfied,
                                this function returns a nested lists of allocations, else it returns None.
                                Circulation with demand and lower bound is utilised here because we have demands from the company and lower bound for the min_shifts also a capacity, 
                                this concept will help us get a fast allocation by keeping track of the flow in the network flow.

        Approach description: Given the preferences of each officer, num officer needed by each company in each shift and min & max num days an officer should work,
                                check if the total number of worker needed in a day by all company is greater than the number of officer or if the number of company
                                is greater than the number of officer, if any is yes then immediately return None because the constraints given will not be satisfied.
                                Otherwise, we will construct a resolved circulation with demands and lower bound as a network flow with residual edges in the same graph.
                                Then we will run ford fulkerson on the network flow, once done, we will check if the flow from super source and flow to sink is maximised,
                                if yes then find the feasible allocation from the network flow and return it, else return None (no feasible allocation).   
                                - Checking for sum of demands = 0 is not needed since by the constraints and maths, this problem will always have 0 sum of demands.            

        Input: 
            preferences: A nested list representing the shift preference of each officer.
            officers_oer_org: A nested list representing the number of officer needed in each shift of each company.
            min_shifts: An integer representing the minimum number of days an officer must work. (0 to 30)
            max_shifts: An integer representing the maximum number of days an officer must work. (0 to 30)
        
        Output: A nested lists representing the feasible allocation or None (there is no feasible allocation).
        
        Time complexity:  O(M * N * N), where N is the number of officer and M is the number of company.

        Time complexity analysis : Given that N is the number of officer and M is the number of company.
                                    The first for loop will run M number of time, so O(M).
                                    'makeNetwork' function have a complexity of O(M * N)
                                    Initialisation of 'path' take O(M + N)
                                    The while loop in ford fulkerson will run MaxFlow times because in each agumentation of a path, the minimum flow is
                                    always 1. MaxFlow will be dependent of N because for each officer, they can only work for K days where min_shifts <= K <= max_shifts
                                    Therefore the total flow will be N*K where K is an integer that varies within a range. Therefore MaxFlow = N, so O(N) for the while loop.
                                    In each iteration of the while loop, we will run BFS and 'augment_path' function which both have complexity of O(|V| + |E|).
                                    From the explanation in function 'makeNetwork' we know that O(|E|) = O(N * M) and O(|V|) = O(N + M). 
                                    So, O(|V| + |E|) = O(N + M) + O(N * M) = O(N * M)
                                    'feasible' function has complexity of O(N + M) and 'find_allocation" have complexity of O(M * N).
                                    Therefore, O(M) + O(M * N) + O(M + N) + O(N)*O(N * M) + O(M * N) = O(M * N * N)
        
            
        Aux Space complexity: O(M * N * N), where N is the number of officer and M is the number of company.

        Aux Space complexity analysis: Given that N is the number of officer and M is the number of company.
                                    'makeNetwork' function takes O(N * M) space.
                                    'path' list takes O(N + M) space.
                                    'BFS' iterated by the while loop that runs for maxFlow times takes O(N)*O(N * M)
                                    'find_allocation' takes O(N * M) space.
                                    Therefore, O(N * M) + O(N + M) + O(N)*O(N * M) + O(N * M) = O(M * N * N)
	"""
    num_days = 30
    sink = 2 # sink located at index 2 of vertices list.
    last_officer = sink + len(preferences) # vertex id that corresponds to the last officer node.
    first_officer = sink + 1 # vertex id that corresponds to the 1st officer node.
    last_company = sink + len(preferences) + (len(preferences)*30) + (len(officers_per_org)* 3 * 30) # vertex id that corresponds to the last company day shift node.

    total_work = 0
    num_worker_needed = 0
    num_of_company = 0 # Number of company with at least 1 officer needed across 3 shifts
    for shifts in officers_per_org: #O(M)
        s1, s2, s3 = shifts
        num_worker_needed += (s1+s2+s3)
        total_work += ((s1+s2+s3) * num_days)
        if s1 == 1 or s2 == 1 or s3 == 1:
            num_of_company += 1
    
    # Check if there is enough worker to work in a day
    if len(preferences) < num_worker_needed:
        return None
    # Number of officer should be greater than number of company
    if num_of_company > len(preferences):
        return None

    # Make Nework Flow
    network = makeNetwork(preferences, officers_per_org, min_shifts, max_shifts, total_work) #O(M*N)
    # List to keep track of the augmenting path
    path = [None] * len(network.vertices) # O(|V|) = O(N+M)

    # Check for augmenting path from super source to sink and augment it if exist. (Ford Fulkerson)
    while network.BFS(path): # O(V + E) for bfs, while loop run for maxflow times (maxflow only increase by 1 for each iteration)
        # Augment path
        network.augment_path(path, 1) # O(V + E)

    # If all demands are met, return allocation list
    if network.feasible():
        return network.find_allocation(len(preferences), len(officers_per_org), first_officer, last_officer, last_company)
    
    return None # If demands are not met, return None


def makeNetwork(preferences, officers_per_org, min_shifts, max_shifts, total_work): 
    """
        Function description: Initialise a network flow for the officer company problem with a resolved demand and lowerbound, and adds the residual edges
                                immediately into the same graph. 

        Approach description: - Initialise a graph in the form of an adjacency list.
                                - Make the 0th index node to be the super source (to resolve the negative demands), the 1th index node to be the starting node, and the 2nd
                                    index node to be the sink (to resolve positive demands)
                                - 1th index node will have edges going out to each officer nodes with lowerbound min_shifts and capacity max_shifts. (Ensure each officer work for 
                                    number of days >= min_shifts and number of days <= max_shifts)
                                    This lower bound and capacity is then resolved by making each officer node have a negative demand of min_shifts and the edges to them will
                                    have capacity of max_shifts - min_shifts. Then 1st index node will have a positive demand of min_shifts*number of officer.
                                - Each officer nodes will have their own days nodes (30 nodes) and each officer will have an edge goint out to each of their days node
                                    with lower bound 0 and capacity 1 (ensure each officer only work once a day).
                                - Each company will have 90 nodes (30 days * 3 shifts) where each node represent a specific shift of the specific day of each company.
                                    And each of these nodes will have a positive demand of the number of employee the company want for the shift the nodes corresponds to.
                                    (Ensure each shift of each day will have the exact number of officer being asked for by the company)
                                - Each officer's days nodes will go to each company's 90 nodes where shift they prefer and the day they corresponds is the same with a
                                    lower bound of 0 and capacity of 1. (If officer 1 prefer shift 1, so day 1 node of officer 1 will have an edge to shift 1 of day 1 of 
                                    each company nodes, and same for day 2 until day 30)
                                - 1th index node's demand will then subtracted by the total number of officer needed accross all shifts of all company * 30 (total number of 
                                    officer needed to be allocated in 30 days)
                                - Resolve all negative demands (Nodes with possible negative demands will be the officers nodes and 1th index node) by making index 0th node to have
                                    edges to each node with negative demand having capacity of the absolute of the negative demand itself.
                                - Resolve all positive demands (Nodes with possible positive demands will be the each company's 90 days shifts nodes, 1th index node's will never have 
                                    positive demands because of the prechecking before this function is called) by adding edge from each node with negative demand to the 2nd index node(sink)
                                    with a capacity of the positive demand itself.

        Input: 
            preferences: A list of list representing the preference of each officer.
            officers_per_org: A list of list representing the number of officer needed in a shift by each company.
            min_shifts: An integer representing the minimum days an officer must work.
            max_shifts: An integer representing the maximum days an officer can work.
            total_work: An integer representing the total number of officer needed by all company accross all shift in 30 days.
            
        Output: graph: A Graph object representing the network flow created.
        
        Time complexity:  O(N * M), where N is the number of officer and M is the number of company.

        Time complexity analysis : Given that N is the number of officer and M is the number of company.
                                    The initialisation of the NetworkFlow object takes O(|V|) where |V| will be the number of vertices/nodes in the networkflow.
                                        Since we know the number of vertices will be 3 + N + 30N + 30*3*M = N + N + M = N + M, therefore  O(N + M).
                                    Nested for loop: The outer loop will iterate N number of times. And the middle loop will iterate 30 times to add edges from each 
                                                        officer node to their corresponding 30 day nodes.
                                                        While the 3 inner loop will iterate for 3M number of times to add edges from the officer's day nodes to each corresponding 
                                                        company's day shift nodes.
                                                        Therefore this nested for loop will take O(N * 30 * 3M) = O(N * M)
                                    The last for loop will iterate M number of times so O(M).
                                    Since 'insert' function take O(1), so it does not affect the complexity.
                                    Therefore, O(M) + O(N + M) + O(N * M) + O(M) = O(N * M)
        
            
        Aux Space complexity: O(N * M), where N is the number of officer and M is the number of company.

        Aux Space complexity analysis: Given that N is the number of officer and M is the number of company.
                                        The NetworkFlow object which initially only takes O(|V|) where |V| is the number of vertices in the graph but will
                                        now occupy O(|V| + |E|) space where |E| is the number of edges in the graph, because after this function finish 
                                        running, we added |E| edges to the NetworkFlow. Space occupied by |V| as explained above is O(N + M)
                                        Since in each iteration of the loop done below we add an edge, so the space the edges occupy will just be the same as the complexity of all
                                            the looping which is O(M) + O(N * M) = O(N * M)
                                        More specific explanation to |E| = O(N * M) will be, since we will make edges from each officer to their 30 day nodes,
                                            we will have 30N edges. Then we have edges from each of the officer's day nodes which is 30N nodes to a company's
                                            corresponding day node with the shift the officer prefer, at maximum an officer prefer 3 shifts, so for every officer's day
                                            nodes we will have 3 edges going out to all 3 shifts the corresponding day nodes of each company, so 30N*3M edges.
                                            Then we will have edges from super source to starting and every officer nodes, so N. We will also have edges from every company
                                            nodes to sink, so 90M.
                                            In total we will have (N + 30N + 30N*3M + 90M)*2 including the residual edges which is why we times by 2.
                                            Therefore the space occupied by the |E| will be O((N + 30N + 30N*3M + 90M)*2) = O(N * M).
                                        Therefore, O(|V| + |E|) =  O(N + M) + O(N * M) = O(N * M)

        Fact obtained from above explanation: O(|E|) = O(N * M)
                                              O(|V|) = O(N + M)

		
	"""
    source = 0 # super source vertex index 0
    starting = 1 # starting vertex index 1
    sink = 2 # sink vertex index 2
    num_vertices = len(preferences) + (len(preferences)*30) + (len(officers_per_org)* 3 * 30) + 3 # number of nodes/vertices = N + 30N + 30*3*M + 3
    first_officer = sink + 1 # vertex id that corresponds to the 1st officer node.
    last_officer = sink + len(preferences) # vertex id that corresponds to the last officer node.
    first_company = sink + len(preferences) + (len(preferences)*30) + 1 # vertex id that corresponds to the 1st company day shift node.
    last_company = sink + len(preferences) + (len(preferences)*30) + (len(officers_per_org)* 3 * 30) # vertex id that corresponds to the last company day shift node.

    graph = NetworkFlow(num_vertices) #O(|V|)

    # minimum shifts of officer * number of officer
    c = min_shifts*len(preferences)
    # Making edge from super source node to starting node
    graph.insert(source, starting, total_work-c)

    add = 0
    # Making edge from source node to each officer node and starting node to each officer node
    for i in range(first_officer, last_officer + 1): # O(N)
        graph.insert(source, i, min_shifts)
        graph.insert(starting, i, max_shifts-min_shifts)
        
 
        curr_officer = i - first_officer
        # each officer can only work at max 1 shift a day (edge from officers node to each officer's day node)
        for j in range((i + len(preferences)), (i + len(preferences) + 30)): #O(30) =  constant
            day = j - (i + len(preferences))
            j += add
            graph.insert(i, j, 1) # Add edge from officer's node to each officer's day node
            

            # Add edge from each officer's day node to the preffered shift of the corresponding day node of the company.
            s1, s2, s3 = preferences[curr_officer]
            if s1 == 1:
                for c in range((first_company + (day*3)), last_company + 1, 90): #O(M)
                    graph.insert(j, c, 1)
            if s2 == 1:
                for c in range((first_company + (day*3)) + 1, last_company + 1, 90): #O(M)
                    graph.insert(j, c, 1)
            if s3 == 1:
                for c in range((first_company + (day*3)) + 2, last_company + 1, 90): #O(M)
                    graph.insert(j, c, 1)
        add += (30 - 1)

    company = 0
    shift = 1
    # Make edge from each company day shift node to sink node
    for c in range(first_company, last_company + 1): #O(M)
        s1, s2, s3 = officers_per_org[company]
        if shift == 1:
            curr_shift = s1
        elif shift == 2:
            curr_shift = s2
        else:
            curr_shift = s3

        if curr_shift != 0: # If the number of officer needed is not 0 (if 0, no need to make edge to sink)
            graph.insert(c, sink, curr_shift)

        shift += 1

        # Go back to shift 1 after each company's day nodes
        if shift == 4:
            shift = 1

        # Move to next company to check the number of officers needed for each shift (in step of 90 because each company have 90 nodes)
        if ((c-first_company + 1) - (company*90)) == 90:
            company += 1

    return graph # Return the resulting graph


class NetworkFlow:
    def __init__(self, vertices) -> None:
        """
            Function description: A constructor to construct NetworkFlow object representing the network flow. 

            Approach description: A network flow represented as an adjacency list with OOP implementation. 
                                    Initialise each vertex object to it's corresponding location in the list of the adjacency list.
                                    Each index will be the id for the vertex in that index.

            Input: 
                vertices: An integer representing the number of vertices that will be in the network flow.
            
            Output: -
            
            Time complexity:  O(|V|), where |V| is the number from the input integer 'vertices'/number of vertices in the graph.

            Time complexity analysis : Given that |V| is the number from the input integer 'vertices'.
                                        Initialising the List of length N will take O(|V|).
                                        Placing the Vertex object to each index of the list will take O(|V|).
                                        Therefore, O(|V|) + O(|V|) = O(|V|)
            
                
            Aux Space complexity: O(|V|), where |V| is the number from the input integer 'vertices'/number of vertices in the graph.

            Aux Space complexity analysis:  Given that |V| is the number from the input integer 'vertices'/number of vertices in the graph.
                                                Initialising the the vertices in a list will take up |V| space.
                                                Therefore, O(|V|)
		
	    """
        self.vertices = [None] * vertices

        for i in range(vertices):
            self.vertices[i] = Vertex(i)


    def BFS(self, path):
        """
            Function description: A Breadth-First Search to find the augmenting path from source to sink. 

            Approach description: Traverse the graph in Breadth-First Search from the source to the sink of a graph and keep track of the path. Once the sink is 
                                    located, stop traversing and return True. If the sink is never found, return False.

            Input: 
                path: a list with length of number of vertices in the graph.
            
            Output: Boolean, indicating the success of finding the sink.
            
            Time complexity:  O(|V| + |E|) where |V| is the number of vertices in the graph and |E| is the number of edges in the graph.

            Time complexity analysis : Given that |V| is the number of vertices in the graph and |E| is the number of edges in the graph.
                                        Making the 'visited' list take O(|V|). Then this implementation of BFS will at worst visit all
                                        vertices once and all edges once, so O(|V|) + O(|E|)
                                        Therefore, O(|V|) + O(|V|) + O(|E|) = O(|V| + |E|)
            
                
            Aux Space complexity: O(|V|) where |V| is the number of vertices in the graph.

            Aux Space complexity analysis: Given that |V| is the number of vertices in the graph.
                                            The 'visited' list and Queue will at most have a length of number of vertices.
                                            Therefore O(|V|) + O(|V|) = O(|V|).
	    """
        source = self.vertices[0]
        visited = [False] * len(self.vertices) #All vertices initially unvisited
        que = deque([source]) # Start from source

        visited[source.id] = True # Source is visited
        path[source.id] = source.id

        while que:
            u = que.popleft()

            for edge in u.edges:
                if not visited[edge.to] and edge.capacity > 0: #if not yet visited and the capacity of edge is not zero
                    visited[edge.to] = True
                    path[edge.to] = edge # parent of next vertex is the current vertex
                    que.append(self.vertices[edge.to])
                    if edge.to == 2: #If reach sink (index = 2), then sink is found, exist an augmenting path, return True
                        return True
        
        return False # Sink is never found, no augmenting path, return False
        

    def insert(self, start, to, capacity):
        """
            Function description: Insert an Edge to the vertex object with id 'start' with capacity of 'capacity' and its corresponding residual edge with capacity of 0.

            Approach description: Insert an Edge to the vertex object with id 'start' to vertex with id 'to' with capacity of 'capacity' and indicate it as True (meaning it is a forward edge from the source/not residual)
                                    and add an Edge to the vertex object with id 'to' to vertex with id 'start' with capacity of 0 and indicate it as False(meaning it is a residual edge).
                                    Make each of the Edge created have a reference to each other as their class variable.

            Input: 
                start: An integer representing the the vertex id of where the edge should come from.
                to: An integer representing the vertex id of where the edge should go to.
                capacity: An integer representing the capacity of this edge from 'start' to 'to'.
                
            Output: -
            
            Time complexity:  O(1)

            Time complexity analysis : Since we are just accessing, creating Edge object and appending Edge object to a Vertex object,
                                        all is done in O(1).
                                        Therefore O(1).
            
                
            Aux Space complexity: O(1).

            Aux Space complexity analysis: Not creating anything that the size depends on input parameter.
                                            Therefore, O(1).
		
	    """
        vertex1 = self.vertices[start]
        vertex2 = self.vertices[to]
        forward = Edge(start, to, capacity, None, True) # Edge to be added
        backward = Edge(to, start, 0, forward, False) # The reverse edge of the edge to be added with capacity 0
        forward.reverse = backward

        vertex1.edges.append(forward) # Add edge to its corresponding source vertex
        vertex2.edges.append(backward) # Add residual edge to the destination vertex
    

    def augment_path(self, path, minFlow):
        """
            Function description: Augment the path of a network flow whenever an augmenting path exist. 

            Approach description: Starting from the sink, traverse the graph by the augmenting 'path' given in the input. In each traversal of an edge, we
                                    increase tha capacity of the residual edge by minFlow and decrease the capacity of the original edge by minFlow.
                                    Once reach the source, stop traversing. 

            Input: 
                path: A list of integers indicating the parent of each vertex (the path from sink to source).
                minFlow: An integer indicating the minimum flow of the augmenting path.
            
            Output: -
            
            Time complexity:  O(|V|), where |V| is the number of vertices in the graph.

            Time complexity analysis : Given that |V| is the number of vertices in the graph.
                                        This function will traverse throught the agumenting path and at worst, the augmenting path will
                                        need to visit all vertices.
                                        Therefore, O(|V|).
            
                
            Aux Space complexity: O(1)

            Aux Space complexity analysis: Since we are just traversing, accessing and updating values, this does not consume space.
                                            Therefore, O(1).
		
	    """
        curr = self.vertices[2].id

        while curr != self.vertices[0].id: # While not reachign source, traverse
            edge = path[curr]
            edge.capacity -= minFlow # Subtract the capacity of the original edge
            edge.reverse.capacity += minFlow # Increase the capacity of the reverse/residual edge
            curr = path[curr].begin

    
    def feasible(self):
        """
        Function description: Check whether after running ford fulkerson, there is a feasible flow that meets all the demand.
                                Return a boolean indicating feasible or not.

        Approach description: Check if all edges capacity going out from the super source is 0 (meaning the flow is maximum, there can be no more flow to be given out and the negative demand is met).
                                Check if all edges capacity going to the sink is 0 (meaning the flow is maximum, there can be no more flow to be given in and the positive demand is met).
                                If both the above condition is met, then there is a feasible solution to the problem (All demand is met) we return True, else we return False.

        Input: - 
        
        Output: Boolean, True to indicate it is feasible and False otherwise.
        
        Time complexity:  O(N + M), where N is the number of officer and M is the number of company.

        Time complexity analysis : Given that N is the number of officer and M is the number of company.
                                    The number of edges going out of the super source will be N + 1, and the edges coming to the sink will be 30*3*M (this is due to 
                                    the design of the network flow).
                                    Therefore, O(N) + O(M) = O(N + M)
        
            
        Aux Space complexity: O(1)

        Aux Space complexity analysis: Since this function only access value and do checking, it doesn't takes up space.
                                        Therefore O(1).
		
	    """
        source = self.vertices[0]
        sink = self.vertices[2]
        for edge in source.edges: #O(N)
            if edge.capacity != 0: # If there exist an edge from super source with a capacity, negative demand is not met
                return False
            
        for edge in sink.edges: #O(M)
            if edge.reverse.capacity != 0: # If there exist an edge to sink with a capacity, meaning positive demand is not met
                return False
            
        return True
    

    def find_allocation(self, num_officers, num_company, first_officer, last_officer, last_company):
        """
        Function description: Find the allocation for each officer to the day and shift of the provided company from the network flow. 

        Approach description: Create a allocation nested lists for the allocation timetable.
                                For each officer node, traverse through their edges that goes to their day nodes, if the capacity of the edge is 0, means the officer work
                                on that day. For the day node that the officer work for, find outgoing edges to the company day shift nodes with capacity of 0,
                                the node which this edge goes to indicate which shift and which company the officer will work for in that day.
                                After the above process, we will place '1' in the allocation list for that specific officer, on that specific company, and in the specific day
                                and shift. Then repeat for all officer.

        Input: 
            num_officer: An integer indicating the number of officer
            num_company: An integer indicating the number of company
            first_officer: An integer indicating the vertex id that corresponds to the 1st officer node.
            last_officer: An integer indicating the vertex id that corresponds to the last officer node.
            last_company: An integer indicating the vertex id that corresponds to the last company day shift node.
            
        Output: allocation: A nested lists representing the allocation.
        
        Time complexity:  O(M * N), where N is the number of officer and M is the number of company.

        Time complexity analysis : Given that N is the number of officer and M is the number of company.
                                    The initialisation of 'allocation' takes O(N * 30 * M) = O(M * N)
                                    3 nested loop: 
                                    The outer loop will run for N times, the middle loop will run for 30 times and the
                                    inner loop will run for 3*M times.
                                    Therefore,  O(M * N) + O(N * 30 * 3*M) = O(M * N)
        
            
        Aux Space complexity: O(M * N), where N is the number of officer and M is the number of company.

        Aux Space complexity analysis: Given that N is the number of officer and M is the number of company.
                                        The 'allocation' nested list will have length of N, with the next inner list with length of M,
                                        then the next inner list with length of 30 and the most inner list with length of 3.
                                        Therefore, O(N * M * 30 * 3) = O(M * N)
	    """
        num_days = 30
        allocation = [[[[0,0,0] for _ in range(num_days)] for _ in range(num_company)] for _ in range(num_officers)] #O(M*N)
        minus = 0

        # Start traversing from each officer nodes
        for i in range(first_officer, last_officer + 1): # O(N)
            officer_id = i - first_officer # officer vertex id
            officer_v = self.vertices[i] # officer vertex object

            for d in officer_v.edges: #O(30) for days
                if d.forward and d.capacity == 0: # If capacity is 0 and it is not a residual edge
                    
                    day = d.to # day vertex id
                    what_day = (day - (i + num_officers)) - minus # The day the officer work on, 0 to 29
                    day_v = self.vertices[day] # day vertex object

                    for shift in day_v.edges: #O(3*M) at worst (when prefer all shifts)
                        if shift.forward and shift.capacity == 0: # If capacity is 0 and it is not a residual edge

                            which_company = math.ceil(((last_company - shift.to) + 1) / 90)  * -1 # The company the officer work for
                            which_shift = ((last_company - shift.to + 1) % 3)
                            
                            if which_shift == 0: # if remainder 0 then work in 1st shift
                                which_shift = 0
                            elif which_shift == 2: # if remainder 2, then work in 2nd shift
                                which_shift = 1
                            else: # if remainder 1, then work in 3rd shift
                                which_shift = 2

                            allocation[officer_id][which_company][what_day][which_shift] = 1 # Place the allocation for the current officer

                            break # Stop iterating once found (officer work 1 shift a day)

            minus += (num_days - 1)

        return allocation # Return the allocation timetable


class Vertex:
    def __init__(self, id) -> None:
        """
        Function description: A constructor to construct Vertex object representing a vertex for the network flow. 

        Approach description: Save the vertex id of this vertex object and save all the outgoing edges from this vertex in a list called 'self.edges'. 

        Input: 
            id: An integer indicating the vertex id of this vertex.
            
        Output: -
        
        Time complexity:  O(1)

        Time complexity analysis : Just an assignment to class variable which is done in O(1).
                                    Therefore O(1)
        
            
        Aux Space complexity: O(1)

        Aux Space complexity analysis: Initially 'self.edges' is empty.
                                         Therefore, O(1).
		
	    """
        self.id = id
        self.edges = []


class Edge:
    def __init__(self, u, v, c, edge, forward) -> None:
        """
        Function description: A constructor to construct Edge object representing an Edge for the network flow. 

        Approach description: Save the vertex id of the source(u) and destination(v) Node with capacity (c). Save the object reference of the reversed edge (for residual).
                                Also have a 'self.forward' class variable which indicate if this is a forward edge from the source (not residual) or reversed edge (residual).

        Input: 
        u: An integer indicating the source vertex id of this edge.
        v: An integer indicating the destination vertex id of this edge.
        c: An integer indicating the capacity of this edge.
        edge: An Edge object that is the reverse edge of this edge.
        forward: A boolean which will indicate whether this edge is a residual edge or not.
        
        Output: -
        
        Time complexity:  O(1)

        Time complexity analysis : Just an assignment to class variable which all done in O(1).
                                    Therefore, O(1).
        
            
        Aux Space complexity: O(1)

        Aux Space complexity analysis: The edge object always store a constant amount of class variable which don't grow (integer/object/boolean).
                                        Therefore, O(1).
		
	    """
        self.begin = u #vertex from
        self.to = v #vertex to
        self.capacity = c #capacity
        self.reverse = edge # the reversed edge
        self.forward = forward # residual or not

