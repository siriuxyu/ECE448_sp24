# submitted.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# submitted should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement bfs function
    import queue
    path = []
    last_points = []
    visited = []
    dist_from_start = []
    infi = maze.size.x * maze.size.y + 1
    for i in range(maze.size.y):
        temp_last = []
        temp_visited = []
        temp_dist = []
        for j in range(maze.size.x):
            temp_last.append((-1,-1))
            temp_visited.append(False)
            temp_dist.append(infi)
        last_points.append(temp_last)
        visited.append(temp_visited)
        dist_from_start.append(temp_dist)
    dist_from_start[maze.start[0]][maze.start[1]] = 0
    
    my_queue = queue.Queue()
    temp_point = maze.start
    target_point = ()
    for i in maze.waypoints:
        target_point = i
        while temp_point != target_point:
            temp_neighbors = maze.neighbors_all(temp_point[0], temp_point[1])
            for i in range(len(temp_neighbors)):
                dist_origin = dist_from_start[temp_neighbors[i][0]][temp_neighbors[i][1]]
                dist_new = dist_from_start[temp_point[0]][temp_point[1]] + 1
                if visited[temp_neighbors[i][0]][temp_neighbors[i][1]] == True:
                    continue
                if dist_new < dist_origin:
                    dist_from_start[temp_neighbors[i][0]][temp_neighbors[i][1]] = dist_new
                    last_points[temp_neighbors[i][0]][temp_neighbors[i][1]] = temp_point
                if temp_neighbors[i] not in my_queue.queue:
                    my_queue.put(temp_neighbors[i])
            next_point = my_queue.get()
            visited[next_point[0]][next_point[1]] = True
            temp_point = next_point
    
    temp_path = maze.waypoints[-1]
    while temp_path != maze.start:
        path.append(temp_path)
        temp_path = last_points[temp_path[0]][temp_path[1]]
    path.append(maze.start)
    path.reverse()
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    #TODO: Implement astar_single
    import queue
    path = []
    last_points = []
    visited = []
    dist_from_start = []
    infi = maze.size.x * maze.size.y + 1
    for i in range(maze.size.y):
        temp_last = []
        temp_visited = []
        temp_dist = []
        for j in range(maze.size.x):
            temp_last.append((-1,-1))
            temp_visited.append(False)
            temp_dist.append(infi)
        last_points.append(temp_last)
        visited.append(temp_visited)
        dist_from_start.append(temp_dist)
    dist_from_start[maze.start[0]][maze.start[1]] = 0
    
    temp_point = maze.start
    priority_queue = queue.PriorityQueue()
    target_point = ()
    for i in maze.waypoints:
        target_point = i
        while temp_point != target_point:
            temp_neighbors = maze.neighbors_all(temp_point[0], temp_point[1])
            for i in range(len(temp_neighbors)):
                dist_origin = dist_from_start[temp_neighbors[i][0]][temp_neighbors[i][1]]
                dist_new = dist_from_start[temp_point[0]][temp_point[1]] + 1
                if visited[temp_neighbors[i][0]][temp_neighbors[i][1]] == True:
                    continue
                if dist_new < dist_origin:
                    dist_from_start[temp_neighbors[i][0]][temp_neighbors[i][1]] = dist_new
                    last_points[temp_neighbors[i][0]][temp_neighbors[i][1]] = temp_point
                temp_tuple = (dist_from_start[temp_neighbors[i][0]][temp_neighbors[i][1]] + 
                              abs(temp_neighbors[i][0] - target_point[0]) + abs(temp_neighbors[i][1] - target_point[1]),
                              temp_neighbors[i])
                if temp_tuple not in priority_queue.queue:
                    priority_queue.put(temp_tuple)
            next_point = priority_queue.get()
            visited[next_point[1][0]][next_point[1][1]] = True
            temp_point = next_point[1]
    
    temp_path = maze.waypoints[-1]
    while temp_path != maze.start:
        path.append(temp_path)
        temp_path = last_points[temp_path[0]][temp_path[1]]
    path.append(maze.start)
    path.reverse()
    return path

# This function is for Extra Credits, please begin this part after finishing previous two functions
def astar_multiple(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    return []
