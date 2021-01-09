

def tri_traversal(cost, heuristic, start_point, goals):
    l = []

    t1 = DFS_Traversal(cost,heuristic,start_point,goals)

    t2 = UCS_Traversal(cost,heuristic,start_point,goals)

    t3 = A_star_Traversal(cost,heuristic,start_point,goals)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l


#THIS IS CODE FOR THE DFS ALGORITHM

def DFS_Traversal_actual(cost,start_point,goals):
    l = [] 
    n=len(cost)    
    visited=[False for i in range(n)]
    stack=[]
    stack.append(start_point) 
    found=False
    
    l = []
    while (len(stack)!=0):  
            # Pop a vertex from stack and print it  
            s = stack[-1]  
            stack.pop() 
  
            l.append(s)
            visited[s] = True 
            
            for g in goals:
                if (g==s):
                    found=True
                    return l
            
            neighbours_orig=cost[s][1:]
            neighbours=neighbours_orig[::-1]
            index=len(neighbours)
            
            atleast=0
            for node in neighbours:
                if (node!=-1 and visited[index]==False): 
                    stack.append(index)
                    atleast=1
                    
                index-=1
            
            if(atleast==0):
                l.remove(s)
                if(len(l)>0):
                    prev=l[-1]
                    l.remove(prev)
                    stack.append(prev)

    if(found==False):
        return []

def DFS_Traversal(cost,heuristic,start_point,goals):
    m = DFS_Traversal_actual(cost,start_point,goals)
    return m








#THIS CODE IS FOR THE UCS ALGORITHM

def UCS_Traversal_actual(cost, start_point, goals):
    if (start_point == goals):
        return ([start_point], 0)
    l = []

    visited = set()

    n = len(cost)
    prev = [None for i in range(n - 1)]
    prev[start_point - 1] = None

    q = dict()
    q[start_point] = 0

    while len(q) != 0:
        sorted_q = sorted(q.items(), key=lambda x: x[1])
        lowest_cost = sorted_q[0]

        s, edge_cost = sorted_q[0]
        q.pop(s)
        visited.add(s)

        neighbours = cost[s][1:]

        index = 1
        for node in neighbours:
            if (node != -1):
                total_cost = edge_cost + neighbours[index - 1]
                parent = s
                child = index

                if (index not in visited and index not in q):
                    q[index] = total_cost
                    prev[index - 1] = s

                elif (index in q):
                    if (total_cost > q[index]):
                        prev[index - 1] = s
                        q[index] = total_cost

                    elif (total_cost == q[index]):
                        if (s < prev[index - 1]):
                            prev[index - 1] = s
                            q[index] = total_cost

            index += 1

    temp = goals

    if (prev[goals - 1] != None):
        while (temp != None):
            l.append(temp)
            temp = prev[temp - 1]
        l = l[::-1]

        if (l[0] == start_point):
            final_cost = 0
            for i in range(len(l) - 1):
                final_cost += cost[l[i]][l[i + 1]]

            return l, final_cost

        else:
            return [], -1
    else:
        return [], -1

def UCS_Traversal(cost,heuristic,start_point,goals):
    ucs = []
    for ele in goals:
        x = UCS_Traversal_actual(cost, start_point, ele)
        if (x[1] != -1):
            ucs.append(x)

    if (len(ucs) == 0):
        t2 = []
    else:
        minimum = ucs[0][1]
        lowest_path = ucs[0][0]

        for ele in ucs:
            if ele[1] < minimum:
                minimum = ele[1]
                lowest_path = ele[0]

        return lowest_path







#THIS IS CODE FOR THE A* ALGORITHM


def A_star_getChildren(cost, curr_node):
    children = dict()
    for i in range(len(cost[curr_node])):
        if cost[curr_node][i] > 0:
            children[i] = cost[curr_node][i]
    return children


def A_star_isGoal(current_node, goal, F, final_path):
    path = [current_node]
    while current_node in final_path:
        current_node = final_path[current_node]
        path.append(current_node)
    return path[::-1], F[goal]


def A_star_Traversal_actual(cost, heuristic, start_point, goal):
    G = dict()
    F = dict()
    G[start_point] = 0
    F[start_point] = heuristic[start_point]
    visted_nodes = set()
    visiting_nodes = set([start_point])
    final_path = {}

    if start_point == goal:
        return [start_point], 0

    while len(visiting_nodes) > 0:
        current_node = None
        current_cost = None
        for i in visiting_nodes:
            if current_node is None or F[i] < current_cost:
                current_cost = F[i]
                current_node = i
        if current_node == goal:
            m = A_star_isGoal(current_node, goal, F, final_path)
            return m
        visiting_nodes.remove(current_node)
        visted_nodes.add(current_node)

        for j in A_star_getChildren(cost, current_node):
            if j in visted_nodes:
                continue
            total_g = G[current_node] + cost[current_node][j]
            if j not in visiting_nodes:
                visiting_nodes.add(j)
            elif total_g >= G[j]:
                continue

            final_path[j] = current_node
            G[j] = total_g
            heuristic_value = heuristic[j]
            F[j] = G[j] + heuristic_value
    return [], 453468456794865

def A_star_Initial(cost,heuristic,start,goals):
    all_paths = dict()
    for i in goals:
        path, the_cost = (A_star_Traversal_actual(cost, heuristic, start, i))
        if the_cost in all_paths.keys():
            all_paths[the_cost] = min(path, all_paths[the_cost])
        else:
            all_paths[the_cost] = path
    temp = all_paths[min(list(all_paths.keys()))]
    return temp


def A_star_Traversal(cost, heuristic, start, goals):
    temp1 = A_star_Initial(cost,heuristic,start,goals)
    return temp1

