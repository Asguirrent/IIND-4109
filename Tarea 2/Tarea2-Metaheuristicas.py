#%%
#----------------------------------------
# Import Information (given by ChatGPT)
#----------------------------------------
import re
import os
#print("Archivos en la carpeta actual:", os.listdir())
print("Directorio actual:", os.getcwd())
def transform_data(file_path):
    data = {
        "depot": None,
        "grafo": {
            "vertices": None,
            "aristas_req": [],
            "aristas_noreq": []
        },
        "vehicles": None,
        "capacity": None
    }
    
    with open(file_path, "r") as file:
        lines = file.readlines()
        
        for line in lines:
            line = line.strip()
            if line.startswith("NOMBRE"):
                data["nombre"] = line.split(":")[-1].strip()
            elif line.startswith("COMENTARIO"):
                data["comentario"] = int(re.findall(r'\d+', line)[0])
            elif line.startswith("VERTICES"):
                data["grafo"]["vertices"] = int(re.findall(r'\d+', line)[0])
            elif line.startswith("ARISTAS_REQ"):
                data["grafo"]["aristas_req_count"] = int(re.findall(r'\d+', line)[0])
            elif line.startswith("ARISTAS_NOREQ"):
                data["grafo"]["aristas_noreq_count"] = int(re.findall(r'\d+', line)[0])
            elif line.startswith("VEHICULOS"):
                data["vehicles"] = int(re.findall(r'\d+', line)[0])
            elif line.startswith("CAPACIDAD"):
                data["capacity"] = int(re.findall(r'\d+', line)[0])
            elif line.startswith("TIPO_COSTES_ARISTAS"):
                data["tipo_costes_aristas"] = line.split(":")[-1].strip()
            elif line.startswith("COSTE_TOTAL_REQ"):
                data["coste_total_req"] = int(re.findall(r'\d+', line)[0])
            elif line.startswith("LISTA_ARISTAS_REQ"):
                continue  # Salta la línea de encabezado
            elif line.startswith("("):
                match = re.findall(r'\d+', line)
                if match:
                    data["grafo"]["aristas_req"].append({
                        "origen": int(match[0]),
                        "destino": int(match[1]),
                        "coste": int(match[2]),
                        "demanda": int(match[3])
                    })
            elif line.startswith("DEPOSITO"):
                data["depot"] = int(re.findall(r'\d+', line)[0])
    
    return data


file_path = r"C:\Users\Usuario\OneDrive - Universidad de los andes\Documentos\ANDES (´,_ ,)\2025-I\VSCODE\T2MH\gdb\gdb2.dat"
data = transform_data(file_path)
#print(data)

#%%
#---------------------
# Creating a graph
#---------------------
data = transform_data(file_path)
import networkx as nx
# Create the graph
def create_graph(data):
    G = nx.Graph()
    
    # Add nodes
    num_vertices = data["grafo"]["vertices"]
    G.add_nodes_from(range(1, num_vertices + 1))
    
    # Add edges
    for arista in data["grafo"]["aristas_req"]:
        G.add_edge(arista["origen"], arista["destino"], weight=arista["coste"], demanda=arista["demanda"])
    
    return G
graph = create_graph(data)
print("Graph G created with:")
print("Number of nodes: ", nx.number_of_nodes(graph))
print("Number of edges: ", nx.number_of_edges(graph))
#---------------------
# Dijkstra
#---------------------
dijkstra_path = nx.dijkstra_path(graph, 1, 5)
# print("Shortest path:", dijkstra_path)
dist_matrix = nx.floyd_warshall_numpy(graph)
dijkstra_cost = nx.dijkstra_path_length(graph, 1, 5)
# %%
#---------------------
# Heuristic
#---------------------
from collections import defaultdict
def carp_heuristic(data, graph):
    depot = data["depot"]
    capacity = data["capacity"]
    required_edges =sorted( [(arista["origen"], 
                       arista["destino"], 
                       arista["coste"], 
                       arista["demanda"]) for arista in data["grafo"]["aristas_req"]], key=lambda x: x[2])
    unserved = set(required_edges)
    routes = []
    all_edges = set()
    for u, v in graph.edges():
        all_edges.add((u, v))
        all_edges.add((v, u)) 
    arc_report = defaultdict(list)
    while unserved:
        #Creating a new route starting from the depot
        current_load = 0
        current_pos = depot
        total_cost = 0
        path_log = []
        #checking if the route has a feasible capacity
        while True:
            candidates = []
            #evaluating all the unserved arcs
            for (u, v, c, d) in unserved:
                #checking if is feasible to include that arc in the route
                #nomatter if is better to pass through that arc in terms of cost
                #the code is only considering the capacity
                if current_load + d > capacity:
                    #print(u,v)
                    continue
                
                #calculating costs 
                cost_to_u = nx.dijkstra_path_length(graph,current_pos,u)
                cost_to_v = nx.dijkstra_path_length(graph,current_pos,v)
                #assuming that each arc is bidirectional, the minimun cost is founded
                min_cost = min(cost_to_u, cost_to_v)
                
                #Greedy criteria
                ratio = d / (min_cost + c)
                candidates.append((-ratio, min_cost, u, v, c, d))
            #there are no candidates to consider because there are not feasible
            if not candidates:
                break   
            #choose the best candidate according to the best ratio of demand and cost
            candidates.sort()
            _, min_cost, u, v, c, d = candidates[0]
            
            #determine if is better to end in the node u or v according to dijkstra
            #and then move to the next best node/arc
            if min_cost == nx.dijkstra_path_length(graph,current_pos,u):
                deadhead_path = nx.shortest_path(graph, current_pos, u, weight='weight')
                #print(deadhead_path, current_pos,u)
                service_direction = (u, v)
            else:
                deadhead_path = nx.shortest_path(graph, current_pos, v, weight='weight')
                service_direction = (v, u)
            
            #Set the deadheading paths with demand=0
            for i in range(len(deadhead_path)-1):
                from_node = deadhead_path[i]
                to_node = deadhead_path[i+1]
                edge_data = graph.get_edge_data(from_node, to_node)
                path_log.append({
                    "tipo": "deadhead",
                    "desde": from_node,
                    "hasta": to_node,
                    "coste": edge_data["weight"],
                    "demanda": 0
                })
                edge=(from_node, to_node)
                arc_report[edge].append({"ruta": len(routes)+1, "tipo": "deadhead"})
                total_cost += edge_data["weight"]
            
            #Set the service paths with demand=d
            path_log.append({
                "tipo": "servicio",
                "desde": service_direction[0],
                "hasta": service_direction[1],
                "coste": c,
                "demanda": d
            })
            edge_servicio=(service_direction[0], service_direction[1])
            arc_report[edge_servicio].append({"ruta": len(routes)+1, "tipo": "servicio"})

            total_cost += c
            current_load += d
            current_pos = service_direction[1]
            #quit from unserved arcs
            unserved.remove((u, v, c, d)) 
        
        #return to depot
        return_path = nx.shortest_path(graph, current_pos, depot, weight='weight')
        for i in range(len(return_path)-1):
            from_node = return_path[i]
            to_node = return_path[i+1]
            edge=(from_node, to_node)
            edge_data = graph.get_edge_data(from_node, to_node)
            path_log.append({
                "tipo": "deadhead",
                "desde": from_node,
                "hasta": to_node,
                "coste": edge_data["weight"],
                "demanda": 0
            })
            arc_report[edge].append({"ruta": len(routes)+1, "tipo": "deadhead"})
            total_cost += edge_data["weight"]
        #add route
        routes.append({
            "ruta": path_log,
            "carga": current_load,
            "coste": total_cost
        })
    
    reporte_final = {}
    serviced_edges = set()
    for edge in arc_report:
        for evento in arc_report[edge]:
            if evento["tipo"] == "servicio":
                serviced_edges.add(edge) 
    for edge in all_edges:
        reversed_edge = (edge[1], edge[0])
        if reversed_edge in serviced_edges:
            continue
        if edge in serviced_edges:
            reporte_final[edge] = {"servido": True,"rutas": [e for e in arc_report[edge] if e["tipo"] == "servicio"]}
        else:
            reporte_final[edge] = { "servido": False, "rutas": arc_report.get(edge, [])}
    return routes, reporte_final
sln,reporte=carp_heuristic(data,graph)
# print(sln)
# print(len(sln))
# print(reporte)
# print(len(reporte))
# %%
#-------------------
#Results
#-------------------
def format_routes(routes, depot_node):
    formatted_routes = []
    for route_num, route in enumerate(routes, 1):
        current_route = str(depot_node)
        
        for step in route["ruta"]:
            tipo = step["tipo"]
            hasta = step["hasta"]

            current_route += f"-{tipo}-{hasta}"
        
        formatted_routes.append(f"Ruta {route_num}: {current_route}")
        formatted_routes.append(f"Costo ruta {route_num}: {route['coste']}")
    
    return formatted_routes
rutas_formateadas = format_routes(sln, 1)

# for ruta in rutas_formateadas:
#     print(ruta)
# %%
#------------
#instances
#------------
import time

def run_all_instances(base_path):
    results = {}
    
    for i in range(1, 24):
        file_path = f"{base_path}{i}.dat"
        instance_name = f"gdb{i}"
        
        try:
            start_time = time.time()
            data = transform_data(file_path)
            graph = create_graph(data)
            
            routes, reporte = carp_heuristic(data, graph)
            end_time = time.time()
 
            elapsed_time = end_time - start_time
            formatted_routes = format_routes(routes, data["depot"])
            total_cost = sum(route["coste"] for route in routes)
            
            results[instance_name] = {
                "tiempo": round(elapsed_time, 4),
                "rutas": formatted_routes,
                "costo_total": total_cost,
                "num_rutas": len(routes),
                "reporte": reporte
            }
            
            print(f"\n--- {instance_name} ---")
            print(f"Time: {results[instance_name]['tiempo']}s")
            print(f"Routes: {results[instance_name]['num_rutas']}")
            print(f"Total Cost: {results[instance_name]['costo_total']}")
            
        except Exception as e:
            print(f"\nError en {instance_name}: {str(e)}")
            results[instance_name] = {"error": str(e)}
    
    return results
file_path = r"C:\Users\Usuario\OneDrive - Universidad de los andes\Documentos\ANDES (´,_ ,)\2025-I\VSCODE\T2MH\gdb\gdb"
resultados = run_all_instances(file_path)
    
print("\nResumen final:")
for instance, data in resultados.items():
    if "error" not in data:
        print(f"{instance}: {data['num_rutas']} rutas, Costo: {data['costo_total']}, Tiempo: {data['tiempo']}s")
# %%
