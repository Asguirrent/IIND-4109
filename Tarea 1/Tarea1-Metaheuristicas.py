import gurobipy as gp
from gurobipy import GRB
import itertools

model = gp.Model("3D_Knapsack_MaxVol")

# Parámetros
W= 5  # Tamaño de la mochila
D=5
H=5
I = list(range(12))  # 12 fichas
P_a = [(1,2,4), (1,4,2), (2,1,4), (2,4,1), (4,1,2), (4,2,1)]
P_b = [(2,2,3),(2,3,2),(3,2,2),(2,2,3),(2,3,2),(3,2,2)]
PP={}
for i in I:
    if i<6:
        PP[i]=P_a
    else:
        PP[i]=P_b
M = 10  # Big-M mayor que S

# Variables
s = model.addVars(I, vtype=GRB.BINARY, name="s")  # Selección de fichas
x = model.addVars(I, lb=0, ub=W, vtype=GRB.INTEGER, name="x")
y = model.addVars(I, lb=0, ub=H, vtype=GRB.INTEGER, name="y")
z = model.addVars(I, lb=0, ub=D, vtype=GRB.INTEGER, name="z")
o = {}  # Orientaciones
for i in I:
    orientaciones = P_a if i < 6 else P_b
    for k in range(len(orientaciones)):
        o[(i, k)] = model.addVar(vtype=GRB.BINARY, name=f"o_{i}_{k}")

# Función objetivo: Volumen total
volume = gp.quicksum(
   (PP[i][k][0] * PP[i][k][1] * PP[i][k][2]) * o[i, k]
    for i in I for k in range(6)
)
model.setObjective(volume, GRB.MAXIMIZE)

# Restricciones
for i in I:
    orient = P_a if i <6 else P_b
    # Selección-orientación
    model.addConstr(gp.quicksum(o[i, k] for k in range(len(orient))) == s[i])
    # Contención con Big-M
    w_i = gp.quicksum(orient[k][0] * o[i, k] for k in range(len(orient)))
    h_i = gp.quicksum(orient[k][1] * o[i, k] for k in range(len(orient)))
    d_i = gp.quicksum(orient[k][2] * o[i, k] for k in range(len(orient)))
    model.addConstr(x[i] + w_i <= W + M * (1 - s[i]))
    model.addConstr(y[i] + h_i <= H + M * (1 - s[i]))
    model.addConstr(z[i] + d_i <= D + M * (1 - s[i]))

# No solapamiento (solo entre fichas seleccionadas)
for i, j in itertools.combinations(I, 2):
    orient_i = P_a if i < 6 else P_b
    w_i = gp.quicksum(orient_i[k][0] * o[i, k] for k in range(len(orient_i)))
    h_i = gp.quicksum(orient_i[k][1] * o[i, k] for k in range(len(orient_i)))
    d_i = gp.quicksum(orient_i[k][2] * o[i, k] for k in range(len(orient_i)))
    
    orient_j = P_a if j < 6 else P_b
    w_j = gp.quicksum(orient_j[k][0] * o[j, k] for k in range(len(orient_j)))
    h_j = gp.quicksum(orient_j[k][1] * o[j, k] for k in range(len(orient_j)))
    d_j = gp.quicksum(orient_j[k][2] * o[j, k] for k in range(len(orient_j)))
    
    # Variables de dirección (b[i,j,m])
    b = model.addVars(6, vtype=GRB.BINARY, name=f"b_{i}_{j}")
    model.addConstr(gp.quicksum(b[m] for m in range(6)) >= 1)
    
    # Restricciones con Big-M ajustado para s_i y s_j
    model.addConstr(x[i] + w_i <= x[j] + M*(1 - b[0]) + M*(2 - s[i] - s[j]))
    model.addConstr(x[j] + w_j <= x[i] + M*(1 - b[1]) + M*(2 - s[i] - s[j]))
    model.addConstr(y[i] + h_i <= y[j] + M*(1 - b[2]) + M*(2 - s[i] - s[j]))
    model.addConstr(y[j] + h_j <= y[i] + M*(1 - b[3]) + M*(2 - s[i] - s[j]))
    model.addConstr(z[i] + d_i <= z[j] + M*(1 - b[4]) + M*(2 - s[i] - s[j]))
    model.addConstr(z[j] + d_j <= z[i] + M*(1 - b[5]) + M*(2 - s[i] - s[j]))

model.optimize()

# Resultados
if model.status == GRB.OPTIMAL:
    print(f"Volumen total: {model.objVal}")
    for i in I:
        if s[i].X > 0:
            orient = P_a if i < 6 else P_b
            #print('3')
            for p in range(len(orient)):
                #print('4')
                if o[i,p].X >0:
                    #print('5')
                    print(f"Ficha {i}:")
                    print(f"  Posición: ({x[i].X:.0f}, {y[i].X:.0f}, {z[i].X:.0f})")
                    print(f"  Orientación: {orient[p]}")
                    break
else:
    print("No solución óptima")