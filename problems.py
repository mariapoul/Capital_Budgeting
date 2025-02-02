import numpy as np 
import gurobipy as gp
from gurobipy import GRB

# Συνάρτηση read_data_capital_budgeting για την ανάγνωση δεδομένων από το αρχείο
def read_data_capital_budgeting(filename):

    # Άνοιγμα του αρχείου και αποθήκευση του στην λίστα lines (όπως είναι, γραμμή προς γραμμή)
    with open(filename, "rt") as f:
        lines = f.readlines()
    
    # Αφαίρεση κενών γραμμών και περιττών κενών από την λίστα lines
    lines = [line.strip() for line in lines if line.strip()]
    
    # Ανάγνωση των παραμέτρων N, F, S και P από την λίστα lines
    N = int(lines[0].split(":=")[1].strip().split(";")[0])
    F = int(lines[1].split(":=")[1].strip().split(";")[0])
    S = int(lines[2].split(":=")[1].strip().split(";")[0])
    P = int(lines[3].split(":=")[1].strip().split(";")[0])
    
    # Εντοπισμός της γραμμής όπου ορίζονται οι παράμετροι performance, cost και staff για το κάθε έργο 
    # (η γραμμή μετά την σειρά 'param: performance cost staff :=')
    data_start_index = lines.index('param: performance cost staff :=') + 1

    # Ανάγνωση της κάθε γραμμής δεδομένων (των δεδομένων του κάθε έργου) και αποθήκευση τους στην λίστα project_data (γραμμή προς γραμμή), 
    # όσο δεν συναντάται το ;
    project_data = []
    for line in lines[data_start_index:]:
        if line.strip() == ";":
            break  # End of data
        project_data.append(list(map(int, line.split())))

    # Μετατροπή της λίστας project_data σε πίνακα
    project_data = np.array(project_data)
    
    # Εξαγωγή των στηλών στις οποίες ανήκουν τα δεδομένα performance, cost και staff των έργων και αποθήκευση τους σε πίνακες
    performance = project_data[:, 1]
    cost = project_data[:, 2]
    staff = project_data[:, 3]
    
    return N, F, S, P, performance, cost, staff

# Συνάρτηση capital_budgeting για την δημιουργία του μοντέλου
def capital_budgeting(filename):

    # Κλήση της συνάρτησης read_data_capital_budgeting για την ανάγνωση δεδομένων από το αρχείο
    N, F, S, P, performance, cost, staff = read_data_capital_budgeting(filename)

    # Ορισμός των μεταβλητών (N έργα + 1 για το z)
    num_vars = N + 1  

    # Αρχικοποίηση του μοντέλου
    model = gp.Model()

    # Ορισμός άνω και κάτω ορίων για τις μεταβλητές 
    # (τα κάτω όρια είναι 0 για τα έργα και το z, τα άνω όρια είναι 1 για τα έργα και άπειρο για το z)
    ub = [1 if i < num_vars-1 else np.inf for i in range(num_vars) ] 
    lb = [0 for i in range(num_vars) ]  

    # Εισαγωγή μεταβλητών στο μοντέλο (έργα + z) με τα άνω και τα κάτω όρια τους 
    x = model.addVars(num_vars, lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name="x")

    # Περιορισμός για το διαθέσιμο κεφάλαιο
    model.addConstr(gp.quicksum(cost[i] * x[i] for i in range(N)) <= F)

    # Περιορισμός για το διαθέσιμο προσωπικό
    model.addConstr(gp.quicksum(staff[i] * x[i] for i in range(N)) <= S)

    # Περιορισμός για τον μέγιστο αριθμό έργων που μπορούν να υλοποιηθούν
    model.addConstr(gp.quicksum(x[i] for i in range(N)) <= P)
    
    # Ορισμός πίνακα c με τους συντελεστές της αντικειμενικής συνάρτησης
    c = [performance[i] for i in range(num_vars - 1)] + [0]

    # Αντικειμενική συνάρτηση
    model.setObjective(gp.quicksum(performance[i] * x[i] for i in range(N)))

    # Μεγιστοποίηση αντικειμενικής συνάρτησης
    model.ModelSense = GRB.MAXIMIZE 

    # Χρήση του Dual Simplex αλγορίθμου του Gurobi
    model.Params.method = 1  

    # Ενημέρωση του μοντέλου
    model.update()

    # Ορισμός των μεταβλητών που πρέπει να έχουν ακέραιες τιμές
    # (τα έργα πρέπει υποχρεωτικά να παίρνουν ακέραιες τιμές, επομένως True, ενώ το z δεν είναι υποχρεωτικό να παίρνει ακέραια τιμή, επομένως False)
    integer_var = [True if i < num_vars-1 else False for i in range(num_vars)] 

    # Εμφάνιση του μοντέλου
    model.display()  
    
    return model, ub, lb, integer_var, num_vars, c