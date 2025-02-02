import pyomo.environ as pyomo
import os
import time

# Αρχικοποίηση abstract μοντέλου
model = pyomo.AbstractModel()

# Ανάγνωση των παραμέτρων N, F, S και P από το αρχείο
model.N = pyomo.Param(within=pyomo.NonNegativeIntegers) # Ν = αριθμός έργων 
model.F = pyomo.Param(within=pyomo.NonNegativeReals)    # F = διαθέσιμο κεφάλαιο 
model.S = pyomo.Param(within=pyomo.NonNegativeIntegers) # S = διαθέσιμο προσωπικό
model.P = pyomo.Param(within=pyomo.NonNegativeIntegers) # P = μέγιστος αριθμός έργων που μπορούν να υλοποιηθούν

# Ορισμός του αριθμού των έργων, αφού γίνει ανάγνωση της παραμέτρου Ν από το αρχείο
model.Projects = pyomo.RangeSet(1, model.N)

# Ανάγνωση των παραμέτρων performance, cost και staff από το αρχείο για όλα τα έργα 
model.performance = pyomo.Param(model.Projects)  # Απόδοση
model.cost = pyomo.Param(model.Projects)         # Κόστος
model.staff = pyomo.Param(model.Projects)        # Προσωπικό

# Ορισμός δυαδικών μεταβλητών για την επιλογή των έργων
model.x = pyomo.Var(model.Projects, domain=pyomo.Binary)

# Αντικειμενική Συνάρτηση (στόχος η μεγιστοποίηση της)
def obj_expression(m):
    return sum(m.performance[i] * m.x[i] for i in m.Projects)

model.obj = pyomo.Objective(sense=pyomo.maximize, rule=obj_expression)

# Περιορισμός για το διαθέσιμο κεφάλαιο
def budget_constraint(m):
    return sum(m.cost[i] * m.x[i] for i in m.Projects) <= m.F

model.budget_cstr = pyomo.Constraint(rule=budget_constraint)

# Περιορισμός για το διαθέσιμο προσωπικό
def staff_constraint(m):
    return sum(m.staff[i] * m.x[i] for i in m.Projects) <= m.S

model.staff_cstr = pyomo.Constraint(rule=staff_constraint)

# Περιορισμός για τον μέγιστο αριθμό έργων που μπορούν να υλοποιηθούν
def equipment_constraint(m):
    return sum(m.x[i] for i in m.Projects) <= m.P

model.equipment_cstr = pyomo.Constraint(rule=equipment_constraint)
 
# Συνάρτηση solve_all_problems για την επίλυση των αρχείων προβλημάτων
def solve_all_problems(problem_folder):
    
    # Ανάγνωση των path όλων των αρχείων προβλημάτων, εντός του φακέλου problems και αποθήκευση τους στην λίστα problems (το κάθε path στην κάθε γραμμή)
    problems = []
    for root, dirs, files in os.walk(problem_folder):
        for file in files:
            if file.endswith(".dat"):
                problems.append(os.path.join(root, file))
    
    # Επιλογή του gurobi solver
    solver = pyomo.SolverFactory('gurobi')

    # Βρόχος για επίλυση όλων των αρχείων προβλημάτων, όπου το path τους ανήκει στην λίστα problems
    for prob_file in problems:
        
        print(f"\nCurrently processing problem file: {prob_file}\n")

        # Φόρτωση δεδομένων από το αρχείο 
        instance = model.create_instance(prob_file)

        # Εκτύπωση του μοντέλου
        instance.pprint()

        # Επίλυση του μοντέλου
        start = time.time()
        solver.solve(instance)
        end = time.time()

        # Εκτύπωση των αποτελεσμάτων
        instance.display() 

        print(f"Time Elapsed: {end - start}")

    """
    print(f"\nCurrently processing problem file: problems\class_1\problem_1.dat\n")

    # Επιλογή του gurobi solver
    solver = pyomo.SolverFactory('gurobi')
    
    # Φόρτωση δεδομένων από το αρχείο 
    instance = model.create_instance("problems\class_1\problem_1.dat")

    # Εκτύπωση του μοντέλου
    instance.pprint()

    # Επίλυση του μοντέλου
    start = time.time()
    solver.solve(instance)
    end = time.time()

    # Εκτύπωση των αποτελεσμάτων
    instance.display() 

    print(f"Time Elapsed: {end - start}")
    """

# Κλήση της συνάρτησης solve_all_problems για επίλυση των αρχείων προβλημάτων του φακέλου 'problems'
solve_all_problems("problems")