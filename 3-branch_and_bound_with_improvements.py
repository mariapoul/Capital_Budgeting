import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from collections import deque
import problems as pr
import os

# Δήλωση global μεταβλητών
isMax = None          # True αν πρόκειται για πρόβλημα μεγιστοποίησης, False αν πρόκειται για πρόβλημα ελαχιστοποίησης
DEBUG_MODE = True     # True για ενεργοποίηση μηνυμάτων debuging, False για απενεργοποίηση μηνυμάτων debuging
nodes = 0             # αριθμός των κόμβων
lower_bound = -np.inf # κάτω όριο = - άπειρο
upper_bound = np.inf  # άνω όριο = άπειρο

# Συνάρτηση is_nearly_integer για την υπολογισμό αν μια τιμή προσεγγίζει μια ακέραια
def is_nearly_integer(value, tolerance=1e-6):

    return abs(value - round(value)) <= tolerance

# Συνάρτηση select_branching_variable για την επιλογή της μεταβλητής για branching
def select_branching_variable(x_candidate, integer_var):

    # Διατρέχουμε τον πίνακα integer_var και για τις μεταβλητές που πρέπει να έχουν ακέραια τιμή, υπολογίzουμε την απόσταση τους από τους πλησιέστερους
    # ακεραίους και κρατάμε την μικρότερη τιμή. Οι τιμές αυτές αποθηκεύονται έπειτα στην λίστα fractional_vars
    fractional_vars = [(i, min(abs(x - np.floor(x)), abs(np.ceil(x) - x))) 
                       for i, x in enumerate(x_candidate) if integer_var[i] and not is_nearly_integer(x)]
    
    # Διατρέχουμε την λίστα fractional_vars και επιστρέφουμε την μεταβλητή με την μεγαλύτερη απόσταση, αν δεν υπάρχει μεταβλητή επιστρέφεται None
    return max(fractional_vars, key=lambda x: x[1])[0] if fractional_vars else None

# Συνάρτηση apply_gomory_cut για την εφαρμογή Gomory Cuts
def apply_gomory_cut(model, x_candidate, integer_var):

    # Διατρέχουμε τον πίνακα των μεταβλητών 
    for i, x in enumerate(x_candidate):
        # Στην περίπτωση που μια μεταβλητή πρέπει να έχει ακέραια τιμή, αλλά δεν είναι ακέραια, εισάγουμε έναν περιορισμό Gomory Cut αναγκάζοντας τη 
        # μεταβλητή να πάρει τιμή μικρότερη ή ίση με το ακέραιο μέρος της
        if integer_var[i] and not is_nearly_integer(x):
            model.addConstr(model.getVars()[i] <= np.floor(x), name=f"GomoryCut_{i}")

def branch_and_bound(model, ub, lb, integer_var):
    
    # Αρχικοποίηση των global μεταβλητών nodes, lower_bound και upper_bound, καθώς τροποποιούνται στην κλήση της συνάρτησης για το κάθε αρχείο 
    # προβλήματος
    global nodes, lower_bound, upper_bound
    nodes = 0 
    lower_bound = -np.inf 
    upper_bound = np.inf 

    # Δημιουργία άδειας στοίβας με την χρήση της deque()
    stack = deque()

    # Δημιουργία μιας κενής λίστας για την αποθήκευση των λύσεων
    solutions = list()
    # Αρχικοποίηση ενός μετρητή για τον αριθμό των λύσεων που βρέθηκαν
    solutions_found = 0
    # Αρχικοποίηση ενός δείκτη για την αποθήκευση του βάθος της καλύτερης λύσης
    best_sol_idx = 0

    # Δημιουργία ενός tuple, το root_node, το οποίο αντιπροσωπεύει την ρίζα του δέντρου και περιλαμβάνει τους πίνακες άνω και κάτω ορίων των μεταβλητών
    # και το βάθος του κόμβου
    root_node = (ub, lb, 0)

    # Εισαγωγή των κόμβου ρίζα στην στοίβα
    stack.append(root_node)

    # Όσο η στοίβα δεν είναι άδεια 
    while stack:
        
        # Αφαίρεση του τελευταίου κόμβου από την στοίβα και αποθήκευση των στοιχείων του στις αντίστοιχες μεταβλητές
        ub, lb, depth = stack.pop()

        # Αύξηση του αριθμού των κόμβων που έχουν εξερευνηθεί κατά 1
        nodes += 1

        # Ανανέωση των κάτω και άνω ορίων των μεταβλητών του κόμβου
        model.setAttr("LB", model.getVars(), lb)
        model.setAttr("UB", model.getVars(), ub)

        # Ενημέρωση του μοντέλου
        model.optimize()

        # Στην περίπτωση που βρεθεί μη εφικτή λύση, εξερεύνησε τον επόμενο κόμβο
        if model.status != GRB.OPTIMAL:
            continue

        # Αποθήκευση των νέων τιμών των μεταβλητών στον πίνακα x_candidate μετά την επίλυση 
        x_candidate = model.getAttr('X', model.getVars())

        # Αποθήκευση της τιμής της αντικειμενικής συνάρτησης στην μεταβλητή x_obj
        x_obj = model.ObjVal
        
        # Στην περίπτωση που βρούμε ακέραια λύση (καλώντας την συνάρτηση is_nearly_integer για κάθε μεταβλητή) προστίθεται η λύση στην λίστα solutions
        # και αυξάνεται ο μετρητής solution_found κατά 1,
        if all(is_nearly_integer(x_candidate[i]) for i, is_int in enumerate(integer_var) if is_int):
            solutions.append((x_candidate, x_obj, depth))
            solutions_found += 1
            # Στην περίπτωση που το πρόβλημα είναι πρόβλημα μεγιστοποίησης και η τιμή της αντικειμενικής συνάρτησης για την ακέραια λύση είναι 
            # μεγαλύτερη από την υπάρχουσα τιμή του κάτω ορίου, τότε η τιμή του κάτω ορίου ανανεώνεται
            if isMax and x_obj > lower_bound:
                lower_bound = x_obj
            # Στην περίπτωση που το πρόβλημα είναι πρόβλημα ελαχιστοποίησης και η τιμή της αντικειμενικής συνάρτησης για την ακέραια λύση είναι 
            # μικρότερη από την υπάρχουσα τιμή του άνω ορίου, τότε η τιμή του άνω ορίου ανανεώνεται
            elif not isMax and x_obj < upper_bound:
                upper_bound = x_obj
            # Εξερεύνησε τον επόμενο κόμβο
            continue  

        # Στην περίπτωση που βρούμε μη ακέραια λύση, καλούμε την συνάρτηση select_branching_variable για την επιλογή της μεταβλητής για branching
        selected_var = select_branching_variable(x_candidate, integer_var)

        # Στην περίπτωση που δεν υπάρχει μεταβλητή για branching, γίνεται εξερεύνηση του επόμενου κόμβου
        if selected_var is None:
            continue 

        # Ανάθεση των πινάκων lb και ub για την δημιουργία άνω ορίων για τον αριστερό κόμβο παιδιού και την δημιουργία κάτω ορίων για τον δεξί κόμβο 
        # παιδιού
        left_ub = np.copy(ub)
        right_lb = np.copy(lb)

        # Έχουμε αποθηκεύσει την θέση της μη ακέραιας μεταβλητής στην μεταβλητή selected_var
        # Δημιουργούμε ένα αριστερό branch στρογγυλοποιώντας προς τα κάτω την τιμή της μη ακέραιας μεταβλητής και ένα δεξί branch στρογγυλοποιώντας 
        # προς τα πάνω την τιμή της μη ακέραιας μεταβλητής και ανανεώνουμε τα όρια των branches για την συγκεκριμένη μεταβλητή
        left_ub[selected_var] = np.floor(x_candidate[selected_var])
        right_lb[selected_var] = np.ceil(x_candidate[selected_var])
        
        # Εισαγωγή των κόμβων παιδιών στην στοίβα, εισάγοντας τους πίνακες άνω και κάτω ορίων των μεταβλητών και το βάθος των κόμβων
        stack.append((ub, right_lb, depth + 1))
        stack.append((left_ub, lb, depth + 1))

        # Κλήση της συνάρτησης apply_gomory_cut για την εφαρμογή Gomory Cuts
        apply_gomory_cut(model, x_candidate, integer_var)
    
    return solutions, best_sol_idx, solutions_found
    
if __name__ == "__main__":

    # Ανάγνωση των path όλων των αρχείων προβλημάτων, εντός του φακέλου problems και αποθήκευση τους στην λίστα problems (το κάθε path στην κάθε γραμμή)
    problems = []
    for root, dirs, files in os.walk("problems"):
        for file in files:
            if file.endswith(".dat"):
                problems.append(os.path.join(root, file))

    # Βρόχος για επίλυση όλων των αρχείων προβλημάτων, όπου το path τους ανήκει στην λίστα problems
    for prob_file in problems:

        print(f"Currently processing problem file: {prob_file}")
    
        # Κλήση της συνάρτησης capital_budgeting του αρχείου pr για την δημιουργία του μοντέλου
        model, ub, lb, integer_var, num_vars, c = pr.capital_budgeting(prob_file)
    
        # Κλήση της συνάρτησης branch_and_bound που υλοποιεί τον αλγόριθμο branch and bound και χρονομέτρηση του χρόνου εκτέλεσης
        print("************************    Running branch and bound with improvements    ************************\n\n")
        start = time.time()
        solutions, best_sol_idx, solutions_found = branch_and_bound(model, ub, lb, integer_var)
        end = time.time()

        # Εκτύπωση των αποτελεσμάτων
        print(f"Optimal Solution: {solutions[best_sol_idx]}")
        print(f"Time Elapsed: {end - start}")
        print(f"Total nodes: {nodes}")

    """
    print(f"Currently processing problem file: problems\class_1\problem_1.dat")
    
    # Κλήση της συνάρτησης capital_budgeting του αρχείου pr για την δημιουργία του μοντέλου
    model, ub, lb, integer_var, num_vars, c = pr.capital_budgeting("problems\class_1\problem_1.dat")
    
    # Κλήση της συνάρτησης branch_and_bound που υλοποιεί τον αλγόριθμο branch and bound και χρονομέτρηση του χρόνου εκτέλεσης
    print("************************    Running branch and bound with heuristic    ************************\n\n")
    start = time.time()
    solutions, best_sol_idx, solutions_found = branch_and_bound(model, ub, lb, integer_var)
    end = time.time()

    # Εκτύπωση των αποτελεσμάτων
    print(f"Optimal Solution: {solutions[best_sol_idx]}")
    print(f"Time Elapsed: {end - start}")
    print(f"Total nodes: {nodes}")
    """