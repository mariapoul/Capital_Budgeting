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

# Κλάση Node που αποθηκεύει πληροφορίες για τους κόμβους του δέντρου
class Node:
    def __init__(self, ub, lb, depth, vbasis, cbasis, branching_var, label=""):
        self.ub = ub
        self.lb = lb
        self.depth = depth
        self.vbasis = vbasis
        self.cbasis = cbasis
        self.branching_var = branching_var
        self.label = label

# Συνάρτηση debug_print για την εκτύπωση πληροφοριών για debuging
def debug_print(node: Node = None, x_obj=None, sol_status=None):

    print("\n\n-----------------  DEBUG OUTPUT  -----------------\n\n")
    print(f"UB:{upper_bound}")
    print(f"LB:{lower_bound}")

    if node is not None:
        print(f"Branching Var: {node.branching_var}")
    if node is not None:
        print(f"Child: {node.label}")
    if node is not None:
        print(f"Depth: {node.depth}")
    if x_obj is not None:
        print(f"Simplex Objective: {x_obj}")
    if sol_status is not None:
        print(f"Solution status: {sol_status}")

    print("\n\n--------------------------------------------------\n\n")

# Συνάρτηση branch_and_bound που υλοποιεί τον αλγόριθμο branch and bound
def branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth, nodes_per_depth, vbasis=[], cbasis=[], depth=0):

    # Αρχικοποίηση των global μεταβλητών nodes, lower_bound και upper_bound, καθώς τροποποιούνται στην κλήση της συνάρτησης για το κάθε αρχείο 
    # προβλήματος
    global nodes, lower_bound, upper_bound
    nodes = 0 
    lower_bound = -np.inf 
    upper_bound = np.inf 

    # Δημιουργία άδειας στοίβας με την χρήση της deque() για την αποθήκευση των αντικειμένων τύπου Node
    stack = deque()

    # Δημιουργία μιας κενής λίστας για την αποθήκευση των λύσεων
    solutions = list()
    # Αρχικοποίηση ενός μετρητή που αντιπροσωπεύει τον αριθμό των λύσεων που βρέθηκαν
    solutions_found = 0
    # Αρχικοποίηση ενός δείκτη για την αποθήκευση της θέσης της καλύτερης λύσης από τις υπάρχουσες λύσεις
    best_sol_idx = 0

    # Αρχικοποίηση της μεταβλητής best_sol_obj στο -άπειρο αν το πρόβλημα μας πρόκειται για πρόβλημα μεγιστοποίησης, 
    # αλλιώς αρχικοποίηση της μεταβλητής best_sol_obj στο άπειρο αν το πρόβλημα μας πρόκειται για πρόβλημα ελαχιστοποίησης
    # Η μεταβλητή best_sol_obj αντιπροσωπεύει την αρχική τιμή της καλύτερης λύσης
    if isMax:
        best_sol_obj = -np.inf
    else:
        best_sol_obj = np.inf

    # Δημιουργία ενός αντικειμένου τύπου Node, το root_node, το οποίο αντιπροσωπεύει την ρίζα του δέντρου
    root_node = Node(ub, lb, depth, vbasis, cbasis, -1, "root")

    # Ανανέωση του πίνακα nodes_per_depth στο επίπεδο 0 (ρίζα), μείωση κατά 1, εφόσον ο κόμβος (ρίζα) εξερευνήθηκε
    nodes_per_depth[0] -= 1

    # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging
    if DEBUG_MODE:
        debug_print()

    # Επίλυση του μοντέλου
    model.optimize()

    # Στην περίπτωση που βρεθεί μη εφικτή λύση 
    # Καλείται η συνάρτηση debug_print με sol_status = Infeasible και επιστρέφεται μια κενή λίστα, η αρχική τιμή της καλύτερης λύσης (ανάλογα με το αν
    # πρόκειται για πρόβλημα μεγιστοποίησης ή ελαχιστοποίησης, αντίστοιχα) και το βάθος της αναζήτησης
    if model.status != GRB.OPTIMAL:
        if isMax:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], -np.inf, depth
        else:
            if DEBUG_MODE:
                debug_print(node=root_node, sol_status="Infeasible")
            return [], np.inf, depth

    # Αποθήκευση των νέων τιμών των μεταβλητών στον πίνακα x_candidate μετά την επίλυση 
    x_candidate = model.getAttr('X', model.getVars())

    # Αποθήκευση της τιμής της αντικειμενικής συνάρτησης στην μεταβλητή x_obj
    x_obj = model.ObjVal

    # Αποθήκευση της τιμής της αντικειμενικής συνάρτησης ως το καλύτερο όριο για το επίπεδο 0
    best_bound_per_depth[0] = x_obj

    # Έχουμε αρχικοποιήσει τον πίνακα integer_vals με τις μεταβλητές που πρέπει υποχρεωτικά να παίρνουν ακέραιες τιμές (έργα)
    # Διατρέχουμε τον πίνακα integer_vals και καλούμε την συνάρτηση is_nearly_integer για την κάθε μεταβλητή για να ελέγξουμε αν είναι ακέραια 
    # Στην περίπτωση που βρεθεί μεταβλητή που δεν είναι ακέραια αποθηκεύουμε την θέση της στην μεταβλητή selected_var_idx
    vars_have_integer_vals = True
    for idx, is_int_var in enumerate(integer_var):
        if is_int_var and not is_nearly_integer(x_candidate[idx]):
            vars_have_integer_vals = False
            selected_var_idx = idx
            break

    # Στην περίπτωση που όλες οι μεταβλητές έχουν ακέραιες τιμές προστίθεται η λύση στην λίστα solutions, αυξάνεται ο μετρητής solution_found κατά 1, 
    # καλείται η συνάρτηση debug_print για την εκτύπωση πληροφοριών για debuging με sol_status = Integer και επιστρέφεται η λίστα solutions, 
    # ο δείκτης best_sol_idx και ο μετρητής solution_found
    if vars_have_integer_vals:

        solutions.append([x_candidate, x_obj, depth])
        solutions_found += 1

        if DEBUG_MODE:
            debug_print(node=root_node, x_obj=x_obj, sol_status="Integer")
        return solutions, best_sol_idx, solutions_found

    # Αλλιώς, ανανεώνεται το άνω ή το κάτω όριο ανάλογα με το αν το πρόβλημα μας πρόκειται για μεγιστοποίησης ή ελαχιστοποίησης αντίστοιχα
    else:
        if isMax:
            upper_bound = x_obj
        else:
            lower_bound = x_obj

    # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging με με sol_status = Fractional
    if DEBUG_MODE:
        debug_print(node=root_node, x_obj=x_obj, sol_status="Fractional")

    # Αποθήκευση των θέσεων των μεταβλητών που έχουν μη μηδενική τιμή και των δεικτών των περιορισμών στους πίνακες vbasis και cbasis αντίστοιχα
    vbasis = model.getAttr("VBasis", model.getVars())
    cbasis = model.getAttr("CBasis", model.getConstrs())

    # Ανάθεση των πινάκων lb και ub για την δημιουργία κάτω (0) και άνω (1) ορίων (πινάκων ορίων) για τις μεταβλητές των κόμβων παιδιών
    left_lb = np.copy(lb)
    left_ub = np.copy(ub)
    right_lb = np.copy(lb)
    right_ub = np.copy(ub)

    # Έχουμε αποθηκεύσει την θέση της μη ακέραιας μεταβλητής στην μεταβλητή selected_var_idx
    # Δημιουργούμε ένα αριστερό branch στρογγυλοποιώντας προς τα κάτω την τιμή της μη ακέραιας μεταβλητής και ένα δεξί branch στρογγυλοποιώντας 
    # προς τα πάνω την τιμή της μη ακέραιας μεταβλητής και ανανεώνουμε τα όρια των branches για την συγκεκριμένη μεταβλητή
    left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
    right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

    # Δημιουργία δυο αντικειμένων τύπου Node, που αντιπροσωπεύουν τους κόμβους παιδιά
    left_child = Node(left_ub, left_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Left")
    right_child = Node(right_ub, right_lb, root_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx, "Right")

    # Εισαγωγή των κόμβων παιδιών στην στοίβα
    stack.append(right_child)
    stack.append(left_child)

    # Όσο υπάρχουν κόμβοι στην στοίβα
    while (len(stack) != 0):

        print("\n********************************  NEW NODE BEING EXPLORED  ******************************** ")

        # Αύξηση του αριθμού των κόμβων που έχουν εξερευνηθεί κατά 1
        nodes += 1

        # Ανάθεση στην μεταβλητή current_node τον τελευταίο κόμβο που μπήκε στην στοίβα, αντικείμενο τύπου Node
        current_node = stack[-1]

        # Αφαίρεση του τελευταίου κόμβου από την στοίβα
        stack.pop()

        # Ανανέωση του πίνακα nodes_per_depth στο επίπεδο του current_node, μείωση κατά 1, εφόσον ο κόμβος εξερευνήθηκε
        nodes_per_depth[current_node.depth] -= 1

        # Αν οι πίνακες vbasis και cbasis δεν είναι κενοί, γίνεται ανανέωση  τους για τις μεταβλητές του κόμβου
        if (len(current_node.vbasis) != 0) and (len(current_node.cbasis) != 0):
            model.setAttr("VBasis", model.getVars(), current_node.vbasis)
            model.setAttr("CBasis", model.getConstrs(), current_node.cbasis)

        # Ανανέωση των κάτω και άνω ορίων των μεταβλητών του κόμβου
        model.setAttr("LB", model.getVars(), current_node.lb)
        model.setAttr("UB", model.getVars(), current_node.ub)

        # Ενημέρωση του μοντέλου
        model.update()

         # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging
        if DEBUG_MODE:
            debug_print()

        # Επίλυση του μοντέλου
        model.optimize()

        # Στην περίπτωση που βρεθεί μη εφικτή λύση 
        # Ανανεώνεται η μεταβλητή infeasible με την τιμή True και η μεταβλητή x_obj με την αρχική τιμή της καλύτερης λύσης (ανάλογα με το αν πρόκειται
        # για πρόβλημα μεγιστοποίησης ή ελαχιστοποίησης). Στην συνέχεια, μειώνουμε τον αριθμό των κόμβων nodes_per_depth στα επόμενα επίπεδα του 
        # δέντρου Branch and Bound
        infeasible = False
        if model.status != GRB.OPTIMAL:
            if isMax:
                infeasible = True
                x_obj = -np.inf
            else:
                infeasible = True
                x_obj = np.inf
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

        else:
            # Αποθήκευση των νέων τιμών των μεταβλητών στον πίνακα x_candidate μετά την επίλυση 
            x_candidate = model.getAttr('X', model.getVars())

            # Αποθήκευση της τιμής της αντικειμενικής συνάρτησης στην μεταβλητή x_obj
            x_obj = model.ObjVal

            # Αν το πρόβλημα είναι μεγιστοποίησης και η τιμή της αντικειμενικής συνάρτησης (x_obj) είναι μεγαλύτερη από το προηγούμενο καλύτερο bound 
            # σε αυτό το βάθος (best_bound_per_depth), ενημερώνουμε το best_bound_per_depth με αυτήν την τιμή
            # Αλλιώς, αν το πρόβλημα είναι ελαχιστοποίησης και η τιμή της αντικειμενικής συνάρτησης (x_obj) είναι μικρότερη από το προηγούμενο καλύτερο
            # bound σε αυτό το βάθος (best_bound_per_depth), ενημερώνουμε το best_bound_per_depth με αυτήν την τιμή
            if isMax == True and x_obj > best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj
            elif isMax == False and x_obj < best_bound_per_depth[current_node.depth]:
                best_bound_per_depth[current_node.depth] = x_obj

            # Αν δεν υπάρχει άλλος κόμβος στο επίπεδο και το πρόβλημα μας πρόκειται για πρόβλημα μεγιστοποίησης, ανανεώνουμε το άνω όριο με την τιμή 
            # του καλύτερου bound σε αυτό το βάθος (best_bound_per_depth), αλλιώς αν το πρόβλημα μας πρόκειται για πρόβλημα ελαχιστοποίησης,
            # ανανεώνουμε το κάτω όριο με την τιμή του καλύτερου bound σε αυτό το βάθος (best_bound_per_depth)
            if nodes_per_depth[current_node.depth] == 0:
                if isMax == True:
                    upper_bound = best_bound_per_depth[current_node.depth]
                else:
                    lower_bound = best_bound_per_depth[current_node.depth]

        # Στην περίπτωση που βρεθεί μη εφικτή λύση
        # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging με με sol_status = Infeasible
        if infeasible:
            if DEBUG_MODE:
                debug_print(node=current_node, sol_status="Infeasible")
            continue

        # Έχουμε αρχικοποιήσει τον πίνακα integer_vals με τις μεταβλητές που πρέπει υποχρεωτικά να παίρνουν ακέραιες τιμές (έργα)
        # Διατρέχουμε τον πίνακα integer_vals και καλούμε την συνάρτηση is_nearly_integer για την κάθε μεταβλητή για να ελέγξουμε αν είναι ακέραια 
        # Στην περίπτωση που βρεθεί μεταβλητή που δεν είναι ακέραια αποθηκεύουμε την θέση της στην μεταβλητή selected_var_idx
        vars_have_integer_vals = True
        for idx, is_int_var in enumerate(integer_var):
            if is_int_var and not is_nearly_integer(x_candidate[idx]):
                vars_have_integer_vals = False
                selected_var_idx = idx
                break

        # Στην περίπτωση που όλες οι μεταβλητές έχουν ακέραιες τιμές
        if vars_have_integer_vals: 
            # Στην περίπτωση που το πρόβλημα είναι πρόβλημα μεγιστοποίησης
            if isMax:
                # Στην περίπτωση που η τιμή της αντικειμενικής συνάρτησης για την ακέραια λύση είναι μεγαλύτερη από την υπάρχουσα τιμή του κάτω ορίου, 
                # τότε η τιμή του κάτω ορίου ανανεώνεται
                if lower_bound < x_obj: 
                    lower_bound = x_obj
                    # Στην περίπτωση που η τιμή του κάτω ορίου ταυτίζεται με αυτήν του άνω ορίου, τότε έχουμε βρει την βέλτιστη λύση
                    if abs(lower_bound - upper_bound) < 1e-6:
                        # Προστίθεται η λύση στην λίστα solutions και αυξάνεται ο μετρητής solution_found κατά 1 
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                        # Στην περίπτωση που η λύση είναι η καλύτερη λύση που έχει βρεθεί μέχρι στιγμής, ανανεώνεται η μεταβλητή best_sol_obj με την 
                        # τιμή αυτή, η μεταβλητή best_sol_idx με την θέση της καλύτερης λύσης και καλείται η συνάρτηση debug_print για την εκτύπωση 
                        # πληροφοριών για debuging με sol_status = Integer/Optimal
                        if (abs(x_obj - best_sol_obj) < 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1

                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # Στην περίπτωση που δεν έχουμε βρει την βέλτιστη λύση 
                    # Προστίθεται η λύση στην λίστα solutions και αυξάνεται ο μετρητής solution_found κατά 1
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    # Στην περίπτωση που η λύση είναι η καλύτερη λύση που έχει βρεθεί μέχρι στιγμής, ανανεώνεται η μεταβλητή best_sol_obj με την τιμή 
                    # αυτή, η μεταβλητή best_sol_idx με την θέση της καλύτερης λύσης και καλείται η συνάρτηση debug_print για την εκτύπωση πληροφοριών
                    # για debuging με sol_status = Integer/Optimal
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # Μείωση του αριθμού των κόμβων nodes_per_depth στα επόμενα επίπεδα του δέντρου Branch and Bound
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging με με sol_status = Integer
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue
            # Στην περίπτωση που το πρόβλημα είναι πρόβλημα ελαχιστοποίησης
            else:
                # Στην περίπτωση που η τιμή της αντικειμενικής συνάρτησης για την ακέραια λύση είναι μικρότερη από την υπάρχουσα τιμή του άνω ορίου, 
                # τότε η τιμή του άνω ορίου ανανεώνεται
                if upper_bound > x_obj: 
                    upper_bound = x_obj 
                    # Στην περίπτωση που η τιμή του κάτω ορίου ταυτίζεται με αυτήν του άνω ορίου, τότε έχουμε βρει την βέλτιστη λύση
                    if abs(lower_bound - upper_bound) < 1e-6: 
                        # Στην περίπτωση που δεν έχουμε βρει την βέλτιστη λύση 
                        # Προστίθεται η λύση στην λίστα solutions και αυξάνεται ο μετρητής solution_found κατά 1
                        solutions.append([x_candidate, x_obj, current_node.depth])
                        solutions_found += 1
                         # Στην περίπτωση που η λύση είναι η καλύτερη λύση που έχει βρεθεί μέχρι στιγμής, ανανεώνεται η μεταβλητή best_sol_obj με την 
                         # τιμή αυτή, η μεταβλητή best_sol_idx με την θέση της καλύτερης λύσης και καλείται η συνάρτηση debug_print για την εκτύπωση 
                         # πληροφοριών για debuging με sol_status = Integer/Optimal
                        if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                            best_sol_obj = x_obj
                            best_sol_idx = solutions_found - 1
                            if DEBUG_MODE:
                                debug_print(node=current_node, x_obj=x_obj, sol_status="Integer/Optimal")
                        return solutions, best_sol_idx, solutions_found

                    # Στην περίπτωση που δεν έχουμε βρει την βέλτιστη λύση 
                    # Προστίθεται η λύση στην λίστα solutions και αυξάνεται ο μετρητής solution_found κατά 1
                    solutions.append([x_candidate, x_obj, current_node.depth])
                    solutions_found += 1
                    # Στην περίπτωση που η λύση είναι η καλύτερη λύση που έχει βρεθεί μέχρι στιγμής, ανανεώνεται η μεταβλητή best_sol_obj με την τιμή 
                    # αυτή, η μεταβλητή best_sol_idx με την θέση της καλύτερης λύσης 
                    if (abs(x_obj - best_sol_obj) <= 1e-6) or solutions_found == 1:
                        best_sol_obj = x_obj
                        best_sol_idx = solutions_found - 1

                    # Μείωση του αριθμού των κόμβων nodes_per_depth στα επόμενα επίπεδα του δέντρου Branch and Bound
                    for i in range(current_node.depth + 1, len(nodes_per_depth)):
                        nodes_per_depth[i] -= 2 * (i - current_node.depth)

                    # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging με με sol_status = Integer
                    if DEBUG_MODE:
                        debug_print(node=current_node, x_obj=x_obj, sol_status="Integer")
                    continue

            # Στην περίπτωση που η λύση, x_obj, δεν βελτιώνει την υπάρχουσα καλύτερη, best_sol_obj (x_obj==best_sol_obj)
            # Μείωση του αριθμού των κόμβων nodes_per_depth στα επόμενα επίπεδα του δέντρου Branch and Bound
            for i in range(current_node.depth + 1, len(nodes_per_depth)):
                nodes_per_depth[i] -= 2 * (i - current_node.depth)

            # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging με με sol_status = Integer (Rejected -- Doesn't improve
            # incumbent)
            if DEBUG_MODE:
                debug_print(node=current_node, x_obj=x_obj,
                            sol_status="Integer (Rejected -- Doesn't improve incumbent)")
            # Εξερεύνηση του επόμενου κόμβου
            continue

        # Στην περίπτωση που το πρόβλημα είναι πρόβλημα μεγιστοποίησης
        if isMax:
            # Στην περίπτωση που η τιμή της αντικειμενικής συνάρτησης είναι μικρότερη από την υπάρχουσα τιμή του κάτω ορίου ή σχεδόν ταυτίζονται
            if x_obj < lower_bound or abs(x_obj - lower_bound) < 1e-6: 
                # Μείωση του αριθμού των κόμβων nodes_per_depth στα επόμενα επίπεδα του δέντρου Branch and Bound
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging με με sol_status = Fractional -- Cut by bound
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                # Εξερεύνηση του επόμενου κόμβου
                continue
        # Στην περίπτωση που το πρόβλημα είναι πρόβλημα ελαχιστοποίησης
        else:
            # Στην περίπτωση που η τιμή της αντικειμενικής συνάρτησης είναι μεγαλύτερη από την υπάρχουσα τιμή του άνω ορίου ή σχεδόν ταυτίζονται
            if x_obj > upper_bound or abs(x_obj - upper_bound) < 1e-6: 
                # Μείωση του αριθμού των κόμβων nodes_per_depth στα επόμενα επίπεδα του δέντρου Branch and Bound
                for i in range(current_node.depth + 1, len(nodes_per_depth)):
                    nodes_per_depth[i] -= 2 * (i - current_node.depth)
                # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging με με sol_status = Fractional -- Cut by bound
                if DEBUG_MODE:
                    debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional -- Cut by bound")
                # Εξερεύνηση του επόμενου κόμβου
                continue

        # Κλήση της συνάρτησης debug_print για την εκτύπωση πληροφοριών για debuging με με sol_status = Fractional
        if DEBUG_MODE:
            debug_print(node=current_node, x_obj=x_obj, sol_status="Fractional")

        # Αποθήκευση των θέσεων των μεταβλητών που έχουν μη μηδενική τιμή και των δεικτών των περιορισμών στους πίνακες vbasis και cbasis αντίστοιχα
        vbasis = model.getAttr("VBasis", model.getVars())
        cbasis = model.getAttr("CBasis", model.getConstrs())

        # Ανάθεση των πινάκων lb και ub για την δημιουργία κάτω (0) και άνω (1) ορίων (πινάκων ορίων) για τις μεταβλητές των κόμβων παιδιών
        left_lb = np.copy(current_node.lb)
        left_ub = np.copy(current_node.ub)
        right_lb = np.copy(current_node.lb)
        right_ub = np.copy(current_node.ub)

        # Έχουμε αποθηκεύσει την θέση της μη ακέραιας μεταβλητής στην μεταβλητή selected_var_idx
        # Δημιουργούμε ένα αριστερό branch στρογγυλοποιώντας προς τα κάτω την τιμή της μη ακέραιας μεταβλητής και ένα δεξί branch στρογγυλοποιώντας 
        # προς τα πάνω την τιμή της μη ακέραιας μεταβλητής και ανανεώνουμε τα όρια των branches για την συγκεκριμένη μεταβλητή
        left_ub[selected_var_idx] = np.floor(x_candidate[selected_var_idx])
        right_lb[selected_var_idx] = np.ceil(x_candidate[selected_var_idx])

        # Δημιουργία δυο αντικειμένων τύπου Node, που αντιπροσωπεύουν τους κόμβους παιδιά
        left_child = Node(left_ub, left_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                          "Left")
        right_child = Node(right_ub, right_lb, current_node.depth + 1, vbasis.copy(), cbasis.copy(), selected_var_idx,
                           "Right")

        # Εισαγωγή των κόμβων παιδιών στην στοίβα
        stack.append(right_child)
        stack.append(left_child)

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
        
        print(f"Currently processing problem file: {prob_file}\n")
        print("************************    Initializing structures...    ************************")

        # Κλήση της συνάρτησης capital_budgeting του αρχείου pr για την δημιουργία του μοντέλου
        model, ub, lb, integer_var, num_vars, c = pr.capital_budgeting(prob_file)

        # Πρόβλημα μεγιστοποίησης
        isMax = True 

        # Αρχικοποίηση του πίνακα best_bound_per_depth στο -άπειρο αν το πρόβλημα μας πρόκειται για πρόβλημα μεγιστοποίησης, 
        # αλλιώς αρχικοποίηση του πίνακα best_bound_per_depth στο άπειρο αν το πρόβλημα μας πρόκειται για πρόβλημα ελαχιστοποίησης
        # Οι τιμές του best_bound_per_depth στο πρόβλημα μεγιστοποίησης είναι οι μικρότερες δυνατές επειδή ψάχνουμε να βρούμε την μεγαλύτερη τιμή
        # για κάθε βάθος του δέντρου αναζήτησης
        # Αντίστοιχα, οι τιμές του best_bound_per_depth στο πρόβλημα ελαχιστοποίησης είναι οι μεγαλύτερες δυνατές επειδή ψάχνουμε να βρούμε την 
        # μικρότερη τιμή για κάθε βάθος του δέντρου αναζήτησης
        if isMax == True:
            best_bound_per_depth = np.array([-np.inf for i in range(num_vars)])
        else:
            best_bound_per_depth = np.array([np.inf for i in range(num_vars)])

        # Δημιουργία του πίνακα nodes_per_depth, ο οποίος καταγράφει τον αριθμό των κόμβων σε κάθε επίπεδο του δέντρου αναζήτησης του αλγορίθμου 
        # branch and bound
        # Το επίπεδο 0 (ρίζα) αρχικοποείται στο 1, ενώ για κάθε επόμενο βάθος ο αριθμός των κόμβων διπλασιάζεται σε σχέση με το προηγούμενο επίπεδο
        nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
        nodes_per_depth[0] = 1
        for i in range(1, num_vars + 1):
            nodes_per_depth[i] = nodes_per_depth[i - 1] * 2

        # Κλήση της συνάρτησης branch_and_bound που υλοποιεί τον αλγόριθμο branch and bound και χρονομέτρηση του χρόνου εκτέλεσης
        print("************************    Solving problem...    ************************")
        start = time.time()
        solutions, best_sol_idx, solutions_found = branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth,
                                                                nodes_per_depth)
        end = time.time()

        # Εκτύπωση των αποτελεσμάτων
        print("========= Optimal Solutions =========")        
        print("solutions:", solutions)
        print("best_sol_idx:", best_sol_idx)
        print(solutions[best_sol_idx][0])
        print(f"Objective Value: {solutions[best_sol_idx][1]}")
        print(f"Tree depth: {solutions[best_sol_idx][2]}")
        print()
        print(solutions)
        print(f"Time Elapsed: {end - start}")
        print(f"Total nodes: {nodes}")

    """
    print(f"Currently processing problem file: problems\class_1\problem_1.dat\n")
    print("************************    Initializing structures...    ************************")

    # Κλήση της συνάρτησης capital_budgeting του αρχείου pr για την δημιουργία του μοντέλου
    model, ub, lb, integer_var, num_vars, c = pr.capital_budgeting("problems\class_1\problem_1.dat")

    # Πρόβλημα μεγιστοποίησης
    isMax = True 

    # Αρχικοποίηση του πίνακα best_bound_per_depth στο -άπειρο αν το πρόβλημα μας πρόκειται για πρόβλημα μεγιστοποίησης, 
    # αλλιώς αρχικοποίηση του πίνακα best_bound_per_depth στο άπειρο αν το πρόβλημα μας πρόκειται για πρόβλημα ελαχιστοποίησης
    # Οι τιμές του best_bound_per_depth στο πρόβλημα μεγιστοποίησης είναι οι μικρότερες δυνατές επειδή ψάχνουμε να βρούμε την μεγαλύτερη τιμή
    # για κάθε βάθος του δέντρου αναζήτησης
    # Αντίστοιχα, οι τιμές του best_bound_per_depth στο πρόβλημα ελαχιστοποίησης είναι οι μεγαλύτερες δυνατές επειδή ψάχνουμε να βρούμε την 
    # μικρότερη τιμή για κάθε βάθος του δέντρου αναζήτησης
    if isMax == True:
        best_bound_per_depth = np.array([-np.inf for i in range(num_vars)])
    else:
        best_bound_per_depth = np.array([np.inf for i in range(num_vars)])

    # Δημιουργία του πίνακα nodes_per_depth, ο οποίος καταγράφει τον αριθμό των κόμβων σε κάθε επίπεδο του δέντρου αναζήτησης του αλγορίθμου 
    # branch and bound
    # Το επίπεδο 0 (ρίζα) αρχικοποείται στο 1, ενώ για κάθε επόμενο βάθος ο αριθμός των κόμβων διπλασιάζεται σε σχέση με το προηγούμενο επίπεδο
    nodes_per_depth = np.zeros(num_vars + 1, dtype=float)
    nodes_per_depth[0] = 1
    for i in range(1, num_vars + 1):
        nodes_per_depth[i] = nodes_per_depth[i - 1] * 2

    # Κλήση της συνάρτησης branch_and_bound που υλοποιεί τον αλγόριθμο branch and bound και χρονομέτρηση του χρόνου εκτέλεσης
    print("************************    Solving problem...    ************************")
    start = time.time()
    solutions, best_sol_idx, solutions_found = branch_and_bound(model, ub, lb, integer_var, best_bound_per_depth,
                                                                nodes_per_depth)
    end = time.time()

    # Εκτύπωση των αποτελεσμάτων
    print("========= Optimal Solutions =========")        
    print("solutions:", solutions)
    print("best_sol_idx:", best_sol_idx)
    print(solutions[best_sol_idx][0])
    print(f"Objective Value: {solutions[best_sol_idx][1]}")
    print(f"Tree depth: {solutions[best_sol_idx][2]}")
    print()
    print(solutions)
    print(f"Time Elapsed: {end - start}")
    print(f"Total nodes: {nodes}")
    """