import os
import random

# Συνάρτηση generate_data_files για την δημιουργία φακέλων κλάσεων από αρχεία προβλημάτων, εντός ενός φακέλου
def generate_data_files(num_classes, problems_per_class, folder):

    # Δημιουργία φακέλου εφόσον δεν υπάρχει, αλλιώς εγγραφή από πάνω
    os.makedirs(folder, exist_ok=True)

    # Βρόχος για δημιουργία ενός φακέλου κλάσης στην κάθε επανάληψη
    for cls in range(1, num_classes + 1):

        # Δημιουργία φακέλου κλάσης εφόσον δεν υπάρχει, αλλιώς εγγραφή από πάνω
        class_folder = os.path.join(folder, f"class_{cls}")
        os.makedirs(class_folder, exist_ok=True)

        # Τυχαία ανάθεση τιμών στις παραμέτρους N, F, S και P
        N = random.randint(8000, 9000) # Αριθμός έργων (για εκτέλεση του branch_and_bound περίπου 5 με 6 ώρες)
        F = random.randint(1, 100000)  # Διαθέσιμο κεφάλαιο 
        S = random.randint(1, 1000)    # Διαθέσιμο προσωπικό 
        P = random.randint(1, N)       # Μέγιστος αριθμός έργων που μπορούν να υλοποιηθούν 
 
        # Βρόχος για δημιουργία ενός αρχείου προβλήματος στην κάθε επανάληψη
        for prob in range(1, problems_per_class + 1):

            # Δημιουργία αρχείου προβλήματος και άνοιγμα του κάθε αρχείου
            filename = os.path.join(class_folder, f"problem_{prob}.dat")
            with open(filename, "w") as f:
                
                # Εγγραφή των παραμέτρων N, F, S και P στο αρχείο 
                f.write(f"param N := {N};\n")
                f.write(f"param F := {F};\n")
                f.write(f"param S := {S};\n")
                f.write(f"param P := {P};\n\n")
                
                # Τυχαία ανάθεση τιμών στις παραμέτρους perfomance, cost και staff και εγγραφή τους στο αρχείο
                f.write("param: performance cost staff :=\n")
                for i in range(1, N + 1):
                    performance = random.randint(1, 100)  # Απόδοση
                    cost = random.randint(1, F)           # Κόστος
                    staff = random.randint(1, S)          # Προσωπικό
                    f.write(f"{i} {performance} {cost} {staff}\n")
                f.write(";\n")
    
    # Ενημέρωση για την επιτυχή δημιουργία
    print(f"Generated {num_classes * problems_per_class} problems in {folder}/")

# Κλήση της συνάρτησης generate_data_files για την δημιουργία 10 φακέλων κλάσεων από 10 αρχεία προβλημάτων στον καθένα, εντός του φακέλου problems
generate_data_files(10, 10, "problems")