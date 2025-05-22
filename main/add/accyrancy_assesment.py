import os 


def calculate_accyracy(log_file):
    total_correct = 0
    total_incorrect = 0
    total_files = 0
    correct_files = 0
    
    with open(log_file, 'r') as file:
        for line in file:
            parts = line.strip().split(",")
            correct = int(parts[1].split(":")[1])
            incorrect = int(parts[2].split(":")[1])
            correct_files = int(parts[3].split(":")[1])
            
            total_correct += correct
            total_incorrect += incorrect
            total_files += 1
            correct_files += correct_files
            
    total = total_correct + total_incorrect
    accyracy = (total_incorrect / total) * 100 if total > 0 else 0
    correct_files_accyracy = (correct_files / total_files) * 100 if total_files > 0 else 0
    
    print(f"Total correct: {total_correct}")
    print(f"Total incorrect: {total_incorrect}")
    print(f"Overlall accuracy: {accyracy:.2f}%")
    print(f"{correct_files} out of {total_files}")
    print(f"Correct files accyracy: {correct_files_accyracy:.2f}%")
    
    
if __name__ == "__main__":
    log_file = "TMP/accuracy_log.txt"
    if os.path.exists(log_file):
        calculate_accyracy(log_file)
    else: 
        print("No log file found (")
    

            
        