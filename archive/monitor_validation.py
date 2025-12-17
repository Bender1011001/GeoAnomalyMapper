import time
import re
from pathlib import Path

def monitor_log(log_path):
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Log file {log_path} not found.")
        return

    print(f"Monitoring {log_path}...")
    
    folds_regex = re.compile(r"Fold (\d+)/(\d+)")
    sensitivity_regex = re.compile(r"LOOCV Sensitivity: (\d+\.\d+)%")
    
    last_fold = 0
    total_folds = 0
    
    try:
        with open(log_file, 'r') as f:
            # Simple tail implementation
            while True:
                line = f.readline()
                if line:
                    print(line.strip())
                    match = folds_regex.search(line)
                    if match:
                        current = int(match.group(1))
                        total = int(match.group(2))
                        if current > last_fold:
                            last_fold = current
                            total_folds = total
                    
                    if "VALIDATION RESULTS" in line:
                        print("\nValidation Complete! Analyzing results...")
                        
                    if sensitivity_regex.search(line):
                        print(f"\nFINAL SENSITIVITY: {sensitivity_regex.search(line).group(1)}%")
                        break
                else:
                    time.sleep(1)
                    
                    # Check if process is done (rough check via file mtime or just wait)
                    # For now just loop
    except KeyboardInterrupt:
        print("\nStopping monitor.")

if __name__ == "__main__":
    monitor_log("full_usa_validation.log")
