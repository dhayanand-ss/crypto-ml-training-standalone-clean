
import os

log_file = "integrity_results.txt"

def read_log():
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return
        
    print(f"Reading {log_file}...\n")
    try:
        # Try different encodings
        for enc in ['utf-16', 'utf-8', 'latin-1']:
            try:
                with open(log_file, "r", encoding=enc) as f:
                    content = f.read()
                    if content:
                        print(f"--- Content (using {enc}) ---")
                        print(content)
                        return
            except:
                continue
        print("Failed to read log with any encoding.")
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    read_log()
