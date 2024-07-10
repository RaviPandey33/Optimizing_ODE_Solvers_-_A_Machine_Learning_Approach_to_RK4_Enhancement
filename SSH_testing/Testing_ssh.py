import time
import datetime

# Number of iterations (adjust as needed for a 5-hour runtime)
iterations = 18  # This is an example, assuming each iteration takes about 1 second
output_file = "output.txt"

print("Script started. Opening file...")

try:
    # Open the file in append mode
    with open(output_file, "a") as file:
        print("File opened. Starting iterations...")
        for i in range(iterations):
            # Get the current timestamp
            timestamp = datetime.datetime.now()
            
            # Write the iteration number and timestamp to the file
            file.write(f"Iteration {i}: {timestamp}\n")
            
            # Flush the buffer to ensure data is written to the file
            file.flush()
            
            # Print to console (optional)
            print(f"Iteration {i}: {timestamp}")
            
            # Sleep for 1 second (simulate work)
            time.sleep(1)
except Exception as e:
    print(f"An error occurred: {e}")

print("Program completed.")