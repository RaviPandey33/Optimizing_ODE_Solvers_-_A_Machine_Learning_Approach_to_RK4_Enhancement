import time
import datetime

# Number of iterations (adjust as needed for a 5-hour runtime)
iterations = 18000  # This is an example, assuming each iteration takes about 1 second
output_file = "output.txt"

# Open the file in write mode
with open(output_file, "w") as file:
    for i in range(iterations):
        # Get the current timestamp
        timestamp = datetime.datetime.now()
        
        # Write the iteration number and timestamp to the file
        file.write(f"Iteration {i}: {timestamp}\n")
        
        # Print to console (optional)
        print(f"Iteration {i}: {timestamp}")
        
        # Sleep for 1 second (simulate work)
        time.sleep(1)

print("Program completed.")
