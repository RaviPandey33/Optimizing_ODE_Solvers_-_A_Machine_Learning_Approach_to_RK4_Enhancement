
## Log into the ssh connection

## Tmux steps :

Step 1: Check any tmux session

    tmux ls

Step 2: Kill any opeaned tmux session

    tmux kill-session -t old_session

Step 3: Opens up a new window (tmux windown)

    tmux new -s new_session

Step 4: Run the script

    python Test_file.py

Step 5: Press "Ctrl + b"  and then press "D"


Step 6: Check if the output file is updating

    tail -f output.txt



