# BLI

Dependencies:
- Python 2.7
- Networkx
- Matplotlib

Input:
The input has to be dimacs graph format. eg.

    p edge 3 3
    e 1 2
    e 2 3
    e 1 3

Execution:

Initially, we need to generate a heuristic decomposition 

    python main.py -i <input file without extension> -o 3 -f <working folder> -p <working folder>

The decomposition along with the width will be saved in the working folder.

To run the local improvement we will need to run following command

    python main.py -i  <input file without extension> -o 4 -f <working folder> -p <temp folder> -l <limit_on_size> -t <timeout_for_SAT_call> -m <radious for bfs>

The last three values are not mandatory.

Disclaimer: The code is not really in the best shape so if you have any question just drop a mail.
