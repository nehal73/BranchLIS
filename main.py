import getopt
import os
import sys
from Ilp import *
from lls import *
import hyG


def check_width(fname, width, time_out, depth=0, tmp=None):
    cmd_make = 'mkfifo ' + tmp + fname + str(width) + '.fifo'
    if depth != 0:
        cmd = 'timeout ' + str(time_out) + ' cat ' + fname + '.edge | ./hybw2sat ' + str(
            width) + ' ' + tmp + fname + str(width) + '.fifo ' + str(int(depth)) + ' &'
    else:
        cmd = 'timeout ' + str(time_out) + ' cat ' + fname + '.edge | ./hybw2sat ' + str(width) + ' ' + tmp + fname + str(
            width) + '.fifo &'
    print cmd
    cmd2 = './glucose -verb=0 ' + tmp + fname + str(width) + '.fifo  ' + tmp + fname + str(width) + '.sol'
    print cmd2
    cmd_remove = 'rm ' + tmp + fname + str(width) + '.fifo'
    os.system(cmd_make)
    os.system(cmd)
    os.system(cmd2)
    os.system(cmd_remove)
    sys.stdout.flush()
    try:
        fil = open(tmp + fname + str(width) + '.sol')
        s = fil.readline()
        print s[0]
        fil.close()
    except Exception as ex1:
        print ex1
        s = 'I'
    if s[0] == "U":
        return False,False
    else:
        return True,True


def find_width(fname, time_out, tmp=None):
    cmd = 'grep -e \'p edge\' ' + fname + '.edge > ' + tmp + fname + 'data'
    print cmd
    os.system(cmd)
    f = open(tmp + fname + 'data', 'r')
    l = f.readline()
    l = l.split()
    e = int(l[3])
    ind = False
    for width in range(1,e / 2+1):
        cmd_make = 'mkfifo ' + tmp + fname + str(width) + '.fifo'
        cmd = 'timeout ' + str(time_out) + ' cat ' + fname + '.edge | ./hybw2sat ' + str(width) + ' ' + tmp + fname + str(
            width) + '.fifo &'
        print cmd
        cmd2 = './glucose -verb=0 ' + tmp + fname + str(width) + '.fifo  ' + tmp + fname + str(width) + '.sol'
        print cmd2
        cmd_remove = 'rm '+ tmp + fname + str(width) + '.fifo'
        os.system(cmd_make)
        os.system(cmd)
        os.system(cmd2)
        os.system(cmd_remove)
        sys.stdout.flush()
        try:
            fil = open(tmp + fname + str(width) + '.sol')
            s = fil.readline()
            print s[0]
            fil.close()
        except Exception as ex1:
            print ex1
            s = 'I'
        # if s[0] == "U":
        #     print "Branchwidth of graph", fname, "is not", width
        if s[0] != 'U':
            return True,width,ind
        elif s[0] == "I":
            ind = True

def print1(cv,width,ind,file_name,time_out):
    if not cv:
        if not ind:
            print "Branchwidth for graph", file_name, "is not", width
        elif ind:
            print "Could not find if", width, "is branchwidth of graph", file_name, "in", time_out, "seconds"
    elif cv:
        if not ind:
            print "Branchwidth for graph", file_name, "is", width
        elif ind:
            print "Branchwidth for graph", file_name, "is atleast", width


def reduced_width(file_name,time_out,width,temp_path):
    width1 = 0
    cv1 = True
    ind1 = False
    G = reduce(file_name)
    for g in range(len(G)):
        if(G[g].number_of_edges()>=5):
            write_dimacs(G[g], file_name, g, temp_path)
            if width != 0:
                cv, ind = check_width(temp_path + file_name + str(g), width, time_out,temp_path)
            else:
                cv, width, ind = find_width(temp_path + file_name + str(g), time_out,temp_path)
            if not cv:
                cv1 = False
            if ind:
                ind1 = True
            if width > width1:
                width1 = width
        elif width1<2:
            width=2
    print1(cv1, width, ind1, file_name, time_out)


def reduce(file_name):
    edge,v=read_hyper_graph(file_name)
    g=nx.Graph()
    g.add_edges_from(edge)
    g1=list(nx.biconnected_component_subgraphs(g))
    for f1 in g1:
        reduction(f1)
    return g1


def write_dimacs(g,file_name,i,temp_path):
    file=open(temp_path+file_name+str(i)+'.edge','a')
    print(temp_path+file_name+str(i)+'.edge')
    file.write('p edge '+str(g.number_of_nodes())+' '+str(g.number_of_edges())+'\n')
    for e in g.edges():
        file.write('e '+str(e[0])+' '+str(e[1])+'\n')
    file.close()
    sys.stdout.flush()


def heuristic(file_name,tmp=None):
    """

    :rtype: None
    :type file_name: str
    :type tmp: str
    """
    cmd="./branchdecomp "+file_name+".edge "+tmp+file_name+".width "+tmp+file_name+".decomp"
    print cmd
    os.system(cmd)
    f=open(tmp+file_name+".width")
    s=f.readline()
    print "Heuristic Branchwidth of",file_name,":",s


def main(argv):
    temp_path = '/home/neha/temp/'
    file_name = 'eil51'
    limit_on_size = 60
    time_out = 600
    width = 0
    option = 1
    depth = 0
    maxdist=6
    folder='/home/neha/Dropbox/python/'
    try:
        opts, args = getopt.getopt(argv, "hi:l:t:p:o:w:d:f:m:", ["ifile=", "lim=","timeout=","temp=","opt=", "width=", "depth=","folder=","maxdist="])
    except getopt.GetoptError:
        print 'test.py -i <input_file(without_extension)> -l <limit_on_size> -t <timeout_for_SAT_call> -p <temp folder> -w <width> -d <depth> -f <workind directory> -m <radius for bfs>'
        sys.exit(2)
    if len(args)<2:
        print 'test.py -i <input_file(without_extension)> -l <limit_on_size> -t <timeout_for_SAT_call> -p <temp folder> -w <width> -d <depth> -f <workind directory> -m <radius for bfs>'
        sys.exit()
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <input_file(without_extension)> -l <limit_on_size> -t <timeout_for_SAT_call> -p <temp folder> -w <width> -d <depth> -f <workind directory> -m <radius for bfs>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            file_name = arg
        elif opt in ("-l", "--lim"):
            limit_on_size = int(arg)
            print 'limit', limit_on_size
        elif opt in ("-t", "--timeout"):
            time_out = int(arg)
        elif opt in ("-p", "--temp"):
            temp_path = arg
        elif opt in ("-o", "--opt"):
            option = int(arg)
        elif opt in ("-w", "--width"):
            width = int(arg)
        elif opt in ("-d", "--depth"):
            depth = int(arg)
        elif opt in ("-f","--folder"):
            folder=arg
        elif opt in ("-m","--maxdist"):
            maxdist=int(arg)
    print 'Input file is ', file_name, 'chosen algorithm',
    os.chdir(folder)
    if option == 1:
        print 'SAT encoding without reduction'
        if width != 0:
            cv,ind=check_width(file_name, width, time_out, depth=depth, tmp=temp_path)
        else:
            cv,width,ind=find_width(file_name, time_out, tmp=temp_path)
        print1(cv,width,ind,file_name,time_out)
    elif option == 3:
        print 'Hicks Heuristic without reduction'
        heuristic(file_name,temp_path)
    elif option == 2:
        print 'SAT encoding with reduction'
        reduced_width(file_name,time_out,width,temp_path)
    elif option==4:
        print 'Local Improvement'
        heuristic(file_name,temp_path)
        local_improve(file_name,limit_on_size,time_out,temp_path,second=1,max_dist=maxdist)


if __name__ == "__main__":
    main(sys.argv[1:])
