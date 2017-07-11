import cplex
from hyG import *
from itertools import combinations


def get_connected_separations(og):
    """

    :param og: A HyG object
    :return: connected separators
    """
    ne = og.primal.edges()
    xs = list()
    for e in ne:
        xs.append(tuple(e))
    # print type(xs)
    sep = list()
    for n in range(2, len(xs) / 2 + 1):
        for e1 in combinations(xs, n):
            g1 = nx.Graph()
            g2 = nx.Graph()
            # print tuple([tuple(set(xs)-set(e1)),e1])
            # print e1
            g1.add_edges_from(e1)
            # print "g1", g1.edges()
            if not nx.is_connected(g1):
                # print "g1"
                # show_graph(g1,1)
                continue
            g2.add_edges_from(list(set(xs) - set(e1)))
            # print "g2", g2.edges(),set(xs)-set(e1)
            if not nx.is_connected(g2):
                # print "g2"
                # show_graph(g2, 1)
                continue
            if tuple(set(xs) - set(e1)) not in sep:
                sep.append(e1)
                # print "next"
    return sep, xs


class Ilp:
    def __init__(self, og1, nv, ne):
        self.qv = dict()
        self.varlist = list()
        self.u = dict()
        self.ng = 0
        self.z = self.ng
        self.t = dict()
        self.z1 = dict()
        self.nr = 1
        for i in range(nv):
            self.qv[i] = list()
            # print og1.primal.edges(i)
            for e in og1.g.edges(i):
                self.qv[i].append(e[1] - nv)
        self.st = nx.complete_multipartite_graph(ne, ne - 2)
        self.m = [i for i in range(ne)]
        self.n = [i for i in range(ne, ne + ne - 2)]
        self.b = self.st.edges()
        for i in range(ne - 1, ne + ne - 2):
            for j in range(i + 1, ne + ne - 2):
                self.st.add_edge(i, j)
        self.a = list(set(self.st.edges()) - set(self.b))
        self.varlist.append('z')
        self.ng += 1
        # print "A",A,"\nz",z
        for i in self.a:
            i1 = tuple(sorted(i))
            # print i1
            self.u[i1] = self.ng
            self.ng += 1
            self.varlist.append('u' + str(i1[0]) + '_' + str(i1[1]))
        # print "u",u
        for i1 in self.b:
            i = tuple(sorted(i1))
            self.t[i] = self.ng
            self.ng += 1
            self.varlist.append('t' + str(i[0]) + '_' + str(i[1]))
        # print "t",t
        for i1 in self.a:
            i = tuple(sorted(i1))
            # print i
            self.z1[i] = dict()
            for v in range(nv):
                # print i,v
                self.z1[i][v] = self.ng
                self.ng += 1
                self.varlist.append('z' + str(i[0]) + '_' + str(i[1]) + '_' + str(v))
                # print "z1",t_z1

    def first_sum(self, prob):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        # First sum i in N t(e,i)=1 e in M
        for e in range(len(self.m)):
            row.append(list())
            name = list()
            formula = list()
            for i in self.n:
                name.append(self.t[tuple(sorted([self.m[e], i]))])
                formula.append(1.0)
            row[len(row) - 1].append(name)
            row[len(row) - 1].append(formula)
            rows.append('r_' + str(self.nr))
            self.nr += 1
            rhs.append(1)
            sense.append("E")
            prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def second_sum(self, prob):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        row.append(list())
        name = list()
        formula = list()
        for e in self.a:
            name.append(self.u[e])
            formula.append(1.0)
        row[len(row) - 1].append(name)
        row[len(row) - 1].append(formula)
        rows.append("r_" + str(self.nr))
        self.nr += 1
        rhs.append(len(self.m) - 3)
        sense.append("E")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def third_sum(self, prob):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            row.append(list())
            name = list()
            formula = list()
            for j in self.n:
                if i != j:
                    name.append(self.u[tuple(sorted([i, j]))])
                    formula.append(1.0)
            for j in self.m:
                name.append(self.t[tuple(sorted([i, j]))])
                formula.append(1.0)
            row[len(row) - 1].append(name)
            row[len(row) - 1].append(formula)
            rows.append("r_" + str(self.nr))
            self.nr += 1
            rhs.append(3)
            sense.append("E")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def fourth_sum(self, prob):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            row.append(list())
            name = list()
            formula = list()
            for j in self.m:
                name.append(self.t[tuple(sorted([i, j]))])
                formula.append(1.0)
            row[len(row) - 1].append(name)
            row[len(row) - 1].append(formula)
            rows.append("r_" + str(self.nr))
            self.nr += 1
            rhs.append(2)
            sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def fifth_sum(self, prob, edge):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            for j in self.m:
                for k in self.m:
                    name = list()
                    formula = list()
                    if j != k:
                        if not set(edge[j]).intersection(set(edge[k])):
                            row.append(list())
                            name.append(self.t[tuple(sorted([i, j]))])
                            formula.append(1.0)
                            name.append(self.t[tuple(sorted([i, k]))])
                            formula.append(1.0)
                            row[len(row) - 1].append(name)
                            row[len(row) - 1].append(formula)
                            rows.append("r_" + str(self.nr))
                            self.nr += 1
                            rhs.append(1)
                            sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def sixth_sum(self, prob):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in range(len(self.n) - 1):
            row.append(list())
            name = list()
            formula = list()
            for j in range(len(self.m)):
                name.append(self.t[tuple(sorted([self.n[i], self.m[j]]))])
                name.append(self.t[tuple(sorted([self.n[i + 1], self.m[j]]))])
                formula.append(pow(2, j) + 1)
                formula.append(-1 * (pow(2, j) + 1))
            row[len(row) - 1].append(name)
            row[len(row) - 1].append(formula)
            rows.append("r_" + str(self.nr))
            self.nr += 1
            rhs.append(0)
            sense.append("G")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def tenth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            for j in set(self.n) - {i}:
                for v in range(nv):
                    name = list()
                    formula = list()
                    name.append(self.z1[tuple(sorted([i, j]))][v])
                    formula.append(1.0)
                    name.append(self.u[tuple(sorted([i, j]))])
                    formula.append(-1.0)
                    row.append(list())
                    row[len(row) - 1].append(name)
                    row[len(row) - 1].append(formula)
                    rhs.append(0)
                    rows.append('r_' + str(self.nr))
                    self.nr += 1
                    sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def eleventh_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            for v in range(nv):
                for e in self.qv[v]:
                    name = list()
                    formula = list()
                    name.append(self.t[tuple(sorted([e, i]))])
                    formula.append(1.0)
                    for j in set(self.n) - {i}:
                        name.append(self.z1[tuple(sorted([i, j]))][v])
                        formula.append((-1.0))
                    row.append(list())
                    row[len(row) - 1].append(name)
                    row[len(row) - 1].append(formula)
                    rhs.append(0)
                    rows.append('r_' + str(self.nr))
                    self.nr += 1
                    sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def thirteenth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for v in range(nv):
            name = list()
            formula = list()
            for i in self.a:
                name.append(self.z1[i][v])
                formula.append(1.0)
            row.append(list())
            row[len(row) - 1].append(name)
            row[len(row) - 1].append(formula)
            rhs.append(len(self.qv[v]) - 2)
            rows.append('r_' + str(self.nr))
            self.nr += 1
            sense.append("G")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def fifteenth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            for j in set(self.n) - {i}:
                name = list()
                formula = list()
                for v in range(nv):
                    name.append(self.z1[tuple(sorted([i, j]))][v])
                    formula.append(1.0)
                name.append(self.z)
                formula.append(-1.0)
                row.append(list())
                row[len(row) - 1].append(name)
                row[len(row) - 1].append(formula)
                rhs.append(0)
                rows.append('r_' + str(self.nr))
                self.nr += 1
                sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def make_vars(self, prob):
        my_ub = []
        my_lb = []
        my_obj = []
        for i in range(self.ng):
            my_ub.append(1)
            my_lb.append(0)
            my_obj.append(0)
        my_obj[self.z] = 1.0
        my_ub[self.z] = 3.0
        my_lb[self.z]=3.0
        types = [prob.variables.type.binary] * self.ng
        types[self.z] = prob.variables.type.integer
        prob.variables.add(obj=my_obj, ub=my_ub, names=self.varlist, types=types,lb=my_lb)


class First(Ilp):
    def __init__(self, og1, nv, ne):
        Ilp.__init__(self, og1, nv, ne)
        self.y = dict()
        self.q = dict()
        for i1 in self.a:
            i = tuple(sorted(i1))
            self.y[i] = dict()
            for v in range(nv):
                # print v,qv[v]
                for j in range(len(self.qv[v])):
                    for k in set(self.qv[v]) - {self.qv[v][j]}:
                        self.y[i][tuple(sorted([self.qv[v][j], k]))] = self.ng
                        self.ng += 1
                        self.varlist.append('y' + str(i[0]) + '_' + str(i[1]) + '_' + str(j) + '_' + str(k))
                        # print('y' + str(i[0]) + '_' + str(i[1]) + '_' + str(j) + '_' + str(k))
                        # print t_y[i]
        # print "y",y
        for i in self.n:
            self.q[i] = dict()
            for v in range(nv):
                for j in range(len(self.qv[v])):
                    for k in set(self.qv[v]) - {self.qv[v][j]}:
                        self.q[i][tuple(sorted([self.qv[v][j], k]))] = self.ng
                        self.ng += 1
                        self.varlist.append('q' + str(i) + '_' + str(j) + '_' + str(k))
                        # print('q' + str(i) + '_' + str(i) + '_' + str(j) + '_' + str(k))
                        # print "q",q

    def seventh_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for v in range(nv):
            for e1 in range(len(self.qv[v])):
                for f1 in range(e1 + 1, len(self.qv[v])):
                    for i in self.n:
                        e = self.qv[v][e1]
                        f = self.qv[v][f1]
                        name = list()
                        formula = list()
                        name.append(self.t[tuple(sorted([e, i]))])
                        name.append(self.t[tuple(sorted([f, i]))])
                        formula.append(1.0)
                        formula.append(1.0)
                        name.append(self.q[i][tuple(sorted([e, f]))])
                        formula.append(-2.0)
                        for j in set(self.n) - {i}:
                            # j=N[j1]
                            # print i,j,e,f
                            name.append(self.y[tuple(sorted([i, j]))][tuple(sorted([e, f]))])
                            formula.append(1.0)
                        # print formula
                        row.append(list())
                        row[len(row) - 1].append(name)
                        row[len(row) - 1].append(formula)
                        rhs.append(0)
                        rows.append('r_' + str(self.nr))
                        self.nr += 1
                        sense.append("E")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def eighth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            for j in set(self.n) - {i}:
                for v in range(nv):
                    for e1 in range(len(self.qv[v])):
                        for f1 in range(e1 + 1, len(self.qv[v])):
                            e = self.qv[v][e1]
                            f = self.qv[v][f1]
                            name = list()
                            formula = list()
                            name.append(self.y[tuple(sorted([i, j]))][tuple(sorted([e, f]))])
                            formula.append(1.0)
                            name.append(self.q[i][tuple(sorted([e, f]))])
                            formula.append(-1.0)
                            row.append(list())
                            row[len(row) - 1].append(name)
                            row[len(row) - 1].append(formula)
                            rhs.append(0)
                            rows.append('r_' + str(self.nr))
                            self.nr += 1
                            sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def ninth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            for j in set(self.n) - {i}:
                for v in range(nv):
                    for e1 in range(len(self.qv[v])):
                        for f1 in range(e1 + 1, len(self.qv[v])):
                            e = self.qv[v][e1]
                            f = self.qv[v][f1]
                            name = list()
                            formula = list()
                            name.append(self.y[tuple(sorted([i, j]))][tuple(sorted([e, f]))])
                            formula.append(1.0)
                            # print i,j,v,z1[tuple(sorted([i,j]))]
                            name.append(self.z1[tuple(sorted([i, j]))][v])
                            formula.append(-1.0)
                            row.append(list())
                            row[len(row) - 1].append(name)
                            row[len(row) - 1].append(formula)
                            rhs.append(0)
                            rows.append('r_' + str(self.nr))
                            self.nr += 1
                            sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def twelfth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        # print self.qv
        for i1 in range(len(self.n)):
            for j1 in range(i1 + 1, len(self.n)):
                i = self.n[i1]
                j = self.n[j1]
                name = list()
                formula = list()
                name.append(self.u[tuple(sorted([i, j]))])
                formula.append(1.0)
                for v in range(nv):
                    for e1 in range(len(self.qv[v])):
                        for f1 in range(e1 + 1, len(self.qv[v])):
                            e = self.qv[v][e1]
                            f = self.qv[v][f1]
                            # print v,t_qv[v],e1,f1,i,j,e,f
                            name.append(self.y[tuple(sorted([i, j]))][tuple(sorted([e, f]))])
                            formula.append(-1.0)
                # print name
                row.append(list())
                row[len(row) - 1].append(name)
                row[len(row) - 1].append(formula)
                rhs.append(0)
                rows.append('r_' + str(self.nr))
                self.nr += 1
                sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def fourteenth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for v in range(nv):
            for e1 in range(len(self.qv[v])):
                for f1 in range(e1 + 1, len(self.qv[v])):
                    e = self.qv[v][e1]
                    f = self.qv[v][f1]
                    name = list()
                    formula = list()
                    for i in self.n:
                        name.append(self.q[i][tuple(sorted([e, f]))])
                        formula.append(1)
                    for g in self.a:
                        name.append(self.y[g][tuple(sorted([e, f]))])
                        formula.append(-1.0)
                    row.append(list())
                    row[len(row) - 1].append(name)
                    row[len(row) - 1].append(formula)
                    rhs.append(1)
                    rows.append('r_' + str(self.nr))
                    self.nr += 1
                    sense.append("E")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)
    #
    # def make_vars(self, prob):
    #     my_ub = []
    #     for i in range(self.ng):
    #         my_ub.append(1)
    #     my_obj = []
    #     for i in range(self.ng):
    #         my_obj.append(0)
    #     my_obj[self.z] = 1.0
    #     my_ub[self.z] = cplex.infinity
    #     types = [prob.variables.type.binary] * self.ng
    #     types[self.z] = prob.variables.type.integer
    #     prob.variables.add(obj=my_obj, ub=my_ub, names=self.varlist, types=types)

    def make_prob(self, edge, nv):
        prob = cplex.Cplex()
        prob.set_error_stream('err.error')
        prob.objective.set_sense(prob.objective.sense.minimize)
        self.make_vars(prob)
        self.first_sum(prob)
        self.second_sum(prob)
        self.third_sum(prob)
        self.fourth_sum(prob)
        self.fifth_sum(prob, edge)
        self.sixth_sum(prob)
        self.seventh_sum(prob, nv)
        self.eighth_sum(prob, nv)
        self.ninth_sum(prob, nv)
        self.tenth_sum(prob, nv)
        self.eleventh_sum(prob, nv)
        self.twelfth_sum(prob, nv)
        self.thirteenth_sum(prob, nv)
        self.fourteenth_sum(prob, nv)
        self.fifteenth_sum(prob, nv)
        prob.write('form.lp')
        prob.solve()
        # print prob.solution.get_values(['t0_5', 't0_6', 't0_7', 't1_5', 't1_6', 't1_7', 't2_5', 't2_6', 't2_7', 't3_5', 't3_6', 't3_7', 't4_5', 't4_6', 't4_7'])
        # print 't0_5', 't0_6', 't0_7', 't1_5', 't1_6', 't1_7', 't2_5', 't2_6', 't2_7', 't3_5', 't3_6', 't3_7', 't4_5', 't4_6', 't4_7'
        # print prob.solution.get_values(['z5_6_0', 'z5_6_1', 'z5_6_2', 'z5_6_3', 'z5_6_4', 'z5_7_0', 'z5_7_1', 'z5_7_2', 'z5_7_3', 'z5_7_4', 'z6_7_0', 'z6_7_1', 'z6_7_2', 'z6_7_3', 'z6_7_4'])
        # print 'z5_6_0', 'z5_6_1', 'z5_6_2', 'z5_6_3', 'z5_6_4', 'z5_7_0', 'z5_7_1', 'z5_7_2', 'z5_7_3', 'z5_7_4', 'z6_7_0', 'z6_7_1', 'z6_7_2', 'z6_7_3', 'z6_7_4'
        # print self.varlist
        print prob.solution.get_values('z')
        # prob.solve()
        # prob.write('formula.lp')


class Second(Ilp):
    def __init__(self, og1, nv, ne):
        Ilp.__init__(self, og1, nv, ne)
        self.x = dict()
        for i in self.n:
            self.x[i] = dict()
            for v in range(nv):
                self.x[i][v] = self.ng
                self.ng += 1
                self.varlist.append('x' + str(i) + '_' + str(v))

    def sixteenth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            for v in range(nv):
                name = list()
                formula = list()
                for e in self.qv[v]:
                    name.append(self.t[tuple(sorted([e, i]))])
                    formula.append(1.0)
                for j in set(self.n) - {i}:
                    name.append(self.z1[tuple(sorted([i, j]))][v])
                    formula.append(1.0)
                name.append(self.x[i][v])
                formula.append(-2.0)
                row.append(list())
                row[len(row) - 1].append(name)
                row[len(row) - 1].append(formula)
                rhs.append(1)
                rows.append('r_' + str(self.nr))
                self.nr += 1
                sense.append("L")
        # print len(row),len(rows),len(rhs),len(sense)
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def seventeenth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for e in self.a:
            for v in range(nv):
                name = list()
                formula = list()
                name.append(self.z1[tuple(sorted([e[0], e[1]]))][v])
                formula.append(1.0)
                name.append(self.x[e[0]][v])
                formula.append(-1.0)
                row.append(list())
                row[len(row) - 1].append(name)
                row[len(row) - 1].append(formula)
                rhs.append(0)
                rows.append('r_' + str(self.nr))
                self.nr += 1
                sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def eighteenth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for i in self.n:
            for v in range(nv):
                for e in self.qv[v]:
                    name = list()
                    formula = list()
                    name.append(self.t[tuple(sorted([e, i]))])
                    formula.append(1.0)
                    name.append(self.x[i][v])
                    formula.append(-1.0)
                    row.append(list())
                    row[len(row) - 1].append(name)
                    row[len(row) - 1].append(formula)
                    rhs.append(0)
                    rows.append('r_' + str(self.nr))
                    self.nr += 1
                    sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def nineteenth_sum(self, prob, nv):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for v in range(nv):
            name = list()
            formula = list()
            for i in self.n:
                name.append(self.x[i][v])
                formula.append(1.0)
            for i in self.n:
                for j in set(self.n) - {i}:
                    if self.z1[tuple(sorted([i, j]))][v] not in name:
                        name.append(self.z1[tuple(sorted([i, j]))][v])
                        formula.append(-1.0)
            # print name
            row.append(list())
            row[len(row) - 1].append(name)
            row[len(row) - 1].append(formula)
            rhs.append(1)
            rows.append('r_' + str(self.nr))
            self.nr += 1
            sense.append("E")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)


    def make_prob(self, edge, nv):
        prob = cplex.Cplex()
        prob.set_error_stream('err.error')
        prob.objective.set_sense(prob.objective.sense.minimize)
        self.make_vars(prob)
        self.first_sum(prob)
        self.second_sum(prob)
        self.third_sum(prob)
        self.fourth_sum(prob)
        self.fifth_sum(prob, edge)
        self.sixth_sum(prob)
        self.tenth_sum(prob, nv)
        self.eleventh_sum(prob, nv)
        self.thirteenth_sum(prob, nv)
        self.fifteenth_sum(prob, nv)
        self.sixteenth_sum(prob, nv)
        self.seventeenth_sum(prob, nv)
        self.eighteenth_sum(prob, nv)
        self.nineteenth_sum(prob, nv)
        prob.write('form.lp')
        prob.solve()

        # print prob.solution.get_values(['t0_5', 't0_6', 't0_7', 't1_5', 't1_6', 't1_7', 't2_5', 't2_6', 't2_7', 't3_5', 't3_6', 't3_7', 't4_5', 't4_6', 't4_7'])
        # print 't0_5', 't0_6', 't0_7', 't1_5', 't1_6', 't1_7', 't2_5', 't2_6', 't2_7', 't3_5', 't3_6', 't3_7', 't4_5', 't4_6', 't4_7'
        # print prob.solution.get_values(['z5_6_0', 'z5_6_1', 'z5_6_2', 'z5_6_3', 'z5_6_4', 'z5_7_0', 'z5_7_1', 'z5_7_2', 'z5_7_3', 'z5_7_4', 'z6_7_0', 'z6_7_1', 'z6_7_2', 'z6_7_3', 'z6_7_4'])
        # print 'z5_6_0', 'z5_6_1', 'z5_6_2', 'z5_6_3', 'z5_6_4', 'z5_7_0', 'z5_7_1', 'z5_7_2', 'z5_7_3', 'z5_7_4', 'z6_7_0', 'z6_7_1', 'z6_7_2', 'z6_7_3', 'z6_7_4'
        # print self.varlist
        print prob.solution.get_values('z')
        # prob.solve()
        # prob.write('formula.lp')


class Third:
    def __init__(self, og, ne):
        self.S, self.ed = get_connected_separations(og)
        print len(self.S)
        # print self.S
        self.x = dict()
        self.ng = 1
        self.z = 0
        self.varlist = list()
        self.varlist.append('z')
        self.nr = 1
        self.e = ne
        for e in range(len(self.S)):
            e1 = self.S[e]
            self.x[e1] = self.ng
            self.ng += 1
            self.varlist.append('x_' + str(e))
        self.f = dict()
        for e in range(len(self.S)):
            s1 = self.S[e]
            s2 = tuple(set(self.ed) - set(self.S[e]))
            f1 = list()
            f2 = list()
            for e1 in s1:
                for v in e1:
                    if v not in f1:
                        f1.append(v)
            for e1 in s2:
                for v in e1:
                    if v not in f2:
                        f2.append(v)
            self.f[s1] = len(set(f1).intersection(set(f2)))
            # print self.f

    def first_sum(self, prob):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        name = list()
        formula = list()
        for e in range(len(self.S)):
            s1 = self.S[e]
            name.append(self.x[s1])
            formula.append(1.0)
        row.append(list())
        row[len(row) - 1].append(name)
        row[len(row) - 1].append(formula)
        rhs.append(self.e - 3)
        rows.append('r_' + str(self.nr))
        self.nr += 1
        sense.append("E")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def second_sum(self, prob):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for s1 in self.S:
            name = list()
            formula = list()
            name.append(self.x[s1])
            formula.append(self.e - 4)
            for s2 in set(self.S) - {s1}:
                if set(s1).issubset(set(s2)) or set(s1).issuperset(set(s2)) or set(s1).issubset(
                                set(self.ed) - set(s2)) or set(s1).issuperset(set(self.ed) - set(s2)):
                    # print s1, s2
                    name.append(self.x[s2])
                    formula.append(-1.0)
            row.append(list())
            row[len(row) - 1].append(name)
            row[len(row) - 1].append(formula)
            rhs.append(0)
            rows.append('r_' + str(self.nr))
            self.nr += 1
            sense.append("L")
        print len(sense)
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def third_sum(self, prob):
        rows = list()
        row = list()
        rhs = list()
        sense = list()
        for s1 in self.S:
            name = list()
            formula = list()
            name.append(self.x[s1])
            formula.append(self.f[s1])
            name.append(self.z)
            formula.append(-1.0)
            row.append(list())
            row[len(row) - 1].append(name)
            row[len(row) - 1].append(formula)
            rhs.append(0)
            rows.append('r_' + str(self.nr))
            self.nr += 1
            sense.append("L")
        prob.linear_constraints.add(lin_expr=row, senses=sense, rhs=rhs, names=rows)

    def make_vars(self, prob):
        my_ub = []
        my_lb = []
        my_obj = []
        for i in range(self.ng):
            my_ub.append(1)
            my_lb.append(0)
            my_obj.append(0)
        my_obj[self.z] = 1.0
        my_ub[self.z] = 3.0
        my_lb[self.z]=3.0
        types = [prob.variables.type.binary] * self.ng
        types[self.z] = prob.variables.type.integer
        prob.variables.add(obj=my_obj, ub=my_ub, names=self.varlist, types=types,lb=my_lb)

    def make_prob(self):
        prob = cplex.Cplex()
        prob.set_error_stream('err.error')
        prob.objective.set_sense(prob.objective.sense.minimize)
        self.make_vars(prob)
        self.first_sum(prob)
        self.second_sum(prob)
        self.third_sum(prob)
        prob.write('form.lp')
        prob.solve()
        # print prob.solution.get_values(['t0_5', 't0_6', 't0_7', 't1_5', 't1_6', 't1_7', 't2_5', 't2_6', 't2_7', 't3_5', 't3_6', 't3_7', 't4_5', 't4_6', 't4_7'])
        # print 't0_5', 't0_6', 't0_7', 't1_5', 't1_6', 't1_7', 't2_5', 't2_6', 't2_7', 't3_5', 't3_6', 't3_7', 't4_5', 't4_6', 't4_7'
        # print prob.solution.get_values(['z5_6_0', 'z5_6_1', 'z5_6_2', 'z5_6_3', 'z5_6_4', 'z5_7_0', 'z5_7_1', 'z5_7_2', 'z5_7_3', 'z5_7_4', 'z6_7_0', 'z6_7_1', 'z6_7_2', 'z6_7_3', 'z6_7_4'])
        # print 'z5_6_0', 'z5_6_1', 'z5_6_2', 'z5_6_3', 'z5_6_4', 'z5_7_0', 'z5_7_1', 'z5_7_2', 'z5_7_3', 'z5_7_4', 'z6_7_0', 'z6_7_1', 'z6_7_2', 'z6_7_3', 'z6_7_4'
        # print self.varlist
        print prob.solution.get_values('z')
        # for i in range(16):
        #     print self.S[i],self.f[self.S[i]]
