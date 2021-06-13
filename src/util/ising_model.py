from os import write
import numpy as np
from util.boolean_formula import Formula


class IsingModel:
    def __init__(self, beta = 1, mu = 1, interactions = []):
        self._numLatticeSites = len(interactions)
        self._beta = beta              # Inverse temperature
        self._mu = mu                # Field orientation
        self._interactions = interactions     # A square upper triangular matrix - diagonal entries are h, others are J

    def toWMC(self):
        """
        Creates a weighted model counting problem whose solution
        is the partition function of the Ising model.

        :return: A Formula which represents the weighted model counting
        problem 
        """
        form = Formula()
        varIds = [[0] * self._numLatticeSites] * self._numLatticeSites
        
        # Create variables for each lattice site and each non-zero pairwise interaction between them
        for i in range(self._numLatticeSites):
            varIds[i][i] = form.fresh_variable(np.exp(-1 * self._mu * self._beta * self._interactions[i][i]), np.exp(self._mu * self._beta * self._interactions[i][i]))
            for j in range(i + 1, self._numLatticeSites):
                varIds[i][j] = form.fresh_variable(np.exp(-1 * self._beta * self._interactions[i][j]), np.exp(self._beta * self._interactions[i][j]))
        
        # Add clauses for each pairwise interaction
        for i in range(self._numLatticeSites):
            for j in range(i + 1, self._numLatticeSites):
                if varIds[i][j] != 0:
                    form.add_clause([varIds[i][j], varIds[i][i], varIds[j][j]])
                    form.add_clause([varIds[i][j], -varIds[i][i], -varIds[j][j]])
                    form.add_clause([-varIds[i][j], varIds[i][i], -varIds[j][j]])
                    form.add_clause([-varIds[i][j], -varIds[i][i], varIds[j][j]])
        return form

    def to_UAI08(self, filename):
        out = open(filename, "w")

        ### PREAMBLE
        # First part of preamble
        out.write("ISING\n")
        out.write(str(self._numLatticeSites) + "\n")
        out.write("2 " * (self._numLatticeSites - 1) + "2\n")
        out.write(str(self.numUnaryFuncs()) + " " + str(self.numBinaryFuncs()) + " " + str(self._beta) + " " + str(self._mu) + "\n")

        # Write function inputs part of preamble 
        for i in range(self._numLatticeSites):
            if self._interactions[i][i] != 0:
                out.write("1 " + str(i) + "\n")
        for i in range(self._numLatticeSites):
            for j in range(i + 1, self._numLatticeSites):
                if self._interactions[i][j] != 0:
                    out.write("2 " + str(i) + " " + str(j) + "\n")

        out.write("\n")
        # FUNCTION TABLE
        for i in range(self._numLatticeSites):
            if self._interactions[i][i] != 0:
                out.write("2\n")
                out.write(" " + str(-self._interactions[i][i]) + " " + str(self._interactions[i][i]) + "\n")
        for i in range(self._numLatticeSites):
            for j in range(i + 1, self._numLatticeSites):
                if self._interactions[i][j] != 0:
                    out.write("4\n")
                    out.write(" " + str(self._interactions[i][j]) + " " + str(-self._interactions[i][j]) + "\n")
                    out.write(" " + str(-self._interactions[i][j]) + " " + str(self._interactions[i][j]) + "\n")
        out.close()


    @staticmethod
    def from_UAI08(filename):
        in_model = open(filename, "r")

        ### PREAMBLE
        # Read meta-information
        if not in_model.readline().startswith("ISING"):
            raise Exception("Misformated file " + filename)
        numLatticeSites = int(in_model.readline())
        in_model.readline()         # Disregard line specifying variable domains
        # The next line may contain optional fields
        metaline = [int(i) for i in in_model.readline().split()]
        num_h_vals, num_J_vals = metaline[:2]
        beta = 1
        mu = 1
        print(len(metaline))
        if len(metaline) > 2:
            beta = float(metaline[2])
            if len(metaline) > 3:
                mu = int(metaline[3])
                
        # Function inputs
        nonzero_h_slots = []
        nonzero_J_slots = []
        for i in range(num_h_vals):
            nonzero_h_slots.append(int(in_model.readline().split()[1]))
        for i in range(num_J_vals):
            nonzero_J_slots.append(tuple([int(j) for j in in_model.readline().split()[1:]]))


        in_model.readline()         # Read empty line
        ### FUNCTION TABLE
        interactions = [[0 for i in range(numLatticeSites)] for j in range(numLatticeSites)]
        # H values
        for i in nonzero_h_slots:
            if in_model.readline() != "2\n":
                #error
                pass
            interactions[i][i] = float(in_model.readline().split()[1])
        # J values
        for (i,j) in nonzero_J_slots:
            if in_model.readline() != "4\n":
                #error
                pass
            interactions[i][j] = float(in_model.readline().split()[0])
            in_model.readline()
        
        in_model.close()
        return(IsingModel(beta = beta, mu = mu, interactions = interactions))

    def numUnaryFuncs(self):
        unaryCount = 0
        for i in range(self._numLatticeSites):
            if self._interactions[i][i] != 0:
                unaryCount += 1
        return unaryCount

    def numBinaryFuncs(self):
        numBinaryFuncs = 0
        for i in range(self._numLatticeSites):
            for j in range(i + 1, self._numLatticeSites):
                if self._interactions[i][j] != 0:
                    numBinaryFuncs += 1
        return numBinaryFuncs
        
    