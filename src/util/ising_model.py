import numpy as np
from util.boolean_formula import Formula


class IsingModel:
    def __init__(self):
        self._numLatticeSites = 0
        self._beta = 1              # Inverse temperature
        self._mu = 1                # Field orientation
        self._interactions = []     # An upper triangular matrix - diagonal entries are h, others are J


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
            varIds[i,i] = form.fresh_variable(np.exp(-1 * self._mu * self._beta * self._interactions[i,i]), np.exp(self._mu * self._beta * self._interactions[i,i]))
            for j in range(i + 1, self._numLatticeSites):
                varIds[i,j] = form.fresh_variable(np.exp(-1 * self._beta * self._interactions[i,j]), np.exp(self._beta * self._interactions[i,j]))
        
        # Add clauses for each pairwise interaction
        for i in range(self._numLatticeSites):
            for j in range(i + 1, self._numLatticeSites):
                if varIds[i,j] != 0:
                    form.add_clause([varIds[i,j], varIds[i,i], varIds[j,j]])
                    form.add_clause([varIds[i,j], -varIds[i,i], -varIds[j,j]])
                    form.add_clause([-varIds[i,j], varIds[i,i], -varIds[j,j]])
                    form.add_clause([-varIds[i,j], -varIds[i,i], varIds[j,j]])
        return form
        
    # def parse?