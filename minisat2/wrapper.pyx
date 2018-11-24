cdef extern from "SolverTypes.h" namespace "Minisat":
  ctypedef int Var
  struct Lit:
    int x
    Lit mkLit(Var, bool)
