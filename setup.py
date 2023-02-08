import numpy as np
import sympy as sym
import scipy.sparse as sp
import scipy.sparse.linalg

# setup the matrix: derivative of equality constraints by completion vars
def hzc(n, m, mStart, mEnd, swInds, Rall, Xall, zij_sym, zji_sym):
    dh_dzc = np.zeros((6*m+2*(n-1), (n-1)+4*m+m))
    # z = [pg\f, vj, zij, yij]  # initial guess from NN
    # zc = [qg\f, pij, qij, pji, qji, zji]  # completion vars, where \f notation means excluding feeder
    # feeder power (p,q) variables will be calculated separately (at least for now)

    # constraints: {zij + zji - 1 = 0, zij + zji - yij = 0} | variable zji
    dh_dzc[0:m, (n-1)+4*m:(n-1)+5*m] = np.eye(m)

    # constraint: pg - pl - sum(pji - pij)  = 0
    # excluding feeder
    mAll = (mStart + mEnd).toarray()
    for r in range(1, n):
        lineInds = np.where(mAll[r, :])[0]
        # dimInds = np.size(lineInds)
        dh_dzc[m + r - 1, (n-1) + lineInds] = 1  # variable pij
        dh_dzc[m + r - 1, (n-1) + 2 * m + lineInds] = -1  # variable pji, sym.ones(1, dimInds)

    # constraint: qg - ql - sum(qji - qij)  = 0
    # exlcuding feeder
    for r in range(1, n):
        lineInds = np.where(mAll[r, :])[0]
        dh_dzc[6 * m + (n - 1) + r - 1, r] = 1  # variable qg
        dh_dzc[6 * m + (n - 1) + r - 1, (n-1) + m + lineInds] = 1  # variable qij
        dh_dzc[6 * m + (n - 1) + r - 1, (n-1) + 3 * m + lineInds] = -1  # variable qji

    # convert to symbolic, so can add the constraints for zij,zji
    dh_dzc_sym = sym.Matrix(dh_dzc)

    # constraint Pij = zij, Pji = zji, when zij,zji = 0 (binding inequality)
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to 2M-(N-1) constraints

    for r in range(m):
        dh_dzc_sym[m + (n - 1) + r, (n-1) + r] = 1 - zij_sym[r]  # for pij = zij | variable pij
        dh_dzc_sym[m + (n - 1) + m + r, (n-1) + 2 * m + r] = 1 - zji_sym[r]  # for pji = zji | variable pji
        dh_dzc_sym[m + (n - 1) + m + r, (n-1) + 4 * m + r] = zji_sym[r] - 1  # for pji = zji | variable zji

    # constraint: vj - vi + 2(R*(Pij - Pji) + Xij*(Qij - Qji)) = 0
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to (N-1) constraints
    for r in range(m):
        lineInds = np.where(mAll[r, :])[0].tolist()
        R = 2 * Rall[lineInds] * (zij_sym[r] + zji_sym[r])
        X = 2 * Xall[lineInds] * (zij_sym[r] + zji_sym[r])
        for pos, ind in enumerate(lineInds):  # loop through
            dh_dzc_sym[3 * m + (n - 1) + r, (n-1) + ind] = R[pos]  # variable pij
            dh_dzc_sym[3 * m + (n - 1) + r, (n-1) + 2*m + ind] = -R[pos]  # variable pji
            dh_dzc_sym[3 * m + (n - 1) + r, (n-1) + m + ind] = X[pos]  # variable qij
            dh_dzc_sym[3 * m + (n - 1) + r, (n-1) + 3*m + ind] = -X[pos]  # variable qji

    # constraint Qij = zij, Qji = zji, when zij,zji = 0 (binding inequality)
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to 2M-(N-1) constraints
    for r in range(m):
        dh_dzc_sym[4 * m + (n - 1) + r, (n-1) + m + r] = 1 - zij_sym[r]  # for qij = zij | variable qij
        dh_dzc_sym[4 * m + (n - 1) + m + r, (n-1) + 3 * m + r] = 1 - zji_sym[r]  # for qji = zji | variable qji
        dh_dzc_sym[4 * m + (n - 1) + m + r, (n-1) + 4 * m + r] = zji_sym[r] - 1  # for qji = zji | variable zji

    # dh_dzc_sym.subs([(zij_sym, sym.Matrix([1, 0, 0, 1]).T), (zji_sym, sym.Matrix([0, 1, 0, 0]).T)])
    # dh_dzc_sym.subs({zij_sym: sym.Matrix([1,0,0,1]).T, zji_sym: sym.Matrix([0,1,0,0]).T})

    return dh_dzc_sym


# setup the matrix: derivative of equality constraints by guessed vars
def hz(n, m, mStartInd, mEndInd, swInds):
    numSwitches = np.size(swInds)

    # DOUBLE CHECK IF -1 ON ROWS IS NEEDED
    dh_dz = np.zeros((6 * m + 2 * (n - 1), n-1 + n + m + numSwitches))
    # z = [pg\f, vj, zij, yij]  # initial guess from NN
    # zc = [qg, pij, qij, pji, qji, zji]  # completion vars, where \f notation means excluding feeder

    # constraints: {zij + zji - 1 = 0, zij + zji - yij = 0}.
    dh_dz[0:m, n-1+n:n-1+n+m] = np.eye(m)  # | variable zij
    dh_dz[m-numSwitches:m, n-1+n+m:n-1+n+m+numSwitches] = -np.eye(numSwitches)  # | variable yij

    # constraint: pg - pl - sum(pji - pij)  = 0
    # excluding feeder
    dh_dz[m:m+n-1, 0:n-1] = np.eye(n-1)  # variable pg

    # constraint: qg - ql - sum(qji - qij)  = 0
    # excluding feeder - note needed to adjust count
    # no dependency on z vars

    # constraint Pij = zij, binding inequality when zij = 0
    # constraint Pji = zji, binding inequality when zij = 0 -- no dependency on z vars
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to 2M-(N-1) constraints
    dh_dz[m+(n-1):m+(n-1)+m, n-1 + n:n-1+n+m] = -np.eye(m)

    # constraint: vj - vi + 2(R*(Pij - Pji) + Xij*(Qij - Qji)) = 0
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to (N-1) constraints
    # dependency on vj, vi
    # mStartInd gives vi and mEndInd gives vj
    dh_dz[range(3*m+(n-1), 3*m+(n-1)+m), (n-1) + mStartInd-1] = -1  # | variable vi
    dh_dz[range(3*m+(n-1), 3*m+(n-1)+m), (n-1) + mEndInd-1] = 1  # | variable vj

    # constraint Qij = zij, binding inequality when zij = 0
    # constraint Qji = zji, binding inequality when zij = 0 -- no dependency on z vars
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to 2M-(N-1) constraints
    dh_dz[4 * m + (n - 1):4 * m + (n - 1) + m, n - 1 + n:n - 1 + n + m] = -np.eye(m)

    return dh_dz


# setup the matrix: to index the active equality constraints, based on {zij, zji} values
def index_dzc_dz(n, m, zij_sym, zji_sym):
    # z = [pg\f, vj, zij, yij]  # initial guess from NN
    # zc = [qg, pij, qij, pji, qji, zji]  # completion vars, where \f notation means excluding feeder
    tempmat1 = sym.zeros(m, m)
    tempmat2 = sym.zeros(m, m)
    tempmat3 = sym.zeros(m, m)
    for ind in range(m):
        tempmat1[ind, ind] = zij_sym[ind]
        tempmat2[ind, ind] = zji_sym[ind]
        tempmat3[ind, ind] = zij_sym[ind] + zji_sym[ind]

    ind_dzc_dz_sym = sym.diag(sym.eye(m + n - 1), tempmat1, tempmat2,
                          tempmat3, tempmat1, tempmat2, sym.eye(n-1))
    # ind_dzc_dz.subs([(zij_sym, sym.Matrix([1, 0, 0, 1]).T), (zji_sym, sym.Matrix([0, 1, 0, 0]).T)])
    # ind_dzc_dz = sym.diag(sym.eye(m+n-1), sym.diag([zij_sym[:]]), sym.diag(zji_sym[:]), sym.diag([zij_sym+zji_sym]), sym.diag(zij_sym[:]), sym.diag(zji_sym[:]), sym.eye(n))

    return ind_dzc_dz_sym

# setup the matrix: derivative of equality constraints by input vars (pl and ql)
def hx(n, m):
    dh_dx = np.zeros((6*m+2*(n-1), 2*(n-1)))
    # x = [pl\f, ql\f]
    # z = [pg\f, vj, zij, yij]  # initial guess from NN
    # zc = [qg\f, pij, qij, pji, qji, zji]  # completion vars, where \f notation means excluding feeder
    # feeder power (p,q) variables will be calculated separately (at least for now)

    # constraints: {zij + zji - 1 = 0, zij + zji - yij = 0}
    # no dependency

    # constraint: pg - pl - sum(pji - pij)  = 0 | variable pl
    # excluding feeder
    dh_dx[m:m+n-1, 0:(n - 1)] = -np.eye(n-1)

    # constraint: qg - ql - sum(qji - qij)  = 0 | variable ql
    # exlcuding feeder
    dh_dx[6 * m + (n - 1):6 * m + 2 * (n - 1), (n - 1) :None] = -np.eye(n-1)

    # constraint Pij = zij, Pji = zji, when zij,zji = 0 (binding inequality)
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to 2M-(N-1) constraints
    # no dependency

    # constraint: vj - vi + 2(R*(Pij - Pji) + Xij*(Qij - Qji)) = 0
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to (N-1) constraints
    # no dependency

    # constraint Qij = zij, Qji = zji, when zij,zji = 0 (binding inequality)
    # there will be 2M of these, dependent on the zij,zji values. When known, it is reduced to 2M-(N-1) constraints
    # no dependency

    return dh_dx


# test = sym.diag(sym.eye(m+n-1), sym.diag([zij_sym[:]]))
# test.subs([(zij_sym, sym.Matrix([1, 0, 0, 1]).T)])
# test.evalf(subs={zij_sym: sym.Matrix([1, 0, 0, 1]).T})
# expr = sin(x)/x
# expr.evalf(subs={x: 3.14})




#### NEW DEFINITIONS FOR TRAINING AND ALL CONSTRAINTS
# \f notation means excluding feeder
# x = [pl\f, ql\f]  # input to NN
# z = [zji, y\last, pij, pji, qji, {qij}sw, v\f, plf, qlf]  # initial guess from NN
# zc = [zij, ylast, {qij}nosw, pg, qg]  # completion vars
#### HAD TO FIX THE VARIABLE SELECTION - actually this is incorrect so we went back to the set above.
# \f notation means excluding feeder
# x = [pl\f, ql\f]  # input to NN
# z = [zji, y\last, pij, pji, qji, v\f, plf, qlf]  # initial guess from NN
# zc = [zij, ylast, qij pg, qg]  # completion vars

# DEBUGGED BELOW
def deq_dx(N, M, swInds):
    numsw = np.shape(swInds)[0]
    # deq_dz_mat = np.zeros((2*M + 2*N + 1 - numsw, 2*(N-1)))
    data = []  # value of the derivative
    row = []  # each equality constraint
    col = []  # each variable in x
    # zij + zji = 1  # no dependency
    # zij + zji = y  # no dependency
    # sum(y) = (N-1) - (M-#sw)  # no dependency
    # pg - pl = sum(pij - pji)  | {-1} for pl\f
    data = np.hstack((data, -np.ones(N-1)))
    row = np.hstack((row, np.arange(M+2, M+1+N)))
    col = np.hstack((col, np.arange(0, N-1)))
    # qg - ql = sum(qij - qji) | {-1} for ql\f
    data = np.hstack((data, -np.ones(N-1)))
    row = np.hstack((row, np.arange(M+N+2, M+2*N+1)))
    col = np.hstack((col, np.arange(N-1, 2*(N-1))))
    # vj - vi = -2[Rij(pij-pji) + Xij(qij-qji)] for M-#sw lines  # no dependency

    deq_dx_mat = sp.coo_matrix((data, (row, col)), shape=(2*M+2*N+1-numsw, 2*(N-1)))

    return deq_dx_mat


def deq_dx_Nov30(N, M, swInds):
    numsw = np.shape(swInds)[0]
    # deq_dz_mat = np.zeros((2*M + 2*N + 1 - numsw, 2*(N-1)))
    data = []  # value of the derivative
    row = []  # each equality constraint
    col = []  # each variable in x
    # zij + zji = 1  # no dependency
    # zij + zji = y  # no dependency
    # sum(y) = (N-1) - (M-#sw)  # no dependency
    # pg - pl = sum(pij - pji)  | {-1} for pl\f
    data = np.hstack((data, -np.ones(N-1)))
    row = np.hstack((row, np.arange(M+2, M+1+N)))
    col = np.hstack((col, np.arange(0, N-1)))
    # vj - vi = -2[Rij(pij-pji) + Xij(qij-qji)] for M-#sw lines  # no dependency
    # qg - ql = sum(qij - qji) | {-1} for ql\f
    data = np.hstack((data, -np.ones(N-1)))
    row = np.hstack((row, np.arange(0, N-1)+(M+1+N+M-numsw+1)))
    col = np.hstack((col, np.arange(N-1, 2*(N-1))))

    deq_dx_mat = sp.coo_matrix((data, (row, col)), shape=(2*M+2*N+1-numsw, 2*(N-1)))

    return deq_dx_mat

# DEBUGGED ABOVE
def deq_dz(N, M, swInds, A, Rall, Xall):
    numsw = np.shape(swInds)[0]
    # deq_dz_mat = np.zeros((2*M+2*N+1-numsw, M+numsw-1+3*M+numsw+N-1+2))
    data = []  # value of the derivative
    row = []  # each equality constraint
    col = []  # each variable in x
    # zij + zji = 1 | {1} for zji
    data = np.hstack((data, np.ones(M-numsw)))
    row = np.hstack((row, np.arange(0, M-numsw)))
    col = np.hstack((col, np.arange(0, M-numsw)))

    # zij + zji = y  | {1} for zji
    data = np.hstack((data, np.ones(numsw)))
    row = np.hstack((row, np.arange(M-numsw, M)))
    col = np.hstack((col, np.arange(M-numsw, M)))
    # zij + zji = y  | {-1} for y\last
    data = np.hstack((data, -np.ones(numsw-1)))
    row = np.hstack((row, np.arange(M-numsw, M-1)))
    col = np.hstack((col, np.arange(M, M+numsw-1)))

    # sum(y) = (N-1) - (M-#sw)  | {1} for y\last
    data = np.hstack((data, np.ones(numsw-1)))
    row = np.hstack((row, np.ones(numsw-1)*M))
    col = np.hstack((col, np.arange(M, M+numsw-1)))

    # pg - pl = sum(pij - pji)  | {-1} for plf
    data = np.hstack((data, -1))
    row = np.hstack((row, M+1))
    col = np.hstack((col, 2*numsw+4*M+N-2))
    # qg - ql = sum(qij - qji) | {-1} for qlf
    data = np.hstack((data, -1))
    row = np.hstack((row, 2*M+1))
    col = np.hstack((col, 2*numsw+4*M+N-1))
    # pg - pl = sum(pij - pji)  | pij and pji
    # qg - ql = sum(qij - qji)  | {qij}_sw and qji
    for r in range(0, N):
        loc = np.argwhere(A[r, :] != 0)[:, 1]  # location of Pij,Qij and Pji,Qji
        val = A[r, loc].toarray()[0]  # -val is derivative for Pij and +val is derivative for Pji
        # Pij and Pji
        data = np.hstack((data, -val, val))
        row = np.hstack((row, (M+1+r)*np.ones(2*np.shape(loc)[0])))
        col = np.hstack((col, M+numsw-1+loc, M+numsw-1+M+loc))
        # {Qij}_sw and Qji
        val_sw = val[loc >= M - numsw]
        loc_sw = loc[loc >= M-numsw]
        data = np.hstack((data, val, -val_sw))   # -val_sw is derivative for {Qij}_sw and +val is derivative for Qji
        row = np.hstack((row, (M+1+N+r)*np.ones(np.shape(loc)[0] + np.shape(loc_sw)[0])))
        col = np.hstack((col, 3*M+numsw-1+loc, 4*M+numsw-1+loc_sw-(M-numsw)))

    # vj - vi = -2[Rij(pij-pji) + Xij(qij-qji)] for M-#sw lines
    # order of data: vj, vi, pij, pji, qji
    # vj index: np.argwhere(A.T[0:M-numsw] == -1)
    # vi index: np.argwhere(A.T[0:M-numsw] == 1)
    # pij index: np.arange(0,M-numsw)
    # pji index: np.arange(0,M-numsw)
    # qji index: np.arange(0,M-numsw)
    numelem_vj = np.size(np.argwhere(A.T[0:M-numsw, 1:None] == -1), 0)
    numelem_vi = np.size(np.argwhere(A.T[0:M-numsw, 1:None] == 1), 0)
    data = np.hstack((data, np.ones(numelem_vj),
                      -np.ones(numelem_vi),
                      2*Rall[0:M-numsw],
                      -2*Rall[0:M-numsw],
                      -2*Xall[0:M-numsw]))
    lines = np.arange(0, M-numsw) + M + 2*N + 1
    row = np.hstack((row, lines, lines, lines,
                     2*N+1+M+np.argwhere(A.T[-numsw:None, 1:None] == -1)[:, 0],
                     2*N+1+M+np.argwhere(A.T[-numsw:None, 1:None] == 1)[:, 0]))
    col = np.hstack((col, np.argwhere(A.T[0:M-numsw, 1:None] == -1)[:, 1] + 4*M+2*numsw-1,
                     np.argwhere(A.T[0:M-numsw, 1:None] == 1)[:, 1] + 4*M+2*numsw-1,
                     np.arange(0, M - numsw) + M+numsw-1,
                     np.arange(0, M - numsw) + 2*M+numsw-1,
                     np.arange(0, M - numsw) + 3*M+numsw-1))

    deq_dz_mat = sp.coo_matrix((data, (row, col)), shape=(2*M+2*N+1-numsw, 4*M+2*numsw+N))

    return deq_dz_mat


def deq_dz_Nov30(N, M, swInds, A, Rall, Xall):
    numsw = np.shape(swInds)[0]
    # deq_dz_mat = np.zeros((2*M+2*N+1-numsw, M+numsw-1+3*M+numsw+N-1+2))
    data = []  # value of the derivative
    row = []  # each equality constraint
    col = []  # each variable in x
    # zij + zji = 1 | {1} for zji
    data = np.hstack((data, np.ones(M-numsw)))
    row = np.hstack((row, np.arange(0, M-numsw)))
    col = np.hstack((col, np.arange(0, M-numsw)))

    # zij + zji = y  | {1} for zji
    data = np.hstack((data, np.ones(numsw)))
    row = np.hstack((row, np.arange(M-numsw, M)))
    col = np.hstack((col, np.arange(M-numsw, M)))
    # zij + zji = y  | {-1} for y\last
    data = np.hstack((data, -np.ones(numsw-1)))
    row = np.hstack((row, np.arange(M-numsw, M-1)))
    col = np.hstack((col, np.arange(M, M+numsw-1)))

    # sum(y) = (N-1) - (M-#sw)  | {1} for y\last
    data = np.hstack((data, np.ones(numsw-1)))
    row = np.hstack((row, M*np.ones(numsw-1)))  # this is a single constraint  # [M]*(numsw-1)
    col = np.hstack((col, np.arange(M, M+numsw-1)))

    # pg - pl = sum(pij - pji)  | {-1} for plf
    data = np.hstack((data, -1))
    row = np.hstack((row, M+1))
    col = np.hstack((col, 2*numsw+4*M+N-2))

    # qg - ql = sum(qij - qji) | {-1} for qlf
    data = np.hstack((data, -1))
    row = np.hstack((row, 2*M+N+1-numsw))
    col = np.hstack((col, 2*numsw+4*M+N-1))

    # pg - pl = sum(pij - pji)  | pij and pji
    # qg - ql = sum(qij - qji)  | {qij}_sw and qji
    for r in range(0, N):
        loc = np.argwhere(A[r, :] != 0)[:, 1]  # location of Pij,Qij and Pji,Qji
        val = A[r, loc].toarray()[0]  # -val is derivative for Pij and +val is derivative for Pji
        # Pij and Pji
        data = np.hstack((data, -val, val))
        row = np.hstack((row, (M+1+r)*np.ones(2*np.shape(loc)[0])))
        col = np.hstack((col, M+numsw-1+loc, M+numsw-1+M+loc))
        # {Qij}_sw and Qji for Q-balance
        val_sw = val[loc >= M - numsw]
        loc_sw = loc[loc >= M - numsw]
        data = np.hstack((data, val, -val_sw))   # -val_sw is derivative for {Qij}_sw and +val is derivative for Qji
        row = np.hstack((row, (2*M+1+N-numsw+r)*np.ones(np.shape(loc)[0] + np.shape(loc_sw)[0])))
        col = np.hstack((col, 3*M+numsw-1+loc, 4*M+numsw-1+loc_sw-(M-numsw)))

    # vj - vi = -2[Rij(pij-pji) + Xij(qij-qji)] for M-#sw lines
    # order of data: vj, vi, pij, pji, qji
    # vj index: np.argwhere(A.T[0:M-numsw] == -1)
    # vi index: np.argwhere(A.T[0:M-numsw] == 1)
    # pij index: np.arange(0,M-numsw)
    # pji index: np.arange(0,M-numsw)
    # qji index: np.arange(0,M-numsw)
    vj_info = np.argwhere(A.T[0:M-numsw, 1:None] == -1)
    numelem_vj = np.size(vj_info, 0)
    line_vj = vj_info[:, 0]  # non-sw line number
    loc_vj = vj_info[:, 1]

    vi_info = np.argwhere(A.T[0:M - numsw, 1:None] == 1)
    numelem_vi = np.size(vi_info, 0)
    line_vi = vi_info[:, 0]  # non-sw line number
    loc_vi = vi_info[:, 1]
    data = np.hstack((data, np.ones(numelem_vj),
                      -np.ones(numelem_vi),
                      2*Rall[0:M-numsw],
                      -2*Rall[0:M-numsw],
                      -2*Xall[0:M-numsw]))
    line_rows = np.arange(0, M-numsw) + M + N + 1
    # Nov30: made a change to the row index here for vj and vi
    row = np.hstack((row, line_rows[line_vj], line_rows[line_vi],
                     line_rows, line_rows, line_rows))
    # 2*N+1+M+np.argwhere(A.T[-numsw:None, 1:None] == -1)[:, 0], 2*N+1+M+np.argwhere(A.T[-numsw:None, 1:None] == 1)[:, 0])
    col = np.hstack((col, loc_vj + 4*M+2*numsw-1,
                     loc_vi + 4*M+2*numsw-1,
                     np.arange(0, M - numsw) + M+numsw-1,
                     np.arange(0, M - numsw) + 2*M+numsw-1,
                     np.arange(0, M - numsw) + 3*M+numsw-1))

    deq_dz_mat = sp.coo_matrix((data, (row, col)), shape=(2*M+2*N+1-numsw, 4*M+2*numsw+N)) # size is OK

    return deq_dz_mat



# DEBUGGED BELOW
def deq_dzc(N, M, swInds, A, Xall):
    numsw = np.shape(swInds)[0]
    # deq_dzc_mat = np.zeros((2*M+2*N+1-numsw, M+1+(M-numsw)+2*N))
    data = []  # value of the derivative
    row = []  # each equality constraint
    col = []  # each variable in x
    # zij + zji = 1  # {1} for zij
    data = np.hstack((data, np.ones(M-numsw)))
    row = np.hstack((row, np.arange(0, M-numsw)))
    col = np.hstack((col, np.arange(0, M-numsw)))

    # zij + zji = y  # {1} for zij, {-1} for ylast
    data = np.hstack((data, np.ones(numsw), -1))
    row = np.hstack((row, np.arange(M-numsw, M), M-1))
    col = np.hstack((col, np.arange(M-numsw, M), M))

    # sum(y) = (N-1) - (M-#sw)  # {1} for ylast
    data = np.hstack((data, 1))
    row = np.hstack((row, M))
    col = np.hstack((col, M))

    # pg - pl = sum(pij - pji)  | {1} for pg
    data = np.hstack((data, np.ones(N)))
    row = np.hstack((row, np.arange(M+1, M+N+1)))
    col = np.hstack((col, M+1+M-numsw+np.arange(0, N)))

    # qg - ql = sum(qij - qji) | {1} for qg, {-1} for {qij}_nosw
    data = np.hstack((data, np.ones(N)))
    row = np.hstack((row, np.arange(M+N+1, M+2*N+1)))
    col = np.hstack((col, M+1+M-numsw+N+np.arange(0, N)))
    for r in range(0, N):
        loc = np.argwhere(A[r, :] != 0)[:, 1]  # location of Qij
        val = A[r, loc].toarray()[0]
        val_nosw = val[loc < M - numsw]  # {Qij}_sw
        loc_nosw = loc[loc < M-numsw]  # {Qij}_sw
        data = np.hstack((data, -val_nosw))   # -val_nosw is derivative for {Qij}_nosw
        row = np.hstack((row, M+N+1+r*np.ones(np.shape(loc_nosw)[0])))
        col = np.hstack((col, M+1+loc_nosw))

    # vj - vi = -2[Rij(pij-pji) + Xij(qij-qji)] for M-#sw lines  # {Qij}_nosw
    data = np.hstack((data, 2*Xall[0:M-numsw]))
    row = np.hstack((row, np.arange(M+1+2*N, M+1+2*N+M-numsw)))
    col = np.hstack((col, np.arange(M+1, 2*M+1-numsw)))

    deq_dzc_mat = sp.coo_matrix((data, (row, col)), shape=(2*M+2*N+1-numsw, M+1+(M-numsw)+2*N))

    return deq_dzc_mat


def deq_dzc_Nov30(N, M, swInds, A, Xall):
    numsw = np.shape(swInds)[0]
    # deq_dzc_mat = np.zeros((2*M+2*N+1-numsw, M+1+(M-numsw)+2*N))
    data = []  # value of the derivative
    row = []  # each equality constraint
    col = []  # each variable in x
    # zij + zji = 1  # {1} for zij
    data = np.hstack((data, np.ones(M-numsw)))
    row = np.hstack((row, np.arange(0, M-numsw)))
    col = np.hstack((col, np.arange(0, M-numsw)))

    # zij + zji = y  # {1} for zij, {-1} for ylast
    data = np.hstack((data, np.ones(numsw), -1))
    row = np.hstack((row, np.arange(M-numsw, M), M-1))
    col = np.hstack((col, np.arange(M-numsw, M), M))

    # sum(y) = (N-1) - (M-#sw)  # {1} for ylast
    data = np.hstack((data, 1))
    row = np.hstack((row, M))
    col = np.hstack((col, M))

    # pg - pl = sum(pij - pji)  | {1} for pg
    data = np.hstack((data, np.ones(N)))
    row = np.hstack((row, np.arange(0, N)+M+1))
    col = np.hstack((col, M+1+M-numsw+np.arange(0, N)))

    # qg - ql = sum(qij - qji) | {1} for qg, {-1} for {qij}_nosw
    data = np.hstack((data, np.ones(N)))
    row = np.hstack((row, np.arange(0, N)+2*M+N+1-numsw))
    col = np.hstack((col, M+1+M-numsw+N+np.arange(0, N)))
    for r in range(0, N):
        loc = np.argwhere(A[r, :] != 0)[:, 1]  # location of Qij
        val = A[r, loc].toarray()[0]
        val_nosw = val[loc < M - numsw]  # {Qij}_sw
        loc_nosw = loc[loc < M-numsw]  # {Qij}_sw
        data = np.hstack((data, -val_nosw))   # -val_nosw is derivative for {Qij}_nosw
        row = np.hstack((row, 2*M+N+1-numsw+r*np.ones(np.shape(loc_nosw)[0])))
        col = np.hstack((col, M+1+loc_nosw))

    # vj - vi = -2[Rij(pij-pji) + Xij(qij-qji)] for M-#sw lines  # {Qij}_nosw
    data = np.hstack((data, 2*Xall[0:M-numsw]))
    row = np.hstack((row, np.arange(0, M-numsw)+M+N+1))
    col = np.hstack((col, np.arange(0, M-numsw)+M+1))

    deq_dzc_mat = sp.coo_matrix((data, (row, col)), shape=(2*M+2*N+1-numsw, M+1+(M-numsw)+2*N))

    return deq_dzc_mat


def dineq_dz(N, M, swInds, Rall, Xall, bigM, A):
    numsw = np.shape(swInds)[0]
    # deq_dz_mat = np.zeros((,))
    data = []  # value of the derivative
    row = []  # each equality constraint
    col = []  # each variable in x

    # pg - pgUpp <= 0, pgLow - pg <= 0  | no dependency
    # qg - qgUpp <= 0, qgLow - qg <= 0  | no dependency
    # -pgf <= 0, -qgf <= 0  | no dependency
    # -plf <= 0, -qlf <= 0  | plf and qlf
    data = np.hstack((data, -1, -1))
    row = np.hstack((row, 4*(N-1)+2, 4*(N-1)+3))
    col = np.hstack((col, 4*M+2*numsw+N-2, 4*M+2*numsw+N-1))

    # v - vUpp <= 0, vLow - v <= 0  | v
    data = np.hstack((data, np.ones(N-1), -np.ones(N-1)))
    row = np.hstack((row, np.arange(4*(N-1)+4,4*(N-1)+4+2*(N-1))))
    col = np.hstack((col, np.arange(0, N-1)+4*M+2*numsw-1, np.arange(0, N-1)+4*M+2*numsw-1))

    # Pij - M*zij <= 0, - Pij <= 0  | Pij
    data = np.hstack((data, np.ones(M), -np.ones(M)))
    row = np.hstack((row, 6*(N-1)+4+np.arange(0, M), 6*(N-1)+4+M+np.arange(0, M)))
    col = np.hstack((col, np.arange(0, M)+M+numsw-1, np.arange(0, M)+M+numsw-1))

    # Pji - M*zji <= 0, - Pji <= 0  | Pji, zji
    data = np.hstack((data, np.ones(M), -np.ones(M), -bigM*np.ones(M)))
    row = np.hstack((row, 6*(N-1)+4+2*M+np.arange(0, M), 6*(N-1)+4+3*M+np.arange(0, M), 6*(N-1)+4+2*M+np.arange(0, M)))
    col = np.hstack((col, np.arange(0, M)+2*M+numsw-1, np.arange(0, M)+2*M+numsw-1, np.arange(0, M)))

    # Qij - M*zij <= 0, - Qij <= 0  | {Qij}_sw
    data = np.hstack((data, np.ones(numsw), -np.ones(numsw)))
    row = np.hstack((row, 6*(N-1)+4+5*M-numsw+np.arange(0, numsw), 6*(N-1)+4+6*M-numsw+np.arange(0, numsw)))
    col = np.hstack((col, np.arange(0, numsw)+4*M+numsw-1, np.arange(0, numsw)+4*M+numsw-1))

    # Qji - M*zji <= 0, - Qji <= 0  | Qji, zji
    data = np.hstack((data, np.ones(M), -np.ones(M), -bigM * np.ones(M)))
    row = np.hstack((row, 6*(N-1)+4+6*M+np.arange(0, M), 6*(N-1)+4+7*M+np.arange(0, M), 6*(N-1)+4+6*M+np.arange(0, M)))
    col = np.hstack((col, np.arange(0, M)+3*M+numsw-1, np.arange(0, M)+3*M+numsw-1, np.arange(0, M)))

    # -zij <= 0  | no dependency
    # -zji <= 0  | {-1} for zji
    data = np.hstack((data, -np.ones(M)))
    row = np.hstack((row, 6*(N-1)+4+9*M+np.arange(0, M)))
    col = np.hstack((col, np.arange(0, M)))

    # -y <= 0  | y\last
    data = np.hstack((data, -np.ones(numsw-1)))
    row = np.hstack((row, 6*(N-1)+4+10*M+np.arange(0, numsw-1)))
    col = np.hstack((col, M+np.arange(0, numsw-1)))

    # vj - vi + 2[Rij(pij-pji) + Xij(qij-qji)] - M*(1-y) <= 0 for #sw lines  | y\last, Pij, Pji, Qji, {Qij}_sw, vj, vi
    # vj index: np.argwhere(A.T[-numsw:None] == -1)
    # vi index: np.argwhere(A.T[-numsw:None] == 1)
    numelem_vj = np.size(np.argwhere(A.T[-numsw:None, 1:None] == -1), 0)
    numelem_vi = np.size(np.argwhere(A.T[-numsw:None, 1:None] == 1), 0)
    data = np.hstack((data, bigM*np.ones(numsw-1),
                      2*Rall[-numsw:None],
                      -2 * Rall[-numsw:None],
                      -2 * Xall[-numsw:None],
                      2 * Xall[-numsw:None],
                      np.ones(numelem_vj),
                      -np.ones(numelem_vi)))
    row_inds = 6*(N-1)+4+10*M+numsw+np.arange(0, numsw)
    row = np.hstack((row, row_inds[0:-1], row_inds, row_inds, row_inds, row_inds,
                     6*(N-1)+4+10*M+numsw+np.argwhere(A.T[-numsw:None, 1:None] == -1)[:, 0],
                     6*(N-1)+4+10*M+numsw+np.argwhere(A.T[-numsw:None, 1:None] == 1)[:, 0]))
    col = np.hstack((col, M+np.arange(0, numsw-1),
                     2*M-1+np.arange(0, numsw),  #M+numsw-1+np.arange(numsw, M) - PROBLEM HERE: SHOULD BE 7, IS 30?
                     3*M-1+np.arange(0, numsw),  #2*M+numsw-1+np.arange(numsw, M) - PROBLEM HERE: SHOULD BE 7, IS 30?
                     4*M-1+np.arange(0, numsw),  #3*M+numsw-1+np.arange(numsw, M) - PROBLEM HERE: SHOULD BE 7, IS 30?
                     4*M+numsw-1+np.arange(0, numsw),
                     4*M+2*numsw-1+np.argwhere(A.T[-numsw:None, 1:None] == -1)[:, 1],
                     4*M+2*numsw-1+np.argwhere(A.T[-numsw:None, 1:None] == 1)[:, 1]))  # exclude feeder voltage from derivative

    # -vj + vi - 2[Rij(pij-pji) + Xij(qij-qji)] - M*(1-y) <= 0 for #sw lines  | y\last, Pij, Pji, Qji, {Qij}_sw, vj, vi
    numelem_vj = np.size(np.argwhere(A.T[-numsw:None, 1:None] == -1), 0)
    numelem_vi = np.size(np.argwhere(A.T[-numsw:None, 1:None] == 1), 0)
    data = np.hstack((data, bigM*np.ones(numsw-1),
                      -2*Rall[-numsw:None],
                      2 * Rall[-numsw:None],
                      2 * Xall[-numsw:None],
                      -2 * Xall[-numsw:None],
                      -np.ones(numelem_vj),
                      np.ones(numelem_vi)))
    row_inds = 6*(N-1)+4+10*M+2*numsw+np.arange(0, numsw)

    row = np.hstack((row, row_inds[0:-1], row_inds, row_inds, row_inds, row_inds,
                     6*(N-1)+4+10*M+2*numsw+np.argwhere(A.T[-numsw:None, 1:None] == -1)[:, 0],
                     6*(N-1)+4+10*M+2*numsw+np.argwhere(A.T[-numsw:None, 1:None] == 1)[:, 0]))
    col = np.hstack((col, M+np.arange(0, numsw-1),
                     2 * M - 1 + np.arange(0, numsw),  # M+numsw-1+np.arange(numsw, M) - PROBLEM HERE: SHOULD BE 7, IS 30?
                     3 * M - 1 + np.arange(0, numsw),  # 2*M+numsw-1+np.arange(numsw, M) - PROBLEM HERE: SHOULD BE 7, IS 30?
                     4 * M - 1 + np.arange(0, numsw),  # 3*M+numsw-1+np.arange(numsw, M) - PROBLEM HERE: SHOULD BE 7, IS 30?
                     4*M+numsw-1+np.arange(0, numsw),
                     4*M+2*numsw-1+np.argwhere(A.T[-numsw:None, 1:None] == -1)[:, 1],
                     4*M+2*numsw-1+np.argwhere(A.T[-numsw:None, 1:None] == 1)[:, 1]))

    # - sum(zij + zji) <= -1 for each N (connectivity)
    data = np.hstack((data, -np.ones(2*M)))
    row = np.hstack((row, np.argwhere(A != 0)[:, 0] + 6*(N-1)+4+10*M+3*numsw))
    col = np.hstack((col, np.argwhere(A != 0)[:, 1]))

    dineq_dz_mat = sp.coo_matrix((data, (row, col)), shape=(6*(N-1)+4+10*M+3*numsw+N, 4*M+2*numsw+N))
    return dineq_dz_mat


def dineq_dzc(N, M, swInds, bigM, A):
    numsw = np.shape(swInds)[0]
    # deq_dz_mat = np.zeros((,))
    data = []  # value of the derivative
    row = []  # each equality constraint
    col = []  # each variable in x

    # pg - pgUpp <= 0, pgLow - pg <= 0  | pg
    # qg - qgUpp <= 0, qgLow - qg <= 0  | qg
    # -pgf <= 0, -qgf <= 0  | pgf and qgf
    data = np.hstack((data, np.ones(N-1), -np.ones(N-1),
                      np.ones(N-1), -np.ones(N-1),
                      -np.ones(2)))
    row = np.hstack((row, np.arange(0, 4*(N-1)+2)))
    col = np.hstack((col, 2*M+1-numsw+np.arange(1, N), 2*M+1-numsw+np.arange(1, N),
                     2*M+1-numsw+N+np.arange(1, N), 2*M+1-numsw+N+np.arange(1, N),
                     2*M+1-numsw, 2*M+1-numsw+N))

    # -plf <= 0, -qlf <= 0  | no dependency
    # v - vUpp <= 0, vLow - v <= 0  | no dependency
    # Pij - M*zij <= 0, - Pij <= 0 | zij
    data = np.hstack((data, -bigM*np.ones(M)))
    row = np.hstack((row, 6*(N-1)+4+np.arange(0, M)))
    col = np.hstack((col, np.arange(0, M)))

    # Pji - M*zji <= 0, - Pji <= 0  | no dependency
    # Qij - M*zij <= 0, - Qij <= 0  | zij, {Qij}_nosw
    data = np.hstack((data, -bigM*np.ones(M),
                      np.ones(M-numsw), -np.ones(M-numsw)))
    row = np.hstack((row, 6*(N-1)+4+4*M+np.arange(0, M),
                     6*(N-1)+4+4*M+np.arange(0, M-numsw), 6*(N-1)+4+5*M+np.arange(0, M-numsw)))
    col = np.hstack((col, np.arange(0, M),
                     M+1+np.arange(0, M-numsw), M+1+np.arange(0, M-numsw)))

    # Qji - M*zji <= 0, - Qji <= 0  | no dependency
    # -zij <= 0  | zij
    data = np.hstack((data, -np.ones(M)))
    row = np.hstack((row, 6*(N-1)+4+8*M+np.arange(0, M)))
    col = np.hstack((col, np.arange(0, M)))

    # -zji <= 0  | no dependency
    # -y <= 0  | ylast
    data = np.hstack((data, -1))
    row = np.hstack((row, 6*(N-1)+4+10*M+numsw-1))
    col = np.hstack((col, M))

    # vj - vi + 2[Rij(pij-pji) + Xij(qij-qji)] - M*(1-y) <= 0 for #sw lines  | no dependency
    # -vj + vi - 2[Rij(pij-pji) + Xij(qij-qji)] - M*(1-y) <= 0 for #sw lines  | no dependency
    # TODO: What about ylast?
    data = np.hstack((data, 1, 1))
    row = np.hstack((row, 6*(N-1)+4+10*M+numsw+numsw-2, 6*(N-1)+4+10*M+numsw+2*numsw-2))
    col = np.hstack((col, M, M))

    # - sum(zij + zji) <= -1 for each N (connectivity)    | zij
    data = np.hstack((data, -np.ones(2 * M)))
    row = np.hstack((row, np.argwhere(A != 0)[:, 0] + 6*(N-1)+4+10*M+3*numsw))
    col = np.hstack((col, np.argwhere(A != 0)[:, 1]))

    # data = np.hstack((data,))
    # row = np.hstack((row,))
    # col = np.hstack((col,))

    dineq_dzc_mat = sp.coo_matrix((data, (row, col)), shape=(6*(N-1)+4+10*M+3*numsw+N, M+1+(M-numsw)+2*N))
    return dineq_dzc_mat


def dzc_vars(N, M, swInds, A, Rall, Xall):
    deq_dx_mat = deq_dx_Nov30(N, M, swInds).todense()
    deq_dz_mat = deq_dz_Nov30(N, M, swInds, A, Rall, Xall).todense()
    deq_dzc_mat = deq_dzc_Nov30(N, M, swInds, A, Xall)

    dzc_deq_mat = sp.linalg.inv(deq_dzc_mat).todense()   # this is where the huge numbers originate

    dzc_dz_mat = -np.matmul(dzc_deq_mat, deq_dz_mat)  # matrix multiplication for sparse matrices  # this is where the huge numbers are retained in the model
    dzc_dx_mat = -np.matmul(dzc_deq_mat, deq_dx_mat)  # matrix multiplication for sparse matrices
    return dzc_dz_mat, dzc_dx_mat
