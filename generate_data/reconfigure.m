
function [z, zc, yij_res, yaltime, solvertime, objval, err] = reconfigure(networkdata, casedata)
    % solve MILP using yalmip
    % LinDistFlow power flow model
    % objective function is line losses
    
    bigM = 10; % big M formulation of conditional constraints
    
    % initialize yalmip variables
    V = sdpvar(networkdata.N,1); % squared voltage magnitude
    PL = sdpvar(networkdata.N,1);
    QL = sdpvar(networkdata.N,1);
    PG = sdpvar(networkdata.N,1);
    QG = sdpvar(networkdata.N,1);
    Pij = sdpvar(networkdata.M,1);
    Qij = sdpvar(networkdata.M,1);
    Pji = sdpvar(networkdata.M,1);
    Qji = sdpvar(networkdata.M,1);
    zij = binvar(networkdata.M,1);
    zji = binvar(networkdata.M,1);
    yij = binvar(networkdata.numSwitches,1); % binary variable for switch status
    
    nosw_inds = setdiff(1:networkdata.M,networkdata.swInds);
    powerflow = [
        networkdata.A(nosw_inds, :)*V == -2*(diag(networkdata.R(nosw_inds))*(Pij(nosw_inds)-Pji(nosw_inds)) + diag(networkdata.X(nosw_inds))*(Qij(nosw_inds)-Qji(nosw_inds))),... % voltage rln
        networkdata.A(networkdata.swInds, :)*V <= -2*(diag(networkdata.R(networkdata.swInds))*(Pij(networkdata.swInds)-Pji(networkdata.swInds)) + diag(networkdata.X(networkdata.swInds))*(Qij(networkdata.swInds)-Qji(networkdata.swInds))) + bigM*(1-zij(networkdata.swInds)-zji(networkdata.swInds)),... % voltage rln
        networkdata.A(networkdata.swInds, :)*V >= -2*(diag(networkdata.R(networkdata.swInds))*(Pij(networkdata.swInds)-Pji(networkdata.swInds)) + diag(networkdata.X(networkdata.swInds))*(Qij(networkdata.swInds)-Qji(networkdata.swInds))) - bigM*(1-zij(networkdata.swInds)-zji(networkdata.swInds)),... % voltage rln
        PG - PL == networkdata.Aflow*(Pij - Pji),... % power balance, P
        QG - QL == networkdata.Aflow*(Qij - Qji)% power balance, Q
        ];
    
    varLims = [
        [0; casedata.PGLow] <= PG,...
        casedata.PGUpp >= PG(2:end),... % feeder PG is unbounded
        [0; casedata.QGLow] <= QG,...
        casedata.QGUpp >= QG(2:end),... % feeder QG is unbounded
        [1; ones(networkdata.N-1,1)*networkdata.vLow^2] <= V,... % includes slack voltage
        [1; ones(networkdata.N-1,1)*networkdata.vUpp^2] >= V % includes slack voltage
        ];
    
    load = [
        PL(2:end) == casedata.PL,...
        QL(2:end) == casedata.QL,...
        PL(1) >= 0,...  % need positivity of PL and PG for all nodes
        QL(1) >= 0  % need positivity of QL and QG for all nodes
        ];
    
    topological = [
        zij(nosw_inds) + zji(nosw_inds) == ones(networkdata.M-networkdata.numSwitches,1),...
        zij(networkdata.swInds) + zji(networkdata.swInds) == yij,...
        zij >= 0,...
        zji >= 0,...
        Pij >= 0,...
        Pji >= 0,...
        Pij <= bigM*zij,...
        Pji <= bigM*zji,...
        Qij >= 0,...
        Qji >= 0,...
        Qij <= bigM*zij,...
        Qji <= bigM*zji,...
        ];

    radiality = [
        abs(networkdata.Aflow)*(zij + zji) >= 1,...
        sum(yij) == (networkdata.N - 1) - (networkdata.M-networkdata.numSwitches)
        ];
    
    objfnc = sum(diag(networkdata.R)*(Pij.^2 + Pji.^2 + Qij.^2 + Qji.^2)); % line losses
    objfnc_voltage = sum((V-1).^2); % flat voltage profile
    % sum((y(VrInds)-ones(3*N*T,1)).^2 + (y(ViInds)-zeros(3*N*T,1)).^2)
    
    ops = sdpsettings('solver','Gurobi','verbose',0); % 0
	dgn = optimize([powerflow, varLims, load,topological, radiality], objfnc,ops); % _voltage
    
    err = dgn.problem;                
    if err ~= 0
%         error('ERROR: Cannot solve reconfiguration. \nYalmip returned %d: %s',err,yalmiperror(err))
        fprintf('\n ERROR: Cannot solve reconfiguration. \nYalmip returned %d: %s \n',err,yalmiperror(err))
        z = NaN(4*networkdata.M+2*networkdata.numSwitches+networkdata.N, 1);
        zc = NaN(2*networkdata.M-networkdata.numSwitches+2*networkdata.N+1, 1);
        yij_res =NaN; yaltime = NaN; solvertime = NaN; objval = NaN;
    else
        % rearrange the results into a form for the ML test/validation        
        netG_P = value(PG) - value(PL);
        netG_Q = value(QG) - value(QL);
        pg = value(PG);
        qg = value(QG);

        if netG_P(1) < 0
            plf = netG_P(1); 
            pg(1) = 0;
        else
            plf = 0;
            pg(1) = netG_P(1);
        end
        
        if netG_Q(1) < 0
            qlf = netG_Q(1); 
            qg(1) = 0;
        else
            qlf = 0;
            qg(1) = netG_Q(1);
        end
        
        % x = [PL, QL]
        % z = [zji, y\last, pij, pji, qji, {qij}_sw, v\f, plf, qlf]
        % zc = [zij, ylast, {qij}_nosw, pg, qg]
        z = value([zji; yij(1:end-1); Pij; Pji; Qji; 
            Qij(networkdata.M-networkdata.numSwitches+1:networkdata.M);
            V(2:end); plf; qlf]);
        zc = value([zij; yij(end); Qij(1:networkdata.M-networkdata.numSwitches); pg; qg]); 
        
        yij_res = value(yij);
    end
    
    yaltime = dgn.yalmiptime;
    solvertime = dgn.solvertime;
    objval = value(objfnc);
end