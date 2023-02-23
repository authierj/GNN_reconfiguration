clear
clc

%% Network data

SBase = 5000; % kVA
VBase = 4160; % V
ZBase = VBase^2/SBase/1000; % Ohm
YBase = 1/ZBase; % Siemens
IBase = SBase*1000/(sqrt(3)*VBase); 

% Topological data
N = 4;
M = 4; % total possible edges
swInds = [3,4]; % index of switched lines by topology (starting from 1)
numSwitches = length(swInds);
mStartInd = [1,2,1,2]; % parent node for line
mEndInd = [2,3,4,4]; % child node for line

% added for GNN
edge_index = [mStartInd; mEndInd]-1; % Graph connectivity in COO format with shape [2, num_edges] (directed graph)
bi_edge_index = [mStartInd, mEndInd; mEndInd, mStartInd]; % Graph connectivity in COO format with shape [2, 2*num_edges] (undirected graph)

% setup negative incidence matrix, A is MxN
mStart = sparse( mStartInd, 1:M, 1, N, M); % NxM
mEnd = sparse( mEndInd, 1:M, 1, N, M); % NxM
A = (mEnd - mStart)'; % MxN, for the voltage eqn
Aflow = -A'; % NxM, for the power flow constraint


% Network parameters, Z = R + jX
R = [0.35, 0.35, 0.35, 0.25]./ZBase;
X = [1, 1, 1, 0.8]./ZBase;

% Variable bounds
vLow = 0.95; % V, magnitude
vUpp = 1.05; % V, magnitude

% save network data
networkKeySet = {'SBase','VBase','ZBase','YBase','IBase','N','M','swInds','numSwitches','mStart','mEnd','A','Aflow','R','X','vLow','vUpp', 'bi_edge_index'};
networkValueSet = {SBase,VBase,ZBase,YBase,IBase,N,M,swInds,numSwitches,mStart,mEnd,A,Aflow,R,X,vLow,vUpp, bi_edge_index};
network_4_data = cell2struct(networkValueSet', networkKeySet);


%% Case data

% Baseline load data
% format: loadInd, PG, QG
loads = [
    2   200     100
    4	150     80
    ];

% Baseline generator data
% format: genInd, Plow, PUpp, QLow, QUpp
gens = [3   0   100     0   0];

% get load and gen data to generate perturbations
load_uniform = load_variations_uniform(); % ISONE load, same profile across all nodes, same P&Q
numcases = numel(load_uniform); 
[load_P, load_Q] = load_variations_perturbed(length(loads(:,1))); % gaussian perturbed profiles across nodes
pf = 1; % fixed pf for renewable generator inverter
[gen_P_rengen, gen_Q_rengen] = gen_variations_rengen(pf);
pf = 0.9; % fixed pf for renewable generator inverter
[gen_P_rengen_pf9, gen_Q_rengen_pf9] = gen_variations_rengen(pf);

% generate load cases
load_inds = loads(:,1);
loads_uniform_P = loads(:,2).*(1+reshape(load_uniform,1,[]));
loads_uniform_Q = loads(:,3).*(1+reshape(load_uniform,1,[]));

loads_pert_P = loads(:,2).*(1+load_P);
loads_pert_Q = loads(:,3).*(1+load_Q);

% higher load at node 4 than node 2
loads_pert_P_high = loads_pert_P;
loads_pert_Q_high = loads_pert_Q;
loads_pert_P_high(2,:) = loads_pert_P_high(2,:)*2;
loads_pert_Q_high(2,:) = loads_pert_Q_high(2,:)*2;


% generate gen cases
gen_inds = gens(:,1);

gen_constant_P_low = gens(:,2).*ones(length(gen_inds),numcases);
gen_constant_P_high = gens(:,3).*ones(length(gen_inds),numcases);
gen_constant_Q_low = gens(:,4).*ones(length(gen_inds),numcases);
gen_constant_Q_high = gens(:,5).*ones(length(gen_inds),numcases);

gen_rengen_P_low = gens(:,2).*gen_P_rengen(1:numcases)'; % for a year
gen_rengen_P_high = gens(:,3).*gen_P_rengen(1:numcases)';
gen_rengen_Q_low = gens(:,4).*gen_Q_rengen(1:numcases)';
gen_rengen_Q_high = gens(:,5).*gen_Q_rengen(1:numcases)';

gen_rengen_P_pf9_low = gens(:,2).*gen_P_rengen_pf9(1:numcases)'; % for a year
gen_rengen_P_pf9_high = gens(:,3).*gen_P_rengen_pf9(1:numcases)';
gen_rengen_Q_pf9_low = gens(:,4).*gen_Q_rengen_pf9(1:numcases)';
gen_rengen_Q_pf9_high = gens(:,5).*gen_Q_rengen_pf9(1:numcases)';

% generation exceeds demand test cases - increase PG and reduce PL
gen_export_P = gens(:,2);
gen_export_P_low = gens(:,2).*ones(length(gen_inds),numcases);
gen_export_P_high = gens(:,3).*ones(length(gen_inds),numcases)*1.5;
gen_export_Q_low = gens(:,4).*ones(length(gen_inds),numcases);
gen_export_Q_high = gens(:,5).*ones(length(gen_inds),numcases);
loads_pert_P_low = loads_pert_P.*0.5;
loads_pert_Q_low = loads_pert_Q.*0.5;


% assemble all the test cases
total_cases = 8*numcases;
PGLow = zeros(N-1,total_cases);
PGUpp = zeros(N-1,total_cases);
QGLow = zeros(N-1,total_cases);
QGUpp = zeros(N-1,total_cases);
PL = zeros(N-1,total_cases);
QL = zeros(N-1,total_cases);

% gen indices are minus one because we don't include the feeder node 1
% case 1: constant gen, uniform loads
ind = 1;
PGLow(gens(:,1)-1, ind:numcases) = gen_constant_P_low;
PGUpp(gens(:,1)-1, ind:numcases) = gen_constant_P_high;
QGLow(gens(:,1)-1, ind:numcases) = gen_constant_Q_low;
QGUpp(gens(:,1)-1, ind:numcases) = gen_constant_Q_high;
PL(loads(:,1)-1, ind:numcases) = loads_uniform_P;
QL(loads(:,1)-1, ind:numcases) = loads_uniform_Q;
% case 2: constant gen, perturbed loads
ind = ind + numcases;
PGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_constant_P_low;
PGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_constant_P_high;
QGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_constant_Q_low;
QGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_constant_Q_high;
PL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_P;
QL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_Q;
% case 3: ren gen, uniform loads
ind = ind + numcases;
PGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_P_low;
PGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_P_high;
QGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_Q_low;
QGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_Q_high;
PL(loads(:,1)-1, ind:ind+numcases-1) = loads_uniform_P;
QL(loads(:,1)-1, ind:ind+numcases-1) = loads_uniform_Q;
% case 4: ren gen, perturbed loads
ind = ind + numcases;
PGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_P_low;
PGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_P_high;
QGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_Q_low;
QGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_Q_high;
PL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_P;
QL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_Q;
% case 5: export gen, low perturbed loads
ind = ind + numcases;
PGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_export_P_low;
PGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_export_P_high;
QGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_export_Q_low;
QGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_export_Q_high;
PL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_P_low;
QL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_Q_low;
% case 6: constant gen, loads high at 4
ind = ind + numcases;
PGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_constant_P_low;
PGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_constant_P_high;
QGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_constant_Q_low;
QGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_constant_Q_high;
PL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_P;
QL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_Q;
% case 7: reg gen, loads high at 4
ind = ind + numcases;
PGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_P_low;
PGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_P_high;
QGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_Q_low;
QGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_Q_high;
PL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_P_high;
QL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_Q_high;
% case 8: reg gen with 0.9 pf, pert loads
ind = ind + numcases;
PGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_P_pf9_low;
PGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_P_pf9_high;
QGLow(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_Q_pf9_low;
QGUpp(gens(:,1)-1, ind:ind+numcases-1) = gen_rengen_Q_pf9_high;
PL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_P;
QL(loads(:,1)-1, ind:ind+numcases-1) = loads_pert_Q;


PGLow = PGLow./SBase;
PGUpp = PGUpp./SBase;
QGLow = QGLow./SBase;
QGUpp = QGUpp./SBase;
PL = PL./SBase;
QL = QL./SBase;

% PGLow(gens(:,1)-1) = gens(:,2)./SBase;
% PGUpp(gens(:,1)-1) = gens(:,3)./SBase;
% QGLow(gens(:,1)-1) = gens(:,4)./SBase;
% QGUpp(gens(:,1)-1) = gens(:,5)./SBase;
% PL(loads(:,1)-1) = loads(:,2)./SBase;
% QL(loads(:,1)-1) = loads(:,3)./SBase;


% clean up any nans or infs from the data
[~, c1] = find(~isfinite(PL));
[~, c2] = find(~isfinite(QL));
[~, c3] = find(~isfinite(PGLow));
[~, c4] = find(~isfinite(PGUpp));
[~, c5] = find(~isfinite(QGLow));
[~, c6] = find(~isfinite(QGUpp));
col_delete = [c1; c2; c3; c4; c5; c6];
col_delete = unique(col_delete);

% save case data
caseKeySet = {'PL','QL','PGLow','PGUpp','QGLow','QGUpp'};
caseValueSet_all = {PL(:, setdiff(1:total_cases, col_delete)),...
    QL(:, setdiff(1:total_cases, col_delete)), ...
    PGLow(:, setdiff(1:total_cases, col_delete)), ...
    PGUpp(:, setdiff(1:total_cases, col_delete)), ...
    QGLow(:, setdiff(1:total_cases, col_delete)), ...
    QGUpp(:, setdiff(1:total_cases, col_delete))};
case_data_all = cell2struct(caseValueSet_all', caseKeySet);
total_cases = total_cases - length(col_delete);

% save('casedata_partial.mat', 'network_4_data', 'case_data_all')

%% Solve reconfiguration problem on cases
resKeySet = {'z', 'zc', 'yalmiptime', 'solvertime', 'objVal'};
z = []; zc = []; yalmiptime = []; solvertime = []; objVal = [];
errored = [];

ind = 1;
caseValueSet = {PL(:, ind), QL(:, ind), PGLow(:, ind), PGUpp(:, ind), QGLow(:, ind), QGUpp(:, ind)};
case_data = cell2struct(caseValueSet', caseKeySet);

% get parameterized model for faster solving
model_paramzed = reconfigure_optimizer.get_paramzed_model(network_4_data);
res = reconfigure_optimizer.solve_case(model_paramzed, network_4_data, case_data);

for ind = total_cases+1:total_cases+length(col_delete) % 1:total_cases+length(col_delete)
    % each column is a new case
    if find(ind==col_delete)
        continue
    end
    
    caseValueSet = {PL(:, ind), QL(:, ind), PGLow(:, ind), PGUpp(:, ind), QGLow(:, ind), QGUpp(:, ind)};
    case_data = cell2struct(caseValueSet', caseKeySet);

    [z_case, zc_case, yaltime_case, solvertime_case, objval_case, error] = reconfigure(network_4_data, case_data);
    
    if error ~= 0 
        errored = [errored, ind]; % remove these test cases later
    else
        z = [z, z_case];
        zc = [zc, zc_case];
        yalmiptime = [yalmiptime, yaltime_case];
        solvertime = [solvertime, solvertime_case];
        objVal = [objVal, objval_case];
    end
    
    if rem(ind,200) == 0
        ind
    end
end

% each column is a new case
resValueSet_all = {z, zc, yalmiptime, solvertime, objVal};
res_data_all = cell2struct(resValueSet_all', resKeySet);

save('casedata_graph.mat', 'network_4_data', 'case_data_all', 'res_data_all')

%% Setup load and generation data
function mm_d_all = load_variations_uniform()
    % assume P and Q follow identical profiles
    % ISONE demand profiles from website
    
    pathname = strcat('./dataprofiles/ISONE_hourlyload/');
    files = dir(strcat(pathname,'*.csv'));
    mm_d_all = zeros(24,length(files));
    for k=1:length(files)
        filename = files(k).name;
        tempmat = readmatrix(strcat(pathname,filename)); % fnc for macOS
        mm_d_all(:,k) = tempmat(1:24,4)'; % total load is fourth column
    end
    mm_d_all = mm_d_all./(max(max(mm_d_all)));% normalize the load data  
    
%     figure; plot(mm_d_all)
end

function [mm_dP, mm_dQ] = load_variations_perturbed(N)
    % perturb the ISONE demand profiles using a Gaussian model
    % generate variations between nodes
    mm_d_all = load_variations_uniform();
    
    sigmaP = 0.1;
    rP = normrnd(0,sigmaP,[N*24,size(mm_d_all,2)]); % need N variations for each dataset
    mm_d_pert = repmat(mm_d_all,N,1).*(1+rP); % N*24 by number of datasets
    % smooth the data to give a more realistic demand profile
    mm_dP = smoothdata(smoothdata(smoothdata(reshape(mm_d_pert,24,[])',1,'gaussian'),2),'gaussian')'; % N by T
    mm_dP = reshape(mm_dP,N*24,[]); 
    
%     reshape(reshape(mm_dP, N*24, [])', )
    
    % generate some variability so P and Q don't follow identical profiles
    sigmaQ = 0.05;
    rQ = normrnd(0,sigmaQ,[N*24,size(mm_d_all,2)]); % need N by T variations
    mm_dQ_pert = mm_dP.*(1+rQ); % N by T
    mm_dQ = smoothdata(smoothdata(smoothdata(reshape(mm_dQ_pert,24,[])',1,'gaussian'),2),'gaussian')'; % N by T
    
	% adjust to have average load be the original load
    meanP = mean(mm_dP); meanQ = mean(mm_dQ);
    mm_dP = mm_dP-meanP; % ./maxP;
    mm_dQ = mm_dQ-meanQ; % ./maxQ;    

%     figure; plot(reshape(mm_dP,24,[]))
%     figure; plot(reshape(mm_dQ,24,[]))
    
    % each row is a node, each column is an hour organized by day1, day2...
	mm_dP = reshape(im2col(mm_dP, [24 size(mm_dP,2)],'distinct')', N, []);
    mm_dQ = reshape(im2col(mm_dQ, [24 size(mm_dQ,2)],'distinct')', N, []);
end

function [gen] = gen_variations_constant()
    gen = 1;
end

function [gen_P, gen_Q] = gen_variations_rengen(pf)
	DCtoAC = 1.2; % fixed DC to AC conversion ratio
	filename = 'PV_distcomm_pheonixAZ_185kW.csv';
	capacity = 185; % of the data, kW
    
    pathname = './dataprofiles/';
    tempmat = readmatrix(strcat(pathname,filename)); 
    gen_P = tempmat(:,2)/DCtoAC; % hourly generation for a year

    % want data as a mult factor for resources with other capacity
	gen_P = gen_P./capacity;
    gen_Q = gen_P.* tan(acos(pf));
    
%     figure; plot(gen_P)
%         mm_gQ = mm_gQT./capacity;
end



