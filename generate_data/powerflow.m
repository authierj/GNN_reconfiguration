%% need to solve PF for default topology
% filename = 'casedata_33_rand';
filename = 'casedata_83_REDS_profile_fixedVmin';
load(strcat(filename, '.mat'))
total_cases = size(case_data_all.PL,2);
PL = case_data_all.PL;
QL = case_data_all.QL;
PGLow = case_data_all.PGLow;
PGUpp = case_data_all.PGUpp;
QGLow = case_data_all.QGLow;
QGUpp = case_data_all.QGUpp;

% set the topology we are interested in
% y_ij_fixed = [1 1 0 0 0 0 0]'; % default for 33
% y_ij_fixed = [0 0 0 0 1 0 1]'; % base for 33

y_ij_fixed = base; % base for 83_REDS (most common config)
% y_ij_fixed = configs(:,7); % second most common
% y_ij_fixed = configs(:,8); % third most common

% get parameterized model for faster solving
model_paramzed = powerflow_optimizer.get_paramzed_model(network_data, y_ij_fixed);


% setup result matrices, solve all cases
resKeySet = {'z', 'zc', 'solvetime', 'objVal'};
caseKeySet = {'PL','QL','PGLow','PGUpp','QGLow','QGUpp'};
z = []; zc = []; solvetime = []; objVal = [];
errored = [];
for ind = 1:total_cases
    % each column is a new case
    caseValueSet = {PL(:, ind), QL(:, ind), PGLow(:, ind), PGUpp(:, ind), QGLow(:, ind), QGUpp(:, ind)};
    case_data = cell2struct(caseValueSet', caseKeySet);

    [z_case, zc_case, solvetime_case, objval_case, err] = powerflow_optimizer.solve_case(model_paramzed, network_data, case_data, y_ij_fixed);
    
    if err ~= 0 
        errored = [errored, ind]; % flag as issue
    else
        z = [z, z_case];
        zc = [zc, zc_case];
        solvetime = [solvetime, solvetime_case];
        objVal = [objVal, objval_case];
    end
    
    if rem(ind,100) == 0
        ind
    end
end

% each column is a new case
resValueSet_all = {z, zc, solvetime, objVal};
res_data_all = cell2struct(resValueSet_all', resKeySet);

% save case data
save(strcat(filename, '_PFother8.mat'), 'network_data', 'case_data_all', 'res_data_all')
