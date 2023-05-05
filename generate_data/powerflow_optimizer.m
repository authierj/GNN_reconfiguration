

classdef powerflow_optimizer
    methods(Static)

        function model_paramzed = get_paramzed_model(networkdata, yij)
            % render a parameterized MILP using yalmip's optimizer function
            % vary PL and QL in different runs of the model
            % LinDistFlow power flow model
            % objective function is line losses
            
            bigM = 10; % big M formulation of conditional constraints
            
            % initialize yalmip variables
            V = sdpvar(networkdata.N,1); % squared voltage magnitude
%             PL = sdpvar(networkdata.N,1);
%             QL = sdpvar(networkdata.N,1);
            QL_feed = sdpvar(1);
            PG = sdpvar(networkdata.N,1);
            QG = sdpvar(networkdata.N,1);
            Pij = sdpvar(networkdata.M,1);
            Qij = sdpvar(networkdata.M,1);
            Pji = sdpvar(networkdata.M,1);
            Qji = sdpvar(networkdata.M,1);
            zij = binvar(networkdata.M,1);
            zji = binvar(networkdata.M,1);
        
            % parameterizations
            PL_param = sdpvar(networkdata.N-networkdata.numFeeders,1);
            QL_param = sdpvar(networkdata.N-networkdata.numFeeders,1);
            PG_upp_param = sdpvar(networkdata.N,1);
            PG_low_param = sdpvar(networkdata.N,1);
            QG_upp_param = sdpvar(networkdata.N,1);
            QG_low_param = sdpvar(networkdata.N,1);
%             yij_param = binvar(networkdata.numSwitches,1);
            
            nosw_inds = setdiff(1:networkdata.M,networkdata.swInds);
            powerflow = [
                networkdata.A(nosw_inds, :)*V == -2*(diag(networkdata.R(nosw_inds))*(Pij(nosw_inds)-Pji(nosw_inds)) + diag(networkdata.X(nosw_inds))*(Qij(nosw_inds)-Qji(nosw_inds))),... % voltage rln
                networkdata.A(networkdata.swInds, :)*V <= -2*(diag(networkdata.R(networkdata.swInds))*(Pij(networkdata.swInds)-Pji(networkdata.swInds)) + diag(networkdata.X(networkdata.swInds))*(Qij(networkdata.swInds)-Qji(networkdata.swInds))) + bigM*(1-zij(networkdata.swInds)-zji(networkdata.swInds)),... % voltage rln
                networkdata.A(networkdata.swInds, :)*V >= -2*(diag(networkdata.R(networkdata.swInds))*(Pij(networkdata.swInds)-Pji(networkdata.swInds)) + diag(networkdata.X(networkdata.swInds))*(Qij(networkdata.swInds)-Qji(networkdata.swInds))) - bigM*(1-zij(networkdata.swInds)-zji(networkdata.swInds)),... % voltage rln
                PG(1:networkdata.numFeeders) == networkdata.Aflow(1:networkdata.numFeeders,:)*(Pij - Pji),... % P balance at feeder, feeder cannot export so PL = 0
                PG(networkdata.numFeeders+1:end) - PL_param == networkdata.Aflow(networkdata.numFeeders+1:end,:)*(Pij - Pji),... % power balance, P
                QG(1:networkdata.numFeeders) == networkdata.Aflow(1:networkdata.numFeeders,:)*(Qij - Qji),... % Q balance at feeder, feeder cannot export so QL = 0
                QG(networkdata.numFeeders+1:end) - QL_param == networkdata.Aflow(networkdata.numFeeders+1:end,:)*(Qij - Qji)% power balance, Q
                ];
            
%             powerflow = [
%                 networkdata.A(nosw_inds, :)*V == -2*(diag(networkdata.R(nosw_inds))*(Pij(nosw_inds)-Pji(nosw_inds)) + diag(networkdata.X(nosw_inds))*(Qij(nosw_inds)-Qji(nosw_inds))),... % voltage rln
%                 networkdata.A(networkdata.swInds, :)*V <= -2*(diag(networkdata.R(networkdata.swInds))*(Pij(networkdata.swInds)-Pji(networkdata.swInds)) + diag(networkdata.X(networkdata.swInds))*(Qij(networkdata.swInds)-Qji(networkdata.swInds))) + bigM*(1-yij),... % voltage rln
%                 networkdata.A(networkdata.swInds, :)*V >= -2*(diag(networkdata.R(networkdata.swInds))*(Pij(networkdata.swInds)-Pji(networkdata.swInds)) + diag(networkdata.X(networkdata.swInds))*(Qij(networkdata.swInds)-Qji(networkdata.swInds))) - bigM*(1-yij),... % voltage rln
%                 PG(1) == networkdata.Aflow(1,:)*(Pij - Pji),... % P balance at feeder, feeder cannot export so PL = 0
%                 PG(2:end) - PL_param == networkdata.Aflow(2:end,:)*(Pij - Pji),... % power balance, P
%                 QG(1) - QL_feed == networkdata.Aflow(1,:)*(Qij - Qji),... % Q balance at feeder, feeder can have negative Q
%                 QG(2:end) - QL_param == networkdata.Aflow(2:end,:)*(Qij - Qji) % power balance, Q
%                 ];
            
            varLims = [
                [PG_low_param] <= PG,...
                PG_upp_param(networkdata.numFeeders+1:end) >= PG(networkdata.numFeeders+1:end),... % feeder PG is unbounded
                [QG_low_param] <= QG,...
                QG_upp_param(networkdata.numFeeders+1:end) >= QG(networkdata.numFeeders+1:end),... % feeder QG is unbounded
                [1; ones(networkdata.N-1,1)*networkdata.vLow^2] <= V,... % includes slack voltage
                [1; ones(networkdata.N-1,1)*networkdata.vUpp^2] >= V % includes slack voltage
                ];

%             varLims = [
%                 [0; PG_low_param] <= PG,...
%                 PG_upp_param >= PG(2:end),... % feeder PG is unbounded
%                 [0; QG_low_param] <= QG,...
%                 QG_upp_param >= QG(2:end),... % feeder QG is unbounded
%                 [1; ones(networkdata.N-1,1)*networkdata.vLow^2] <= V,... % includes slack voltage
%                 [1; ones(networkdata.N-1,1)*networkdata.vUpp^2] >= V % includes slack voltage
%                 ];
            
            load = [
                QL_feed >= 0  % need positivity of QL and QG for all nodes
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
                ];
            
            objfnc = sum(diag(networkdata.R)*(Pij.^2 + Pji.^2 + Qij.^2 + Qji.^2)); % line losses
%             objfnc_voltage = sum((V-1).^2); % flat voltage profile
            % sum((y(VrInds)-ones(3*N*T,1)).^2 + (y(ViInds)-zeros(3*N*T,1)).^2)
            
            ops = sdpsettings('solver','Gurobi','verbose',0); % 0
            model_paramzed = optimizer([powerflow, varLims, load,topological, radiality], objfnc, ops, [PL_param; QL_param; PG_upp_param; PG_low_param; QG_upp_param; QG_low_param], [V;QL_feed;PG;QG;Pij;Qij;Pji;Qji;zij;zji]); % PL;QL; after V
            %  yij_param
        
        
        end % end function


        function [z, zc, solvetime, objval, err] = solve_case(model_paramzed, networkdata, casedata, yij) % res

            time_start = tic;
            [res, err] = model_paramzed([casedata.PL(networkdata.numFeeders+1:end); casedata.QL(networkdata.numFeeders+1:end); casedata.PGUpp; casedata.PGLow; casedata.QGUpp; casedata.QGLow]);
            time_end = toc(time_start);

            solvetime = time_end - time_start; 

            if err ~= 0
        %         error('ERROR: Cannot solve reconfiguration. \nYalmip returned %d: %s',err,yalmiperror(err))
                fprintf('\n ERROR: Cannot solve reconfiguration. \nYalmip returned %d: %s \n',err,yalmiperror(err))
                z = NaN(4*networkdata.M+2*networkdata.numSwitches+networkdata.N, 1);
                zc = NaN(2*networkdata.M-networkdata.numSwitches+2*networkdata.N+1, 1);
            else
                % decompose result vector into power variables
                ind = 1;
                V = res(ind:networkdata.N); ind = ind + networkdata.N; % squared voltage magnitude
%                 PL = res(ind:ind+networkdata.N-1); ind = ind + networkdata.N;
%                 QL = res(ind:ind+networkdata.N-1); ind = ind + networkdata.N;
                PL = casedata.PL;
                QL_feed = res(ind); ind = ind + 1;
                QL = casedata.QL;
                PG = res(ind:ind+networkdata.N-1); ind = ind + networkdata.N;
                QG = res(ind:ind+networkdata.N-1); ind = ind + networkdata.N;
                Pij = res(ind:ind+networkdata.M-1); ind = ind + networkdata.M;
                Qij = res(ind:ind+networkdata.M-1); ind = ind + networkdata.M;
                Pji = res(ind:ind+networkdata.M-1); ind = ind + networkdata.M;
                Qji = res(ind:ind+networkdata.M-1); ind = ind + networkdata.M;
                zij = res(ind:ind+networkdata.M-1); ind = ind + networkdata.M;
                zji = res(ind:ind+networkdata.M-1); ind = ind + networkdata.M;

                % calculate the objective functiion value
                objval = sum(diag(networkdata.R)*(Pij.^2 + Pji.^2 + Qij.^2 + Qji.^2)); % line losses

                % decompose res into z and zc: form for the ML test/validation
                netG_P = PG - PL;
                netG_Q = QG - QL;
        
                if netG_P(1) < 0
                    plf = netG_P(1); 
                    PG(1) = 0;
                else
                    plf = 0;
                    PG(1) = netG_P(1);
                end
                
                if netG_Q(1) < 0
                    qlf = netG_Q(1); 
                    QG(1) = 0;
                else
                    qlf = 0;
                    QG(1) = netG_Q(1);
                end
                
                % x = [PL, QL]
                % z = [zji, y\last, pij, pji, qji, {qij}_sw, v\f, plf, qlf]
                % zc = [zij, ylast, {qij}_nosw, pg, qg]
                z = [zji; yij(1:end-1); Pij; Pji; Qji; 
                    Qij(networkdata.M-networkdata.numSwitches+1:networkdata.M);
                    V(2:end); plf; qlf];
                zc = [zij; yij(end); Qij(1:networkdata.M-networkdata.numSwitches); PG; QG];
            end

        end % end function


    end % end methods
end % end classdef
