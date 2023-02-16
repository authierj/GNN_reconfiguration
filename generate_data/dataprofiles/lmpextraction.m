%% Extract 5min LMP data from the ISONE data
clear 
clc

id = 'May142019';
filename = strcat(id,'/lmp_5min_20190514_');
% filename = 'May142019_5minLMP';
periods = ["00-04","04-08","08-12","12-16","16-20","20-24"];
units = 'USD/kWh'; % ISONE LMP data given in $/MWh, we convert to $/kWh
mm_p = zeros(1,24*60/5);

% retrieve different pricing data, generate different demand profiles
% locIDlist = [1,4,13];
locIDlist = [1];
mm_p_list = {}; mm_p_list{size(locIDlist,2)} = {};

% Extract 5min load profile from regional total load profile
filename_d = strcat(id,'/rt_fiveminsysload_20190514.csv');
tempmat = xlsread(filename_d);
mm_d_master = tempmat(:,1)'; % total load is first column
mm_d_master = mm_d_master./(max(mm_d_master));% normalize the load data

figure; hold on;
for n = 1:size(locIDlist,2)
    locIDpos = locIDlist(n);
    startind = 1;
    endind = 4*60/5; % 5 minute LMPS, broken into 4 hour files

    for i=1:size(periods,2)
        filenamefull = strcat(filename,periods(i),'.csv');
        tempmat = xlsread(filenamefull);
        locID = tempmat(locIDpos,1); % f1 = (1,1), f2 = (4,1), f3 = (13,1)
        inds = tempmat(:,1) == locID;

        tempsubmat = tempmat(inds,:);
        tempLMPs = tempsubmat(:,end);

        mm_p(startind:endind) = tempLMPs/1e3'; % convert to $/kWh
        startind = endind + 1;
        endind = endind + 4*60/5;
    end
    
    mm_p_list{n} = mm_p;
    
    % generate load profile by perturbing ISO-NE profile
%     r = normrnd(0,0.075,size(mm_d_master));
%     mm_d = mm_d_master.*(ones(size(mm_d_master))+r);
%     mm_d = smoothdata(smoothdata(mm_d)); % twice smoothing needed for variance of 0.1
%     mm_d = mm_d./(max(mm_d));% normalize the load data
    mm_d = mm_d_master./(max(mm_d_master));% normalize the load data
    
    % plot and check data
    subplot(2,1,1); hold on; plot(mm_d);
    subplot(2,1,2); hold on; plot(mm_p)
    
    label = strcat('_f',int2str(n));
    save(strcat(id,'/','isonedata','_',id),'locID','mm_p','mm_d','units'); %,label

end
