%% Extract load and solar data from Hawaii's database
clear 
clc

id = 'Aug222020'; 
beginDateExtract = datetime(2020,08,22,00,00,00);
endDateExtract = datetime(2020,08,23,00,00,00); % end date + 1

filenameSolar = strcat(id,'_solar');
filenameLoad = strcat(id,'_load');


% extract the load data and average to get demand profile
% column: "P"
filename_l = strcat(filenameLoad,'.csv');
tempmat_l = readtable(filename_l);
TTload = table2timetable( tempmat_l, 'rowTimes','Time_HT_' ); % convert to timetable for easy data manipulation
TTload_day = retime(TTload,'daily','mean'); % average the data over each day
TTload_hr = retime(TTload,'hourly','mean'); % average the data over each hour of each day
TTload_min = retime(TTload,'minutely','mean'); % average the data over each minute of each day

all_d = TTload_min(timerange(beginDateExtract,endDateExtract),:).P;
mm_d = all_d./max(all_d); % normalize
mm_d(mm_d < 0) = 0; % deal with any negative values (make them = 0)

% extract the solar data and average to get generation profile
% column: "P"
filename_s = strcat(filenameSolar,'.csv');
tempmat_s = readtable(filename_s);
TTsolar = table2timetable( tempmat_s, 'rowTimes','Time_HT_' ); % convert to timetable for easy data manipulation
TTsolar_day = retime(TTsolar,'daily','mean'); % average the data over each day
TTsolar_hr = retime(TTsolar,'hourly','mean'); % average the data over each hour of each day
TTsolar_min = retime(TTsolar,'minutely','mean'); % average the data over each minute of each day

all_gen = TTsolar_min(timerange(beginDateExtract,endDateExtract),:).P;
mm_genSolar = all_gen./8000; % normalize against capacity of 8kW
mm_genSolar(mm_genSolar < 0) = 0; % deal with any negative values (make them = 0)

units = 'W'; % load and gen data is in watts

% mm_d is minutely profile
save(strcat('hawaiidata','_',id),'mm_d','mm_genSolar','TTload_day','TTload_hr','TTload_min','TTsolar_day','TTsolar_hr','TTsolar_min','units');
