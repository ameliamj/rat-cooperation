warning off
clear all
%% ANALYZE CHEATING TRAINING
% Set thresholds

magEntryThresh = 0;
magExitThresh = .01;
%% Load data files / Get folder directories

folderName = 'D:\Dropbox (ChangLab)\RodentProjects\Training_COOPERATION';
a = string(questdlg('Load existing data file?', ...
    'Yes', 'No'));
if a == "Yes"
    [fileName, folderName] = uigetfile(folderName, 'Select training mat file');
    load([folderName filesep fileName]);
else
    allSessions = table();
end

a = "No";
while a == "No"
    a = string(questdlg('Select main data folder?', ...
        'Yes', 'No'));
    if a == "Yes"
        mainFolderName = uigetdir(folderName, 'Select main training data folder');
    else
        b = msgbox('Well, that is too bad, you have to select it','Wrong answer',"warn");
        pause(5)
        delete(b)
    end
end
%% Load data files / Get folder directories

sessionNames = struct2table(dir(mainFolderName));
sessionNames(contains(sessionNames.name,'.'),:) = [];
%% Rename
% for s = 1:height(sessionNames) sessionFolderName = [mainFolderName filesep 
% sessionNames.name{s}];
%%
% 
%     dataFileList = dir(sessionFolderName);
%     dataFileList = struct2table(dataFileList);
%
%%
% 
%     badFiles = dataFileList.bytes == 0 | dataFileList.isdir == 1;
%     dataFileList(badFiles,:) = [];
%
%%
% 
%     % Loop through each file
%     for i = 1:height(dataFileList)
%         fileName = [dataFileList.folder{i} filesep dataFileList.name{i}];
%         if contains(fileName, 'loaded')
%             movefile(fileName, [fileName(1:end-7)])
%         end
%     end
%  end
%
%% 
% allSessions = table();
%% Load raw data files

for s = 1:height(sessionNames)
    sessionFolderName = [mainFolderName filesep sessionNames.name{s}];
    
    dataFileList = dir(sessionFolderName);
    dataFileList = struct2table(dataFileList);
    
    badFiles = dataFileList.bytes == 0 | dataFileList.isdir | contains(dataFileList.name, ".mp4") == 1;
    dataFileList(badFiles,:) = [];
    n = height(allSessions);
    
    % Loop through each file
    for i = 1:height(dataFileList)
        fileName = [dataFileList.folder{i} filesep dataFileList.name{i}];
        %if contains(fileName, 'loaded')
        %     disp(['File ' num2str(i) ' of ' num2str(height(dataFileList)) ...
        %         ' in session ' num2str(s) ' of ' num2str(height(sessionNames)) ' already loaded...'])
        % else
        if true
            % Load, read, analyze
            disp(['Loading file ' num2str(i) ' of ' num2str(height(dataFileList)) ...
                ' in session ' num2str(s) ' of ' num2str(height(sessionNames))])
            j = i + n;
            allSessions.SessionNum(j) = j; trialData = medpcParseStruct(fileName); % Read data file
            allSessions = analyzeCoopTraining(trialData,allSessions,j,magEntryThresh,magExitThresh);
            disp(['Successfully loaded file ' num2str(j)])
            
            movefile(fileName, [fileName(1:end) '_loaded'])
            
            if contains(sessionFolderName, 'Opaque')
                allSessions.Cond(j) = "Paired-Opaque";
            elseif contains(sessionFolderName, 'Translucent')
                allSessions.Cond(j) = "Paired-Translucent";
            elseif contains(sessionFolderName, 'Transparent')
                allSessions.Cond(j) = "Paired-Transparent";
            elseif contains(dataFileList.name{i}, 'TP')
                allSessions.Cond(j) = "CNO-TrainingPartner";
            elseif contains(dataFileList.name{i}, 'UF')
                allSessions.Cond(j) = "CNO-Unfamiliar";
            end
        end
    end
end
%% Clean up data table

allSessions.GroupNum(ismissing(allSessions.GroupNum)) = extractBefore(allSessions.AnimalID(ismissing(allSessions.GroupNum)),3);
allSessions.GroupNum(contains(allSessions.GroupNum, "EB")) = "EB";
allSessions.GroupNum(contains(allSessions.GroupNum, "HF")) = "HF";
allSessions.GroupNum(contains(allSessions.GroupNum, "NF")) = "NF";
allSessions.GroupNum(contains(allSessions.GroupNum, "NM")) = "NM";
allSessions.GroupNum(contains(allSessions.GroupNum, "KM")) = "KM";

ebIDs = ["EB001" "EB002" "EB003" "EB004" "EB005" "EB006" "EB007" "EB008" "EB009",...
    "EB010"	"EB011"	"EB012"	"EB015"	"EB016"	"EB017"	"EB018"	"EB019"	"EB020"	"EB021",...
    "EB022"	"EB023"	"EB024"	"EB027"	"EB028"	"EB029"	"EB030"	"EB031"	"EB032"	"EB033"	"EB034"];
ebColors = ["R" "G" "B" "G" "Y" "G" "R" "G" "B" "G" "Y" "G",...
    "R" "G" "B" "G" "Y" "G" "R" "G" "B" "G" "Y" "G" "R" "G" "B" "G" "Y" "G"];

[~,idx] = ismember(allSessions.AnimalID, ebIDs);
loc = find(idx>0); idx = idx(loc);
allSessions.AnimalID(loc) = strcat(ebIDs(idx)', ebColors(idx)');

% allSessions.Cond(strlength(allSessions.AnimalID)<8 & allSessions.Cond == "") = "Yoked-Single";
% allSessions.Cond(strlength(allSessions.AnimalID)>8 & allSessions.Cond == "") = "Yoked-Paired";
%% SAVE

% save([mainFolderName filesep 'AllTrainingData.mat'],'allSessions')
save([mainFolderName filesep 'AllTrainingDataII.mat'],'allSessionsOrig')

disp('Saved training mat file!')
%%
allSessionsOrig = allSessions;
allSessionsOrig.GroupNum(strlength(allSessionsOrig.GroupNum)>5) = extractBefore(allSessionsOrig.GroupNum(strlength(allSessionsOrig.GroupNum)>5),6);

% depVars = ["HitLat", "firstLevPressLat","coopSuccTrialLat", "coopSuccPressLat",...
%     "nMiss", "nFA", "nRevisit","nRePressBeforeCoop","nRePressAfterCoop"];
% for k = 1:length(depVars)
%     dum = cell2mat(allSessionsOrig.(depVars(k)));
%     allSessionsOrig.(depVars(k)) = dum(:,3);
% end
% allSessionsOrig.CoopSuccMeanAll(isnan(allSessionsOrig.CoopSuccMeanAll)) = 0;
%% 
% allSessionsOrig = sortrows(allSessionsOrig,"Cond","ascend"); allSessionsOrig(allSessionsOrig.CoopSuccMeanAll 
% == 0,:) = []; allSessionsOrig(allSessionsOrig.nFA>5,:) = []; allSessionsOrig(allSessionsOrig.firstLevPressLat>2,:) 
% = []; allSessionsOrig(allSessionsOrig.coopSuccTrialLat>2,:) = [];

% depVars = ["CoopSuccMeanAll", "HitLat", "firstLevPressLat","coopSuccTrialLat", "coopSuccPressLat",...
%     "nMiss", "nFA", "nRevisit","nRePressBeforeCoop","nRePressAfterCoop"];
%
% figure('Renderer', 'painters', 'Position', [200 200 1000 1000])
% t = tiledlayout(4,3,'TileSpacing','Compact','Padding','Compact');
% for n = 1:length(depVars)
%     nexttile(); hold on
%     scatter(categorical(allSessionsOrig.Cond), allSessionsOrig.(depVars(n)),'filled','MarkerFaceAlpha',.2)
%     % [~,tbl, stats] = anova1(allSessionsOrig.(depVars(n)), allSessionsOrig.Cond,'');
%     [statsMean, statsSEM, statsVar] = grpstats(allSessionsOrig.(depVars(n)), allSessionsOrig.Cond,{'mean','sem','var'});
%     errorbar(statsMean, statsSEM,"LineStyle","none","MarkerSize",10)
%     ylabel(depVars(n))
% end
%% Plot by animal and group

opts = struct(); opts.error = 'c95';
opts.color_area = [0.1490    0.1490    0.1490];
opts.color_line = [0.1490    0.1490    0.1490];
opts.alpha      = 0.1;
opts.line_width = 1;

allSessionsOrig(ismissing(allSessionsOrig.LearningType),:) = [];
% allSessionsOrigOrig(cellfun(@height,allSessionsOrigOrig.allTrials)<1,:) = [];
trainingTypes = unique(allSessionsOrig.LearningType);

plotGroupIDs = {["KL001", "KL002"],...
    ["KL003", "KL004", "KL005", "KL006", "KL007", "KL008"],...
    ["KL001", "KL002", "KL003", "KL004", "KL005", "KL006", "KL007", "KL008"],...
    ["EB"],...
    ["HF"], "NM", "NF", "KM"};
plotGroupList = {"KL001-002", "KL003-008", "KL001-008", "EB", "HF", "NM", "NF", "KM"};
[idx,~] = listdlg('ListString',plotGroupList, 'PromptString', 'Choose which set of groups you want to plot');
plotGroupIDs = plotGroupIDs(idx);

[idx,~] = listdlg('ListString',trainingTypes, 'PromptString', 'Choose which training type you want to plot');
trainingTypes = trainingTypes(idx);
%%
for ii = 1:length(plotGroupIDs)
    for tt = 1:length(trainingTypes)
        allSessions = allSessionsOrig(allSessionsOrig.LearningType == trainingTypes(tt) ...
            & ismember(allSessionsOrig.GroupNum, plotGroupIDs{ii}),:);
        allSessions(ismissing(allSessions.AnimalID),:) = [];
        
        if height(allSessions)>0
            nTrialsPer = findElementRep(allSessions.AnimalID);
            
            uniqueAnimals = sort(nTrialsPer(:,1));
            animalColors = lower(eraseBetween(uniqueAnimals,1,5));
            nTrialsPer = str2double(nTrialsPer(:,2));
            uniqueGroups = unique(allSessions.GroupNum);
            
            markers = ["-x","-o","-+","-^","--x","--o","--+","--^",":x",":o",":+",":^","-.x","-.o","-.+","-.^"];
            cmapIdx = ["r", "g", "b", "y"];
            cmap = [0.6350 0.0780 0.1840;    ...
                0.4660 0.6740 0.1880;    ...
                0 0.4470 0.7410;         ...
                0.9290 0.6940 0.1250];
            figxlim = [0 max(nTrialsPer)+1];
            
            conds = ["Mag A", "Mag B", "Both Mags"];
            % depVars = ["HitLat", "TrialDur", "nMiss", "nFA", "nRevisit", "SwitchLat", "leverLat"];
            depVars = ["HitLat", "nMiss", "nFA", "nRevisit", "CoopSuccMeanAll","CoopSuccMeanInfl","firstLevPressLat",...
                "coopSuccTrialLat","coopSuccPressLat", "nRePressBeforeCoop", "nRePressAfterCoop"];
            figyLabs = ["Latency to mag entry (s)", "Trial Duration (s)", "Number of misses per trial",...
                "Number of false alarms per trial", "Number of revisits per trial",...
                "Latency to mag 2 from mag 1 (s)", "Latency to lever press (s)"];
            for i = 1:length(depVars)
                data = repmat({nan(length(uniqueAnimals),max(nTrialsPer))},3,1);
                
                for n = 1:length(uniqueAnimals)
                    sessionDum = allSessions(allSessions.AnimalID == uniqueAnimals(n),:);
                    sessionDum = sortrows(sessionDum, "Date");
                    
                    if ~ismember(depVars(i), ["SwitchLat","CoopSuccMeanAll","CoopSuccMeanInfl"])
                        dataDum = cell2mat(sessionDum.(depVars(i)));
                        for j = 1:3
                            data{j}(n,1:size(dataDum,1)) = dataDum(:,j);
                        end
                    else
                        dataDum = sessionDum.(depVars(i));
                        data{3}(n,1:size(dataDum,1)) = dataDum;
                    end
                end
                
                figure('Renderer', 'painters', 'Position', [200 200 1000 400])
                t = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
                s = nan(length(uniqueAnimals),1);
                if ~ismember(depVars(i), ["SwitchLat","CoopSuccMeanAll","CoopSuccMeanInfl"])
                    ax = cell(3,1);
                    for n = 1:3
                        ax{n} = nexttile();
                        hold on
                        for a = 1:length(uniqueAnimals)
                            dum = char(uniqueAnimals(a));
                            if contains(dum,'EB')
                                dum = dum(1:2);
                            else
                                dum = dum(1:end-1);
                            end
                            s(a) = plot(data{n}(a,:), 'LineWidth',1);
                        end
                        plot_areaerrorbar(movmean(data{n},2,"omitnan"), opts);
                        %                 plot(nanmean(data{n}), 'LineWidth',2, 'Color',[0 0 0 .3])
                        title(conds(n))
                        xlim(figxlim)
                        %                 ylim([0 150])
                        xlabel('Session number')
                        % ylabel(figyLabs(i))
                    end
                    linkaxes([ax{1} ax{2} ax{3}])
                else
                    nexttile(3)
                    hold on
                    for a = 1:length(uniqueAnimals)
                        dum = char(uniqueAnimals(a));
                        if contains(dum,'EB')
                            dum = dum(1:2);
                        else
                            dum = dum(1:end-1);
                        end
                        s(a) = plot(data{3}(a,:), 'LineWidth',1);
                    end
                    plot_areaerrorbar(movmean(data{3},2,"omitnan"), opts);
                    %             plot(nanmean(data{3}), 'LineWidth',2, 'Color',[0 0 0 .3])
                    title(conds(3))
                    xlim(figxlim)
                    xlabel('Session number')
                    % ylabel(figyLabs(i))
                end
                lg = legend(s,uniqueAnimals);
                lg.Layout.Tile = 'East';
                xlabel(t, 'SessionNum')
                title(t, strcat(depVars(i), "-", trainingTypes(tt)))
            end
            
            figure('Renderer', 'painters', 'Position', [100 100 1400 800])
            t = tiledlayout(ceil(length(uniqueAnimals)/4),4,'TileSpacing','Compact','Padding','Compact');
            for n = 1:length(uniqueAnimals)
                sessionDum = allSessions(allSessions.AnimalID == uniqueAnimals(n),:);
                sessionDum = sortrows(sessionDum, "Date");
                data = nan(height(sessionDum),100);
                cmap = brewermap(height(sessionDum)+1,'BuPu'); cmap(1,:) = [];
                
                s = nan(height(sessionDum),1);
                ax = nexttile;
                hold on
                for i = 1:height(sessionDum)
                    % [f,~,mu] = polyfit(sessionDum.allTrials{i}.TrialNum,sessionDum.allTrials{i}.HitLat,3);
                    % data(i,:) = polyval(f,1:60,[],mu);
                    data(i,1:length(sessionDum.allTrials{i}.coopSuccTrialLat)) = movmean(sessionDum.allTrials{i}.coopSuccTrialLat,10,"omitnan");
                    s(i) = plot(data(i,:), 'Color',cmap(i,:), 'LineWidth',1);
                end
                title(uniqueAnimals(n))
                lg = legend(ax,s,string(sessionDum.Date), 'Location', 'bestoutside','FontSize',6);
            end
            % lg.Layout.Tile = 'East';
            xlabel(t, 'trial Num')
            ylabel(t, 'Expected latency to reach mag (s)')
            title(t, strcat("Latency to cooperate over time across sessions (smoothed over 10 trials) - ", trainingTypes(tt)))
            
            figure('Renderer', 'painters', 'Position', [100 100 1400 800])
            t = tiledlayout(ceil(length(uniqueAnimals)/4),4,'TileSpacing','Compact','Padding','Compact');
            for n = 1:length(uniqueAnimals)
                sessionDum = allSessions(allSessions.AnimalID == uniqueAnimals(n),:);
                sessionDum = sortrows(sessionDum, "Date");
                data = nan(height(sessionDum),100);
                cmap = brewermap(height(sessionDum)+1,'BuPu'); cmap(1,:) = [];
                
                s = nan(height(sessionDum),1);
                ax = nexttile;
                hold on
                for i = 1:height(sessionDum)
                    % [f,~,mu] = polyfit(sessionDum.allTrials{i}.TrialNum,sessionDum.allTrials{i}.HitLat,3);
                    % data(i,:) = polyval(f,1:60,[],mu);
                    data(i,1:length(sessionDum.allTrials{i}.coopSuccPressLat)) = movmean(sessionDum.allTrials{i}.coopSuccPressLat,10, "omitnan");
                    s(i) = plot(data(i,:), 'Color',cmap(i,:), 'LineWidth',1);
                end
                title(uniqueAnimals(n))
                lg = legend(ax,s,join([string(sessionDum.Date), repmat("Time Limit ", height(sessionDum),1), num2str(sessionDum.CoopTimeLimit)]),...
                    'Location', 'bestoutside','FontSize',6);
            end
            % lg.Layout.Tile = 'East';
            xlabel(t, 'trial Num')
            ylabel(t, 'Expected latency to reach mag (s)')
            title(t, strcat("Time to coordinate between the two animals (smoothed over 10 trials) - ", trainingTypes(tt)))
            
            figure('Renderer', 'painters', 'Position', [100 100 1400 800])
            t = tiledlayout(ceil(length(uniqueAnimals)/4),4,'TileSpacing','Compact','Padding','Compact');
            for n = 1:length(uniqueAnimals)
                sessionDum = allSessions(allSessions.AnimalID == uniqueAnimals(n),:);
                sessionDum = sortrows(sessionDum, "Date");
                data = nan(height(sessionDum),100);
                cmap = brewermap(height(sessionDum)+1,'BuPu'); cmap(1,:) = [];
                
                s = nan(height(sessionDum),1);
                ax = nexttile;
                hold on
                for i = 1:height(sessionDum)
                    % [f,~,mu] = polyfit(sessionDum.allTrials{i}.TrialNum,sessionDum.allTrials{i}.HitLat,3);
                    % data(i,:) = polyval(f,1:60,[],mu);
                    data(i,1:length(sessionDum.allTrials{i}.coopSucc)) = movmean(sessionDum.allTrials{i}.coopSucc,10, "omitnan");
                    s(i) = plot(data(i,:), 'Color',cmap(i,:), 'LineWidth',1);
                end
                title(uniqueAnimals(n))
                lg = legend(ax,s,join([string(sessionDum.Date), repmat("Time Limit ", height(sessionDum),1), num2str(sessionDum.CoopTimeLimit)]),...
                    'Location', 'bestoutside','FontSize',6);
            end
            % lg.Layout.Tile = 'East';
            xlabel(t, 'trial Num')
            ylabel(t, 'Expected latency to reach mag (s)')
            title(t, strcat("Probability to cooperate (smoothed over 10 trials) - ", trainingTypes(tt)))
            
        end
    end
end
%%
allSessionsOrig = load('AllTrainingDataII.mat')
%%
for ii = 1:height(allSessionsOrig)
    trial = allSessionsOrig.allTrials{ii, 1}
    if height(trial) > 0
        leverData = trial.allLevPress{1, 1}
        magData = trial.allmagEntries{1, 1}
        for jj = 2:height(trial)
            leverData = vertcat(leverData, trial.allLevPress{jj, 1})
            magData = vertcat(magData, trial.allmagEntries{jj, 1})
        end
        date_str = allSessionsOrig.Date(ii)
        date_obj = datetime(date_str, 'InputFormat', 'dd-MMM-yyyy');
        date = datestr(date_obj, 'mmddyy')
    
        animal = allSessionsOrig.AnimalID(ii)
        if istable(leverData)
            writetable(leverData, append('behav/lever/', date, '_', animal, '_lever.csv'),'Delimiter',',','QuoteStrings','all')
        end
        if istable(magData)
            writetable(magData, append('behav/mag/', date, '_', animal, '_mag.csv'),'Delimiter',',','QuoteStrings','all')
        end
    end
end  
%%
% 353
for ii = 1:height(allSessions)
    trial = allSessions.allTrials{ii, 1}
    if height(trial) > 0
        leverData = trial.allLevPress{1, 1}
        magData = trial.allmagEntries{1, 1}
        for jj = 2:height(trial)
            leverData = vertcat(leverData, trial.allLevPress{jj, 1})
            magData = vertcat(magData, trial.allmagEntries{jj, 1})
        end
        
        % Add CoopTimeLimit column to leverData
        leverData.CoopTimeLimit = allSessions.CoopTimeLimit(ii);

        date_str = allSessions.Date(ii)
        date_obj = datetime(date_str, 'InputFormat', 'dd-MMM-yyyy');
        date = datestr(date_obj, 'mmddyy')
    
        animal = allSessions.AnimalID(ii)
        if istable(leverData)
            writetable(leverData, append('behav/lever/', date, '_', animal, '_lever.csv'),'Delimiter',',','QuoteStrings','all')
        end
        if istable(magData)
            writetable(magData, append('behav/mag/', date, '_', animal, '_mag.csv'),'Delimiter',',','QuoteStrings','all')
        end
    end
end    

%% 
% Plot all data opts = struct(); opts.error = 'c95'; opts.color_area = [0.1490 
% 0.1490 0.1490]; opts.color_line = [0.1490 0.1490 0.1490]; opts.alpha = 0.2; 
% opts.line_width = 1;
% 
% allSessionsOrig(ismissing(allSessionsOrig.LearningType),:) = []; % allSessionsOrig(cellfun(@height,allSessionsOrig.allTrials)<10,:) 
% = []; trainingTypes = unique(allSessionsOrig.LearningType);
% 
% plotGroupIDs = {["KL001", "KL002", "KL003", "KL004", "KL005", "KL006", "KL007", 
% "KL008"],... ["EB"],... ["HF"], "NM", "NF"}; plotGroupList = {"KL", "EB","HF","NM","NF"}; 
% [idx,~] = listdlg('ListString',plotGroupList, 'PromptString', 'Choose which 
% set of groups you want to plot'); plotGroupIDs = plotGroupIDs(idx);
% 
% [idx,~] = listdlg('ListString',trainingTypes, 'PromptString', 'Choose which 
% training type you want to plot'); trainingTypes = trainingTypes(idx);
% 
% for ii = 1:length(plotGroupIDs) for tt = 1:length(trainingTypes) allSessions 
% = allSessionsOrig(allSessionsOrig.LearningType == trainingTypes(tt) ... & ismember(allSessionsOrig.GroupNum, 
% plotGroupIDs{ii}),:); allSessions(ismissing(allSessions.AnimalID),:) = [];
%%
% 
%         % if trainingTypes(tt) == "IS"
%         %     c = cmap(2,:);
%         % else
%         %     c = cmap(1,:);
%         % end
%
%%
% 
%         if height(allSessions)>0
%             nTrialsPer = findElementRep(allSessions.AnimalID);
%
%%
% 
%             uniqueAnimals = sort(nTrialsPer(:,1));
%             animalColors = lower(eraseBetween(uniqueAnimals,1,5));
%             nTrialsPer = str2double(nTrialsPer(:,2));
%             uniqueGroups = unique(allSessions.GroupNum);
%
%%
% 
%             cmap = [0 0.4470 0.7410; 0.9290 0.6940 0.1250];
%             figxlim = [0 8];
%
%%
% 
%             conds = ["Mag A", "Mag B", "Both Mags"];
%             depVars = ["HitLat", "TrialDur", "nMiss", "nFA", "nRevisit", "SwitchLat", "leverLat"];
%             figyLabs = ["Latency to mag entry (s)", "Trial Duration (s)", "Number of misses per trial",...
%                 "Number of false alarms per trial", "Number of revisits per trial",...
%                 "Latency to mag 2 from mag 1 (s)", "Latency to lever press (s)"];
%             for i = 1:length(depVars)
%                 data = repmat({nan(length(uniqueAnimals),max(nTrialsPer))},3,1);
%
%%
% 
%                 for n = 1:length(uniqueAnimals)
%                     sessionDum = allSessions(allSessions.AnimalID == uniqueAnimals(n),:);
%                     sessionDum = sortrows(sessionDum, "Date");
%
%%
% 
%                     if depVars(i) ~= "SwitchLat"
%                         dataDum = cell2mat(sessionDum.(depVars(i)));
%                         for j = 1:3
%                             data{j}(n,1:size(dataDum,1)) = dataDum(:,j);
%                         end
%                     else
%                         dataDum = sessionDum.(depVars(i));
%                         data{3}(n,1:size(dataDum,1)) = dataDum;
%                     end
%                 end
%
%%
% 
%                 figure('Renderer', 'painters', 'Position', [200 200 1000 400])
%                 t = tiledlayout(1,3,'TileSpacing','Compact','Padding','Compact');
%                 s = nan(length(uniqueAnimals),1);
%                 if depVars(i) ~= "SwitchLat"
%                     ax = cell(3,1);
%                     for n = 1:3
%                         ax{n} = nexttile();
%                         hold on
%                         for a = 1:length(uniqueAnimals)
%                             dum = char(uniqueAnimals(a));
%                             if contains(dum,'EB')
%                                 dum = dum(1:2);
%                             else
%                                 dum = dum(1:end-1);
%                             end
%                             s(a) = plot(data{n}(a,:), '-', 'Color', [c .3], 'LineWidth',.5);
%                         end
%                         plot_areaerrorbar(movmean(data{n},2), opts);
%                         % plot(nanmean(data{n}), 'LineWidth',2, 'Color',[c .8])
%                         title(conds(n))
%                         xlim(figxlim)
%                         %                 ylim([0 150])
%                         xlabel('Session number')
%                         ylabel(figyLabs(i))
%                     end
%                     linkaxes([ax{1} ax{2} ax{3}])
%                 else
%                     nexttile(3)
%                     hold on
%                     for a = 1:length(uniqueAnimals)
%                         dum = char(uniqueAnimals(a));
%                         if contains(dum,'EB')
%                             dum = dum(1:2);
%                         else
%                             dum = dum(1:end-1);
%                         end
%                         s(a) = plot(data{3}(a,:), '-', 'Color', [c .3], 'LineWidth',.5);
%                     end
%                     plot_areaerrorbar(movmean(data{3},2), opts);
%                     % plot(nanmean(data{3}), 'LineWidth',2, 'Color',[c .8])
%                     title(conds(3))
%                     xlim(figxlim)
%                     xlabel('Session number')
%                     ylabel(figyLabs(i))
%                 end
%                 % lg = legend(s,uniqueAnimals);
%                 % lg.Layout.Tile = 'East';
%                 xlabel(t, 'SessionNum')
%                 title(t, strcat(depVars(i), "-", trainingTypes(tt)))
%             end
%         end
%     end
%  end
%