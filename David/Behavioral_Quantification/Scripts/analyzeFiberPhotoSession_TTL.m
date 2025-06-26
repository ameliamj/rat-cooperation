function processedData = analyzeFiberPhotoSession_TTL(photometryData, varargin)
% analyzeFiberPhotoSession Analyze fiber photometry data with 405nm correction and visualization
%
% Inputs:
%   photometryData: Structure containing fiber photometry data with fields:
%       - x405.data: 405nm channel data
%       - x465.data: 465nm channel data
%       - x560.data: 560nm channel data
%       - TTL.data: TTL data
%       - TTL.onset: TTL onset times
%       - x465.fs: Sampling frequency
%
% Optional Name-Value Pairs:
%   'BaselinePeriod': [start end] in seconds, default [-5 -2]
%   'WindowSize': [start end] in seconds, default [-5 10]
%   'DownsampleFactor': Integer, default 10
%   'SmoothingWindow': Integer, default 20
%   'PlotFigures': Logical, default true
%
% Output:
%   processedData: Structure containing processed data and analysis results

% Parse inputs
p = inputParser;
addParameter(p, 'BaselinePeriod', [-5 -2], @isnumeric);
addParameter(p, 'WindowSize', [-5 10], @isnumeric);
addParameter(p, 'DownsampleFactor', 10, @isnumeric);
addParameter(p, 'SmoothingWindow', 20, @isnumeric);
addParameter(p, 'PlotFigures', true, @islogical);
addParameter(p, 'SavePath', '', @ischar);  % <-- NEW
parse(p, varargin{:});

% Extract parameters
BASELINE_PER = p.Results.BaselinePeriod;
winSize = p.Results.WindowSize;
downSamp = p.Results.DownsampleFactor;
SMOOTH_WINDOW = p.Results.SmoothingWindow;

% Define channels
chan = ["x405" "x465" "x560"];
ARTIFACT_THRESH = [-inf inf];

% Create downsampled timestamp array
ts = linspace(0, (1/photometryData.x465.fs)*length(photometryData.x465.data), length(photometryData.x465.data));
ts = mean(reshape([ts(:); nan(mod(-numel(ts),10),1)],downSamp,[]));

% Initialize TTLs table
TTLs = table();
TTLs.code = diff([0; photometryData.TTL.data]);
TTLs.ts = photometryData.TTL.onset;
uniqueCodes = unique(TTLs.code(TTLs.code > 1));
uniqueCodes(~ismember(uniqueCodes, [2 4 8 16])) = [];

% Calculate window parameters
samplesPerSecond = photometryData.x465.fs/downSamp;
windowSamples = round(diff(winSize) * samplesPerSecond);
timeVector = linspace(winSize(1), winSize(2), windowSamples);

% Initialize processed data structure
processedData = struct();

% First pass: Get all raw data
for c = 1:length(chan)
    % Downsample the data
    dataDum = photometryData.(chan(c)).data;
    dataDum = mean(reshape([dataDum(:); nan(mod(-numel(dataDum),10),1)],downSamp,[]));
    
    % Remove artifacts
    dataDum(dataDum > ARTIFACT_THRESH(2) | dataDum < ARTIFACT_THRESH(1)) = NaN;
    
    % Initialize cell array for this channel
    TTLs.(chan(c)) = cell(height(TTLs), 1);
    
    for i = 1:height(TTLs)
        startIdx = find(ts > round(TTLs.ts(i),4)+winSize(1),1,'first');
        endIdx = find(ts < round(TTLs.ts(i),4)+winSize(2),1,'last');
        
        if isempty(startIdx) || isempty(endIdx)
            TTLs.(chan(c)){i} = nan(1, windowSamples);
            continue;
        end
        
        dum = dataDum(startIdx:endIdx-1);
        
        % Standardize window size
        if length(dum) < windowSamples
            dum = [dum, nan(1, windowSamples - length(dum))];
        elseif length(dum) > windowSamples
            dum = dum(1:windowSamples);
        end
        
        TTLs.(chan(c)){i} = dum;
    end
    
    % Convert cell array to matrix for easier processing
    processedData.(chan(c)) = cell2mat(TTLs.(chan(c)));
end

% Second pass: Apply 405 fitting and z-scoring
for c = 1:length(chan)
    if ~strcmp(chan(c), 'x405')
        currentSignal = processedData.(chan(c));
        reference405 = processedData.x405;
        correctedSignal = zeros(size(currentSignal));
        
        for i = 1:size(currentSignal, 1)
            F465 = currentSignal(i, :);
            F405 = reference405(i, :);
            
            validIdx = ~isnan(F465) & ~isnan(F405);
            if sum(validIdx) > 2
                x = F405(validIdx);
                y = F465(validIdx);
                
                % Remove any remaining NaNs
                valid = ~isnan(x) & ~isnan(y);
                x = x(valid);
                y = y(valid);
                
                % Only proceed if we have enough and varied points
                if numel(x) > 2 && range(x) > 1e-6
                    bls = polyfit(x, y, 1);
                else
                    warning('polyfit skipped: insufficient or flat F405 data');
                    bls = [NaN NaN];  % or use fallback baseline
                end
                
                Y_fit = bls(1) .* F405 + bls(2);
                correctedSignal(i, :) = F465 - Y_fit;
            else
                correctedSignal(i, :) = nan(1, size(currentSignal, 2));
            end
        end
        
        corrected_fieldname = [char(chan(c)) '_corrected'];
        processedData.(corrected_fieldname) = correctedSignal;
        
        baselineIdx = timeVector >= BASELINE_PER(1) & timeVector <= BASELINE_PER(2);
        zall = zeros(size(correctedSignal));
        
        for i = 1:size(correctedSignal, 1)
            baselineData = correctedSignal(i, baselineIdx);
            zb = mean(baselineData, 'omitnan');
            zsd = std(baselineData, 'omitnan');
            
            if zsd > 0
                zall(i,:) = (correctedSignal(i,:) - zb) / zsd * 2;
            else
                zall(i,:) = nan(1, size(correctedSignal, 2));
            end
        end
        
        zscored_fieldname = [char(chan(c)) '_zscored'];
        processedData.(zscored_fieldname) = zall;
    end
end

% Store additional information in output structure
processedData.TTLs = TTLs;
processedData.timeVector = timeVector;
processedData.uniqueCodes = uniqueCodes;

% If plotting is requested
if p.Results.PlotFigures
    plotFiberPhotoResults(processedData, timeVector, uniqueCodes, TTLs, SMOOTH_WINDOW);
end

% Save the processed data if SavePath is provided
if ~isempty(p.Results.SavePath)
    try
        save(p.Results.SavePath, 'processedData');
        fprintf('Processed data saved to: %s\n', p.Results.SavePath);
    catch saveErr
        warning('Failed to save processed data: %s', saveErr.message);
    end
end

end

%%
function plotFiberPhotoResults(processedData, timeVector, uniqueCodes, TTLs, SMOOTH_WINDOW)
% Define plotting parameters
smoothSignal = @(x, window) movmean(x, window, 2, 'omitnan');
channelColors = containers.Map();
channelColors('x465') = [0 0.5 0];
channelColors('x560') = [0.6 0 0];

% Define TTL code labels
ttlLabels = containers.Map();
ttlLabels('1') = 'Session Start';
ttlLabels('2') = 'Left Lever Press';
ttlLabels('4') = 'Right Lever Press';
ttlLabels('8') = 'Left Magazine Entry';
ttlLabels('16') = 'Right Magazine Entry';

% Create figure
figure('Position', [100 100 600 1000]);

plotChan = ["x465" "x560"];

for i = 1:length(uniqueCodes)
    currentCode = uniqueCodes(i);
    codeIndices = TTLs.code == currentCode;
    
    % Get TTL label
    ttlLabel = ttlLabels(num2str(currentCode));
    if isempty(ttlLabel)
        ttlLabel = sprintf('Code %d', currentCode);
    end
    
    for c = 1:length(plotChan)
        basePos = (i-1)*length(plotChan)*2 + (c-1)*2;
        
        zscored_fieldname = [char(plotChan(c)) '_zscored'];
        data = processedData.(zscored_fieldname)(codeIndices,:);
        currentColor = channelColors(char(plotChan(c)));
        
        % Smooth and sort the data
        smoothed_data = zeros(size(data));
        for trial = 1:size(data,1)
            smoothed_data(trial,:) = smoothSignal(data(trial,:), SMOOTH_WINDOW);
        end
        
        % [peak_values, peak_times] = max(abs(smoothed_data(:,timeVector>0)), [], 2);
        % [~, sort_idx] = sortrows([peak_values, peak_times], [-1 2]);
        % smoothed_data = smoothed_data(sort_idx,:);
        % data = data(sort_idx,:);
        
        % Plot mean trace
        ax1 = subplot(length(uniqueCodes)*2, length(plotChan), basePos + 1);
        meanTrace = mean(smoothed_data, 1, 'omitnan');
        semTrace = std(smoothed_data, [], 1, 'omitnan') / sqrt(sum(codeIndices));
        
        fill([timeVector fliplr(timeVector)], ...
             [meanTrace+semTrace fliplr(meanTrace-semTrace)], ...
             currentColor, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
        hold on;
        plot(timeVector, meanTrace, 'Color', currentColor, 'LineWidth', 2);
        
        line([0 0], ylim, 'Color', 'k', 'LineStyle', '--');
        line(xlim, [0 0], 'Color', 'k', 'LineStyle', ':');
        
        % Updated title with TTL label
        if strcmp(plotChan(c), 'x465')
            signalLabel = 'GCaMP';
        else
            signalLabel = 'jRECO';
        end
        title(sprintf('%s - %s (n=%d)', ttlLabel, signalLabel, sum(codeIndices)), 'Interpreter', 'none');
        xlabel('Time (s)');
        ylabel('Z-score');
        grid on;
        y_limits = ylim;
        
        % Plot heatmap
        ax2 = subplot(length(uniqueCodes)*2, length(plotChan), basePos + 2);
        imagesc(timeVector, 1:size(smoothed_data,1), smoothed_data);
        % clim(y_limits);
        colormap(ax2, 'turbo');
        
        cb = colorbar;
        ylabel(cb, 'Z-score');
        
        hold on;
        line([0 0], ylim, 'Color', 'w', 'LineStyle', '--');
        
        % Updated title with TTL label
        title(sprintf('Instance - %s - %s', ttlLabel, signalLabel), 'Interpreter', 'none');
        xlabel('Time (s)');
        ylabel('Instance # (Sorted)');
        
        axis tight;
        set(gca, 'YDir', 'reverse');
    end
end
end