% Path to the CSV
csvPath = '/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_all_valid.csv';

% Read the table with original column headers preserved
T = readtable(csvPath, 'VariableNamingRule', 'preserve');

% Initialize list of valid file paths
filePaths = {};

% Construct .mat file paths only for valid rows
for i = 1:height(T)
    % Robust check for "fiber pho" being true
    fiberPhoValue = T.("fiber pho")(i);
    if ismember(lower(string(fiberPhoValue)), ["true", "1"])
        session = T.session{i};
        vid = T.vid{i};
        fullPath = fullfile('/gpfs/radev/pi/saxena/aj764/PairedTestingSessions', session, 'Neuronal', [vid '.mat']);
        if exist(fullPath, 'file') == 2
            filePaths{end+1} = fullPath;
        else
            warning('File does not exist: %s', fullPath);
        end
    end
    end

% Output folders
outputMatFolder = '/gpfs/radev/pi/saxena/aj764/neuronalData/processed';
outputCSVFolder = '/gpfs/radev/pi/saxena/aj764/neuronalData/processed_csv';
%csvSubfolders = {'x465_corrected', 'x560_corrected', 'TTLs', 'timeVector'};
csvSubfolders = {'x405', 'x465', 'x560'};

% Create output directories if they don’t exist
if ~exist(outputMatFolder, 'dir'), mkdir(outputMatFolder); end
for s = 1:length(csvSubfolders)
    sub = fullfile(outputCSVFolder, csvSubfolders{s});
    if ~exist(sub, 'dir'), mkdir(sub); end
end

% Loop through and process each file
for i = 1:length(filePaths)
    inputFile = filePaths{i};

    try
        % Load the .mat file
        data = load(inputFile);

        % Try to find the photometryData struct
        varNames = fieldnames(data);
        found = false;
        for v = 1:length(varNames)
            if isstruct(data.(varNames{v})) && isfield(data.(varNames{v}), 'x465')
                photometryData = data.(varNames{v});
                found = true;
                break;
            end
        end

        if ~found
            warning('No valid photometryData struct found in %s. Skipping.', inputFile);
            continue;
        end

        % Get filename without extension
        [~, name, ~] = fileparts(inputFile);

        % Output file path for processed .mat
        outputMatFile = fullfile(outputMatFolder, [name '_processed.mat']);

        % Run the analysis and store output
        fprintf('Processing %s...\n', inputFile);
        processedData = analyzeFiberPhotoSession_TTL(photometryData, ...
            'SavePath', outputMatFile, ...
            'PlotFigures', false);

        % Create expanded TTL tables for x405, x465, x560
        TTLs = processedData.TTLs;
        baseCols = table2cell(TTLs(:, {'code', 'ts'}));

        for ch = {'x405', 'x465', 'x560'}
            chName = ch{1};

            if ~isfield(TTLs, chName)
                warning('Field %s missing in TTLs for %s. Skipping.', chName, name);
                continue;
            end

            nestedField = TTLs.(chName);

            % Sanity check: make sure it's a table or cell array of tables
            if ~iscell(nestedField) && ~istable(nestedField)
                warning('%s is not a cell or table for file %s. Skipping.', chName, name);
                continue;
            end

            numRows = height(TTLs);
            expandedData = cell(numRows, 1528);
            baseCols = table2cell(TTLs(:, {'code', 'ts'}));

            for r = 1:numRows
                try
                    expandedData{r, 1} = baseCols{r, 1};  % code
                    expandedData{r, 2} = baseCols{r, 2};  % ts

                    % Extract the row of numeric data
                    rowContent = nestedField{r};  % should be a 1x1526 table
                    if istable(rowContent)
                        rowArray = table2array(rowContent);
                    elseif isnumeric(rowContent)
                        rowArray = rowContent;
                    else
                        error('Unexpected format in %s row %d.', chName, r);
                    end

                    if length(rowArray) ~= 1526
                        error('Length of %s row %d is not 1526.', chName, r);
                    end

                    expandedData(r, 3:end) = num2cell(rowArray);
                catch innerErr
                    warning('Skipping row %d of %s in file %s: %s', r, chName, name, innerErr.message);
                end
            end

            % Create and save table
            colNames = [{'code', 'ts'}, arrayfun(@num2str, 1:1526, 'UniformOutput', false)];
            expandedTable = cell2table(expandedData, 'VariableNames', colNames);
            outPath = fullfile(outputCSVFolder, chName, [name '_' chName '_TTLs.csv']);

            try
                writetable(expandedTable, outPath);
                fprintf('Saved: %s\n', outPath);
            catch writeErr
                warning('Could not write %s: %s', outPath, writeErr.message);
            end
        end
        
        % Save CSVs to respective folders
        %writematrix(processedData.x465_corrected, fullfile(outputCSVFolder, 'x465_corrected', [name '_x465_corrected.csv']));
        %writematrix(processedData.x560_corrected, fullfile(outputCSVFolder, 'x560_corrected', [name '_x560_corrected.csv']));
        %writetable(processedData.TTLs, fullfile(outputCSVFolder, 'TTLs', [name '_TTLs.csv']));
        %writematrix(processedData.timeVector(:), fullfile(outputCSVFolder, 'timeVector', [name '_timeVector.csv']));

    catch ME
        warning('Error processing %s: %s', inputFile, ME.message);
    end
end

fprintf('Batch processing + CSV export complete.\n');
