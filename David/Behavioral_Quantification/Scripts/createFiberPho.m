% Path to the CSV
csvPath = '/gpfs/radev/project/saxena/drb83/rat-cooperation/David/Behavioral_Quantification/Sorted_Data_Files/dyed_preds_all_valid.csv';

% Read the table
T = readtable(csvPath, 'VariableNamingRule', 'preserve');

% Initialize file path list
filePaths = {};

% Filter and build file paths
for i = 1:height(T)
    if islogical(T.("fiber pho")(i)) && T.("fiber pho")(i)
        session = T.session{i};
        vid = T.vid{i};
        fullPath = fullfile('/gpfs/radev/pi/saxena/aj764/PairedTestingSessions', session, 'Neuronal', [vid '.mat']);
        filePaths{end+1} = fullPath;
    end
end

% Create output directory if it doesn't exist
outputFolder = '/gpfs/radev/pi/saxena/aj764/processedFiberPho'
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Create output directories
outputMatFolder = '/gpfs/radev/pi/saxena/aj764/neuronalData/processed'
outputCSVFolder = 'gpfs/radev/pi/saxena/aj764/neuronalData/processed_csv';
subfolders = {'x465_corrected', 'x560_corrected', 'TTLs', 'timeVector'};
if ~exist(outputMatFolder, 'dir'), mkdir(outputMatFolder); end
for s = 1:length(subfolders)
    sub = fullfile(outputCSVFolder, subfolders{s});
    if ~exist(sub, 'dir'), mkdir(sub); end
end

% Loop through each file
for i = 1:length(filePaths)
    inputFile = filePaths{i};
    
    try
        % Load the .mat file
        data = load(inputFile);

        % Try to detect the photometry data variable name
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

        % Get base name of file (without extension)
        [~, name, ~] = fileparts(inputFile);

        % Output file paths
        outputMatFile = fullfile(outputMatFolder, [name '_processed.mat']);

        % Run the analysis
        fprintf('Processing %s...\n', inputFile);
        processedData = analyzeFiberPhotoSession_TTL(photometryData, ...
            'SavePath', outputMatFile, ...
            'PlotFigures', false);  % Store return value

        % Export CSVs
        writematrix(processedData.x465_corrected, fullfile(outputCSVFolder, 'x465_corrected', [name '_x465_corrected.csv']));
        writematrix(processedData.x560_corrected, fullfile(outputCSVFolder, 'x560_corrected', [name '_x560_corrected.csv']));
        writetable(processedData.TTLs, fullfile(outputCSVFolder, 'TTLs', [name '_TTLs.csv']));
        writematrix(processedData.timeVector(:), fullfile(outputCSVFolder, 'timeVector', [name '_timeVector.csv']));  % Ensure column format

    catch ME
        warning('Error processing %s: %s', inputFile, ME.message);
    end
end

fprintf('Batch processing + CSV export complete.\n');
