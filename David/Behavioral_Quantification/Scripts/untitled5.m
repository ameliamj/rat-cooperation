% List of full paths to your .mat files
filePaths = {
    'data/100324_Cam1_TrNum1_PV_KL005R.mat';
    'data/100324_Cam1_TrNum2_PV_KL005R.mat';
    'data/100324_Cam1_TrNum3_PV_KL005R.mat'
    % Add more file paths here
};

% Create output directory if it doesn't exist
outputFolder = 'processed';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
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

        % Create output filename
        [~, name, ~] = fileparts(inputFile);
        outputFile = fullfile(outputFolder, [name '_processed.mat']);

        % Run the analysis and save
        fprintf('Processing %s...\n', inputFile);
        analyzeFiberPhotoSession_TTL(photometryData, ...
            'SavePath', outputFile, ...
            'PlotFigures', false);  % Turn off plots for batch run

    catch ME
        warning('Error processing %s: %s', inputFile, ME.message);
    end
end

fprintf('Batch processing complete.\n');
