close all; clear; clc;


% set the ground truth path and the foreground map path
gtPath = 'C:\Users\Administrator\Desktop\M2CS\Evaluation\dut500-gt\';
% fgPath = 'C:\Users\Administrator\Desktop\M2CS\my_SG2\dut\';
fgPath = 'C:\Users\Administrator\Desktop\M2CS\Evaluation\SOTA\DUT\SG';

% load the gtFiles
gtFiles = dir(fullfile(gtPath));
preFiles = dir(fullfile(fgPath));

% for each gtFiles
S_score = zeros(1,length(gtFiles)-2);
for i = 1:length(gtFiles)-2
    fprintf('Processing %d/%d...\n',i,length(gtFiles)-2);
    
    % load GT
    [GT,map] = imread(fullfile(gtPath,gtFiles(i+2).name));
    if numel(size(GT))>2
        GT = rgb2gray(GT);
    end
    GT = logical(GT);
    
    % in some dataset(ECSSD) some ground truth is reverse when map is not none
%     if ~isempty(map) && (map(1)>map(2))
%         GT = ~GT;
%     end
    
    % load FG
    prediction = imread(fullfile(fgPath,preFiles(i+2).name));
    if numel(size(prediction))>2
        prediction = rgb2gray(prediction);
    end
    
    % Normalize the prediction.
    d_prediction = double(prediction);
    if (max(max(d_prediction))==255)
        d_prediction = d_prediction./255;
    end
    d_prediction = reshape(mapminmax(d_prediction(:)',0,1),size(d_prediction));
    
    % evaluate the S-measure score
    score = StructureMeasure(d_prediction,GT);
    S_score(i) = score;
    
end

fprintf('The average S-measure is:%.3f\n',mean2(S_score));


