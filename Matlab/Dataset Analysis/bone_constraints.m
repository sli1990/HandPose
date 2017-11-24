%% Import the Data
% write .txt (in same folder) file into MATLAB array with row offset 0 and 
% column offset 1 (skip image name)
load('GroundTruth.mat');

%% Calculate Bone Lengths (Normalized by Hand Scale)
% specify the pairs of 3D positions that represent start and end of the 
% respective bone 
posPairs = [1 2 ; 1  3 ; 1  4; 1 5; 1 6;   %wrist to finger bases
            2 7 ; 7  8 ; 8  9;             %thumb
            3 10; 10 11; 11 12;            %index
            4 13; 13 14; 14 15;            %middle
            5 16; 16 17; 17 18;            %ring
            6 19; 19 20; 20 21];           %little

% Reshape GroundTruth from num x 63 to num x 3 x 21
groundTruthReshaped = reshape(groundTruth,[length(groundTruth),3,21])/1000;

% Initialize vector for bone lengths
boneLengths = zeros(length(groundTruthReshaped),20);

% Calculate bone lengths
for i=1:20
    boneLengths(:,i) = sqrt(sum((groundTruthReshaped(:,:,posPairs(i,1))-groundTruthReshaped(:,:,posPairs(i,2))).^2,2));
end
boneLengths = boneLengths';

% get vector of mean bone lengths
boneLengths_mean = mean(boneLengths,2);

% get hand scales 
handScales = sum(boneLengths,1)./sum(boneLengths_mean,1);

% normalize bone lengths by hand scales
boneLengths = boneLengths./handScales;

%% Find Min and Max value for each bone
% Initialize matrix to store values as Float32
boneConstraints = single(zeros(2,20));

for j=1:20
    boneConstraints(1,j) = single(min(boneLengths(j,:)));
    boneConstraints(2,j) = single(max(boneLengths(j,:)));
end