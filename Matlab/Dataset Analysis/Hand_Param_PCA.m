%% Import the Data
% write .txt (in same folder) file into MATLAB array with row offset 0 and 
% column offset 1 (skip image name)
load('GroundTruth.mat');

%% Calculate Bone Lengths (Normalized by Hand Scale and De-meaned)
% specify the pairs of 3D positions that represent start and end of the 
% respective bone 
posPairs = [1 2 ; 1  3 ; 1  4; 1 5; 1 6;   %wrist to finger bases
            2 7 ; 7  8 ; 8  9;             %thumb
            3 10; 10 11; 11 12;            %index
            4 13; 13 14; 14 15;            %middle
            5 16; 16 17; 17 18;            %ring
            6 19; 19 20; 20 21];           %little

% Reshape GroundTruth from num x 63 to num x 3 x 21
groundTruthReshaped = reshape(groundTruth,[length(groundTruth),3,21]);

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

% get vector of mean bone lengths
boneLengths_mean = mean(boneLengths,2);

% subtract mean from bone lengths to get offsets
boneLengths = (boneLengths-boneLengths_mean)';

%% PCA of Scale Normalized and De-meaned Bone Lengths
% Calculate PCA of bone lengths: transform matrix + eigenvalues
[bonePCA,~,eigenVal] = pca(boneLengths);

%% Project Data on PCA base
% Number of components Taken into account
components = 9;

% Project and re-project data
W = bonePCA(:,1:components);
boneLengths_repr = (W*W'*boneLengths')';

% Calculate re-projection error (in mm avg per bone)
repr_error = sum(sum(abs(boneLengths_repr-boneLengths)))/(size(boneLengths,1)*size(boneLengths,2));

%% Calculate Scale Normalized and De-meaned Base Shape Parameters
% specify the pairs of start and end 
shapePairs = [2 3; 3 4; 4 5; 5 6];   %finger bases to each other

% Calculate base shapes
baseShapes = zeros(length(boneLengths),4);
for i=1:4
    baseShapes(:,i) = sqrt(sum((groundTruthReshaped(:,:,shapePairs(i,1))-groundTruthReshaped(:,:,shapePairs(i,2))).^2,2));
end
baseShapes = baseShapes';

% normalize base shapes by hand scales
baseShapes = baseShapes./handScales;

% get vector of mean base shapes
baseShapes_mean = mean(baseShapes,2);

% subtract mean from base shapes to get offsets
baseShapes = (baseShapes-baseShapes_mean)';

%% PCA only of Shape Parameters
% Calculate PCA of hand shape paramters
[shapeOnlyPCA,~,eigenValShapeOnly] = pca(baseShapes);

% Number of components Taken into account
components_bs = 3;

% Project and re-project data
Wbs = shapeOnlyPCA(:,1:components_bs);
baseShapes_repr = (Wbs*Wbs'*baseShapes')';

% Calculate re-projection error (in mm avg per shape parameter)
repr_error_base = sum(sum(abs(baseShapes-baseShapes_repr)))/(size(baseShapes,1)*size(baseShapes,2));

%% Base Shapes and Bone Lengths Together
% Initialize vector for shape parameters
boneLengths_aug = [boneLengths baseShapes];

% Calculate PCA of hand shape paramters
[shapePCA,~,eigenValShape] = pca(boneLengths_aug);

% Number of components Taken into account
components_aug = 11;

% Project and re-project data
Wb = shapePCA(:,1:components_aug);
boneLengths_aug_repr = (Wb*Wb'*boneLengths_aug')';

% Calculate re-projection error (in mm avg per parameter)
repr_error_aug = sum(sum(abs(boneLengths_aug_repr-boneLengths_aug)))/(size(boneLengths_aug,1)*size(boneLengths_aug,2));
