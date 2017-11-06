%% Import the Data
% read the ground truth joint positions
load('GroundTruth.mat')

% Reshape GroundTruth from num x 63 to num x 3 x 21
groundTruthReshaped = reshape(groundTruth,[size(groundTruth,1),3,21]);

%% Calculate largest x,y,z distance to MCP for every image 
% initialize matrix for results
distance_xyz = zeros(size(groundTruth,1),3);

% Calculate distances
tic
for i=1:size(groundTruth,1)
    distance_temp = abs(squeeze(groundTruthReshaped(i,:,:))-repmat(groundTruthReshaped(i,:,4)',1,21));
    distance_xyz(i,:) = max(distance_temp,[],2)';
end
toc
%% Create Histograms
% x distances
figure('Name','x distances')
hist(distance_xyz(:,1),40)
title('x distances')

% y distances
figure('Name','y distances')
hist(distance_xyz(:,2),40)
title('y distances')

% z distances
figure('Name','z distances')
hist(distance_xyz(:,3),40)
title('z distances')

%% Extract largest overall x,y,z distance to MCP
max_dist = max(distance_xyz);
fprintf('Max x distance: %f\n',max_dist(1));
fprintf('Max y distance: %f\n',max_dist(2));
fprintf('Max z distance: %f\n',max_dist(3));
