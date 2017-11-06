%% Import the Data
% read the ground truth offsets
load('shift_gt.mat')

% take absolute value
offset = abs(shift_gt');

%% Convert to pixel/normalized shift
offset(1:2,:) = offset(1:2,:)*(64/135);
offset(3,:) = offset(3,:)/235;

%% Create Histograms
% x offsets
figure('Name','x offsets')
hist(offset(1,:),40)
title('x offsets')

% y offsets
figure('Name','y offsets')
hist(offset(2,:),40)
title('y offsets')

% z offsets
figure('Name','z offsets')
hist(offset(3,:),40)
title('z offsets')

%% Extract largest overall x,y,z offset
max_off = max(offset,[],2);
fprintf('Max x offset: %f\n',max_off(1));
fprintf('Max y offset: %f\n',max_off(2));
fprintf('Max z offset: %f\n',max_off(3));