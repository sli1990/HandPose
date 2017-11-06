%% Bone length, hand scale calculation

% load matrix with ground truth joint positions (957032x63)
load GroundTruth.mat;

% edge definition to calculate bone lengths
edges = [1 2; 1 3; 1 4; 1 5; 1 6; %wrist to finger bases
    2 7; 7 8; 8 9;  %thumb
    3 10; 10 11; 11 12 %index
    4 13; 13 14; 14 15 %middle
    5 16; 16 17; 17 18 %ring
    6 19; 19 20; 20 21]; %little

% reshape joint matrix into num x 3 x 21 and define 20 x num bone length
% matrix
xyzs= reshape(groundTruth, [length(groundTruth),3,21]);
bonelengths=zeros(20,length(groundTruth));

% calculate bone lengths and fill into matrix
for i=1:20
    thisBoneLengths = xyzs(:,:,edges(i,1))-xyzs(:,:,edges(i,2));
    thisBoneLengths = sqrt(sum(thisBoneLengths.^2,2));
    bonelengths(i,:) = thisBoneLengths;
end

% get vector of mean bone lengths
bonelengths_mean = mean(bonelengths,2);

% get hand scales and individual bone length scales
bonelengthsScales = sum(bonelengths,1) ./ sum(bonelengths_mean,1);
bonelengthsIndividualScales = bonelengths ./ (mean(bonelengths_mean,2)*ones(1,957032));

%% Calculate shift ground truth
shift_gt = zeros(957032,3);

for frameIndex=1:957032
    % signal every 100th frame
    if mod(frameIndex,100)==0
        disp(frameIndex)
    end
    
    % get png image and read into double
    filename = ['/home/dhri-dz/Documents/HandPose/Dataset/Training/images/image_D',num2str(frameIndex,'%08d'),'.png'];
    image = double(imread(filename));
    
    % get ground truth for current image and reshape into 3x21
    xyz = groundTruth(frameIndex, :);
    xyz = reshape(xyz, [3,21]);

    % crop size
    cropSize = 135;
    cropSizePlus = cropSize*1.5;
    cropDepth = 135;
    cropDepthPlus = cropDepth + 100;
    
    %% Get bounding box centroid
    % transform ground truth joint locations into camera coordinate system
    [u,v,dd] = xyz2uvd(xyz(1,:),xyz(2,:),xyz(3,:));
    
    % find minimum and maximum joint location in u, v pixel coordinates
    us = min(max(min(u),1),635); ue = max(min(max(u),640),5); 
    vs = min(max(min(v),1),475); ve = max(min(max(v),480),5);

    % round the bounding box corner pixel values
    vs=round(vs); ve=round(ve); us=round(us); ue=round(ue);
    
    %% Get 3D centroid
    % round the bounding box corner pixel values
    vs=round(vs); ve=round(ve); us=round(us); ue=round(ue);

    % extract the bounding box from the image (-->cropping)
    cropped = image(vs:ve, us:ue);

    % maked cropped image a vector and set all 0 values to max + 1
    depths = cropped(:);
    depths(depths==0) = max(depths(:))+1;

    %% Cluster
    % do kmean clustering of depth values into 2 clusters: small and large
    % depth values
    [idx, C] = kmeans(depths(:),2,'Start',[min(depths(:)); max(depths(:))+1]);

    % take cluster center of small depth values as depth center of hand (as
    % close to camera)
    centerDepth = min(C);

    %% Get 3D center
    % take depth value of middle finger base as depth center if depth
    % center is more than 100 away from it (catch erroneous depth centers)
    if (abs(centerDepth-xyz(3,4))>100)
        centerDepth=xyz(3,4);
    end
    
    % 3D center is center of bounding box + depth center --> transform back
    % into world coordinates
    center3d = zeros(3,1);
    [center3d(1), center3d(2), center3d(3)]= uvd2xyz(us+0.5*(ue-us),vs+0.5*(ve-vs),centerDepth);   
    
    %% Joints de-mean and rotation
    % get rotations in x and y direction, rotate the 3D center, the ground
    % truth, get the rotation matrix and subtract rotation center from
    % ground truth (left handed coordinate system)
    center3dOrig = center3d;
    aroundYAngle = atan2(center3d(1),center3d(3))/pi*180;
    center3d = roty(-aroundYAngle)*center3d;
    aroundXAngle = atan2(center3d(2),center3d(3))/pi*180;
    rotMat = roty(-aroundYAngle)*rotx(aroundXAngle);
    xyz = rotMat*xyz;
    center3dRot = rotMat*center3dOrig;
    xyzDemean = xyz - center3dRot *ones(1,21);
    
    %% Determine bounding box shift
    % simply the MCP position in COM coordinate system
    shift_gt(frameIndex,:) = [xyzDemean(1,4) xyzDemean(2,4) xyzDemean(3,4)];
    
end
