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

%% Define HDF 5 file and components: depth image, 3D joint positions, hand
% scale, finger scales, 3D center, ..., ...
h5fileSize = 27032;
depthImages=zeros(192,192,1,h5fileSize);
joints=zeros(63,h5fileSize);
scales=ones(1,h5fileSize);
fingerScales=ones(20,h5fileSize);
center3ds=ones(3,h5fileSize);
center3dsOrig=ones(3,h5fileSize);
centerOffset=zeros(3,h5fileSize);

%% Calculate the components of the HDF5 file 

for frameIndex=930001:957032
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
    
    %% Crop image
    % get start (upper left corner) and end (bottom right corner) pixel
    % coordinates for cropping the image (in uvd camera coordinates)
    cropStart = zeros(3,1);
    [cropStart(1), cropStart(2), cropStart(3)]= xyz2uvd(center3d(1)-cropSizePlus*1.41, center3d(2)-cropSizePlus*1.41 ,center3d(3));
    cropEnd = zeros(3,1);
    [cropEnd(1), cropEnd(2), cropEnd(3)]= xyz2uvd(center3d(1)+cropSizePlus*1.41, center3d(2)+cropSizePlus*1.41 ,center3d(3));    
    cropStart=round(cropStart);     cropEnd=round(cropEnd);

    % 3D array with x,y indice maps in the first two components of the
    % first array dimension
    croppedUVD = zeros(3, cropEnd(2)-cropStart(2)+1, cropEnd(1)-cropStart(1)+1);      
    croppedUVD(1,:,:) = ones(cropEnd(2)-cropStart(2)+1,1)*(cropStart(1):cropEnd(1));
    croppedUVD(2,:,:) = (cropStart(2):cropEnd(2))' * ones(1,cropEnd(1)-cropStart(1)+1);
    
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
    
    %% Padding
    % Pad 900 zeros around every image border
    imageSize = size(image);
    padSize = 900;
    imagePad = [zeros(imageSize(1),padSize),image,zeros(imageSize(1),padSize)];
    imagePad = [zeros(padSize,imageSize(2)+padSize*2);imagePad;zeros(padSize,imageSize(2)+padSize*2)];

    %% Finalize u,v,d 3D representation
    % extraxt crop excerpt from padded image and put it in third component
    % of firrst array dimension (z = d)
    croppedUVD(3,:,:) = imagePad(cropStart(2)+padSize:cropEnd(2)+padSize, cropStart(1)+padSize:cropEnd(1)+padSize);

    % rename components into u,v,d --> 3D array with u,v,d components of
    % cropped image
    u = croppedUVD(1,:,:); v = croppedUVD(2,:,:); d=croppedUVD(3,:,:);

    %% Normalize origin
    % transform u,v,d components back to x,y,z and put them into 3 columns
    % of points vector, then transpose --> 3x...
    points = zeros(length(croppedUVD(:))/3, 3);
    [points(:,1),points(:,2),points(:,3)] = uvd2xyz(u(:),v(:),d(:));
    points= points';

    % only keep points whose depth value lies within certain threshold
    % around the crop center depth
    points = points(:,(points(3,:)<center3dOrig(3)+cropDepthPlus*1.41));
    points = points(:,(points(3,:)>center3dOrig(3)-cropDepthPlus*1.41));

    % rotate x and y and subtract rotation center
    points = rotMat*points;
    points(1,:) = points(1,:)-center3dRot(1)*ones(1,size(points,2));
    points(2,:) = points(2,:)-center3dRot(2)*ones(1,size(points,2));

    % --> rotation center is origin now
    center3dRot(1)=0;
    center3dRot(2)=0;

    %% Threshold depth values
    % only keep points that lie within 3D crop cube and transform to camera
    % coordinate system uvd
    points = points(:,(points(3,:)<center3dRot(3)+cropDepthPlus));
    points = points(:,(points(3,:)>center3dRot(3)-cropDepthPlus));
    points = points(:,(points(1,:)>-cropSizePlus & points(1,:)<cropSizePlus & points(2,:)>-cropSizePlus & points(2,:)<cropSizePlus));
    [u,v,d] = xyz2uvd(points(1,:),points(2,:),points(3,:));
    
    %% Create height map
    % sort remaining points in uvd into a 480x640 "height map": if a point
    % with a certain u,v coordinate exists: insert depth value at that
    % position
    heightmap = zeros(480,640);
    for ptInd=1:size(points,2)
        vvv = max(min(round(v(ptInd)),480),1);
        uuu = max(min(round(u(ptInd)),640),1);
        heightmap(vvv, uuu) = d(ptInd);
    end

    %% Crop image finally
    % calculate top left and bottom right positions of cropping frame in
    % xyz and extract crop from padded height map (in xyz)
    cropStart = zeros(3,1);
    [cropStart(1), cropStart(2), cropStart(3)]= xyz2uvd(-cropSizePlus, -cropSizePlus ,center3dRot(3));
    cropEnd = zeros(3,1);
    [cropEnd(1), cropEnd(2), cropEnd(3)]= xyz2uvd(cropSizePlus, cropSizePlus ,center3dRot(3));    
    cropStart=round(cropStart);     cropEnd=round(cropEnd);
    imagePad = [zeros(imageSize(1),padSize),heightmap,zeros(imageSize(1),padSize)];
    imagePad = [zeros(padSize,imageSize(2)+padSize*2);imagePad;zeros(padSize,imageSize(2)+padSize*2)];
    
    d = imagePad(cropStart(2)+padSize:cropEnd(2)+padSize, cropStart(1)+padSize:cropEnd(1)+padSize);

    % index of image within current file: 127536 (30000) --> 7536
    batchIndex = frameIndex-floor((frameIndex-1)/h5fileSize)*h5fileSize;
    
    %% Size normalization
    % just normalized depth image
    centerDepth = center3dRot(3);

    % resize normalized image to 192x192 with nearest neighbor
    % interpolation and save size before re-sizing
    sizeBeforeResize = size(d);
    d = imresize(squeeze(d),[192,192], 'nearest'); % squeeze removes singleton dimensions
    
    %% 3x3 median filtering
    d = medfilt2(d,[3,3]);
    
    %% Determine bounding box shift
    % simply the MCP position in COM coordinate system
    boundingBoxOffset = [xyzDemean(1,4) xyzDemean(2,4) xyzDemean(3,4)];

    %% Depth normalization
    % set negative depth values to maximum value and set rest of the depth
    % values if too low or too high (centerDepth+-cropDepth) to threshold
    % values
    d(d<=0) = centerDepth+cropDepthPlus;
    d((d-centerDepth)>cropDepthPlus) = centerDepth+cropDepthPlus;
    d((centerDepth-d)>cropDepthPlus) = centerDepth-cropDepthPlus;

    % subtract center depth and normalize to crop depth --> -1 ... 1
    d = (d-centerDepth)./(cropDepthPlus);

    %% Write HDF5 file
    % depth image
    depthImages(:,:,1,batchIndex) = d;
    % joint ground truth centered and rotated
    joints(:,batchIndex)= (reshape(xyzDemean,[63,1])./1000);
    % hand and individual finger scales
    scales(batchIndex) = bonelengthsScales(frameIndex);
    fingerScales(:,batchIndex) = bonelengthsIndividualScales(:,frameIndex);
    % 3D center prior to any rotations
    center3dsOrig(:,batchIndex) = center3dOrig;
    % rotated and made origin center
    center3ds(:,batchIndex) = center3dRot;
    % for re-centering
    centerOffset(:,batchIndex) = (boundingBoxOffset./1000);

    if (frameIndex==957032)
         h5filename = ['Hands2017Train_32b.h5'];
         h5filename
         delete(h5filename);
         h5create(h5filename,'/depth',[192,192,1,h5fileSize],'Datatype','single');
         h5write(h5filename,'/depth', depthImages);
         h5create(h5filename,'/joint',[63,h5fileSize],'Datatype','double');
         h5write(h5filename,'/joint',joints);
         h5create(h5filename,'/handScale',[1,h5fileSize],'Datatype','double');
         h5write(h5filename,'/handScale',scales);
         h5create(h5filename,'/fingerScale',[20,h5fileSize],'Datatype','double');
         h5write(h5filename,'/fingerScale',fingerScales);
         h5create(h5filename,'/center3ds',[3,h5fileSize],'Datatype','double');
         h5write(h5filename,'/center3ds',center3ds);
         h5create(h5filename,'/center3dsOrig',[3,h5fileSize],'Datatype','double');
         h5write(h5filename,'/center3dsOrig',center3dsOrig);
         h5create(h5filename,'/offset',[3,h5fileSize],'Datatype','double');
         h5write(h5filename,'/offset',centerOffset);
         
        depthImages=zeros(192,192,1,h5fileSize);
        joints=zeros(63,h5fileSize);
        scales=ones(1,h5fileSize);
        fingerScales=ones(20,h5fileSize);
        center3ds=ones(3,h5fileSize);
        center3dsOrig=ones(3,h5fileSize);
        centerOffset=zeros(3,h5fileSize);

    end

end