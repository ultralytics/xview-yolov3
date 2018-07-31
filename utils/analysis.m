% 659.bmp and 769.bmp are bad pictures, delete

clc; clear; close all
load json_data.mat

% make_small_chips()
% return

chip_id = zeros(numel(chips),1);  % 1-847
chip_number = zeros(numel(chips),1);  % 5-2619
for i=1:numel(chips)
    chip_id(i) = find(strcmp(chips(i),uchips));
    
    s = chips{i};
    s = s(1:end-4);
    chip_number(i) = str2double(s);
end

uchips_numeric = zeros(numel(uchips),1);
for i=1:numel(uchips)
    uchips_numeric(i) = eval(uchips{i}(1:end-4));
end

% clean coordinates that fall off images (remove or crop)
image_h = shapes(chip_id,1);
image_w = shapes(chip_id,2);
[coords, v] = clean_coords(coords, classes, image_h, image_w);
mean(v)

chip_id = chip_id(v);
chips = chips(v);
classes = classes(v);
image_h = image_h(v);
image_w = image_w(v); 
chip_number = chip_number(v); clear v i

% % reject images with < 10 targets
[uchip_number,~,~,n]=fcnunique(chip_number);
% fprintf('Images target count range: %g-%g\n',min(n),max(n))
fig; histogram(n,linspace(0,300,301))
a=sortrows([n, uchip_number],-1);
i=find(a(:,1)<2);
%fprintf('rm -rf %g.bmp\n',a(i,2))

% Target box width and height
w = coords(:,3) - coords(:,1);
h = coords(:,4) - coords(:,2);

% stats for outlier bbox rejection
[class_mu, class_sigma, class_cov] = per_class_stats(classes,w,h);
[~,~,~,n] = fcnunique(classes(:));
weights = 1./n(:)';  weights=weights/sum(weights);
vpa(n(:)')

% image weights (1395 does not exist, remove it)
%image_weights = accumarray(chip_id,weights(xview_classes2indices(classes)),[1 847]);
%i=uchips_numeric ~= 1395; 
%image_weights = image_weights(i)./sum(image_weights(i));
%fig; bar(uchips_numeric(i), image_weights)

%a=class_stats(:,[7 9]); 
%[~,i]=sort(prod(a,2));  a=a(i,:);
%vpa(a(:)',4)

% K-means normalized with and height for 9 points
C = fcn_kmeans([w h], 90);
[~, i] = sort(C(:,1).*C(:,2));
C = C(i,:)';

% image mean and std
i = ~all(stats==0,2);
shapes=shapes(i,:);
stats=stats(i,:);  % rgb_mean, rgb_std
stat_means = zeros(1,12);
for i=1:12
    stat_means(i) = mean(fcnsigmarejection(stats(:,i),6,3));
end

% output RGB stats (comes in BGR from cv2.imread)
rgb_mu = stat_means([3 2 1])   % dataset RGB mean
rgb_std = stat_means([6 5 4])  % dataset RGB std mean
hsv_mu = stat_means(7:9)   % dataset RGB mean
hsv_std = stat_means(10:12)  % dataset RGB std mean
anchor_boxes = vpa(C(:)',4)  % anchor boxes

wh = single([image_w, image_h]);
targets = single([classes(:), coords]);
id = single(chip_number);  numel(id)
% save('targets_c6.mat','wh','targets','id','class_mu','class_sigma','class_cov')


function [] = make_small_chips()
clc; close; clear
load('targets_c60.mat')
path_a = '/Users/glennjocher/downloads/DATA/xview/';

% rmdir([path_a 'classes'],'s')
% for i=0:59
%     mkdir(sprintf([path_a 'classes/%g'],i))
% end

uid = unique(id)';
class_count = zeros(1,60);
f_count = 0;  % file count
c_count = 0;  % chip count
length = 16;  % with padding
lengh_inner = 40;  % core size
X = zeros(650000,length,length,3, 'uint8');
Y = zeros(1,650000,'uint8');
border = 8;  % extra area around object of interest (for augmentation)
for i = uid
    f_count = f_count+1;
    fprintf('%g/847\n',f_count)
    target_idx = find(id==i)';
    img = imread(sprintf([path_a 'train_images/%g.bmp'],i));

    %fig(4,4)
    for j = target_idx
        t = targets(j,:); %#ok<*NODEF>
        class = t(1);
        x1=t(2)+1;  y1=t(3)+1;  x2=t(4)+1;  y2=t(5)+1;
        class_count(class+1) = class_count(class+1) + 1;
        w = x2-x1;  h = y2-y1;
        xc = (x2 + x1)/2;  yc = (y2 + y1)/2;
        image_wh = wh(j,:);
        
        % make chip a square
        l = round((max(w,h)*1.1 + 4) * length/lengh_inner); if mod(l,2)~=0; l = l + 1; end  % normal
        x1 = max(xc-l/2,1); x2 = min(xc+l/2, image_wh(1)); 
        y1 = max(yc-l/2,1); y2 = min(yc+l/2, image_wh(2));
        img1 = img(int16(y1:y2),int16(x1:x2),:);
        img2 = imresize(img1,[length length], 'bicubic');
        
        c_count = c_count + 1;
        Y(c_count) = class;
        X(c_count,:,:,:) = img2;

        % imwrite(img2,sprintf([path_a 'classes/%g/%g.bmp'],class,class_count(class+1)));
        % sca; imshow(img2); axis equal ij; title([num2str(class) ' - ' xview_names(class)])
        
        if mod(c_count,16)==0
            ''
        end
    end
end
%rgb_mean = [60.134, 49.697, 40.746];
%rgb_std = [29.99, 24.498, 22.046];
%for i=1:3
%   X(:,:,:,i) =  (X(:,:,:,i)-rgb_mean(i)) / rgb_std(i);
%end

X = permute(X(1:c_count,:,:,:),[1 4 2 3]); %#ok<*NASGU> permute to pytorch standards
Y = Y(1:c_count);

X = permute(X,[4,3,2,1]);  % for hd5y only (reads in backwards permuted)
save('-v7.3','class_chips40+16','X','Y')

% 32 + 14 = 46
% 40 + 16 = 56
% 48 + 16 = 64
% 64 + 26 = 90
end


function [class_mu, class_sigma, class_cov] = per_class_stats(classes,w,h)
% measure the min and max bbox sizes for rejecting bad predictions
area = log(w.*h);
aspect_ratio = log(w./h);
uc = unique(classes(:)); 
n = numel(uc);

class_mu = zeros(n,4,'single'); % width, height, area, aspect_ratio
class_sigma = zeros(n,4,'single'); 
class_cov = zeros(n,4,4,'single');

for i = 1:n
    j = find(classes==uc(i));
    wj = log(w(j));  hj = log(h(j));  aj = area(j);  arj = aspect_ratio(j,:);
    data = [wj, hj, aj, arj];
    class_mu(i,:) = mean(data,1);
    class_sigma(i,:) = std(data,1);
    class_cov(i,:,:) = cov(data);
    %[~,C] = kmeans([wj hj],1,'MaxIter',5000,'OnlinePhase','on');
    
    %class_name = xview_names(xview_classes2indices(uc(i)));
    %close all; hist211(arj,hj,40); title(sprintf('%s, cor = %g',class_name,corr(wj,hj)))
    
    % close all; hist211(wj,hj,{linspace(0,max(wj),40),linspace(0,max(hj),40)}); plot(C(:,1),C(:,2),'g.','MarkerSize',50); title(sprintf('%s, cor = %g',class_name,corr(wj,hj)))
    % ha=fig; histogram(arj, linspace(-3,3,50)); title(class_name); ha.YScale='linear';
end
end


function [coords, valid] = clean_coords(coords, classes, image_h, image_w)
x1 = coords(:,1);
y1 = coords(:,2);
x2 = coords(:,3);
y2 = coords(:,4);
w = x2-x1;
h = y2-y1;
area = w.*h;

% crop
x1 = min( max(x1,0), image_w);
y1 = min( max(y1,0), image_h);
x2 = min( max(x2,0), image_w);
y2 = min( max(y2,0), image_h);
w = x2-x1;
h = y2-y1;
new_area = w.*h;

% no nans or infs in bounding boxes
i0 = ~any(isnan(coords) | isinf(coords), 2);

% sigma rejections on dimensions (entire dataset)
[~, i1] = fcnsigmarejection(area,21, 3);
[~, i2] = fcnsigmarejection(w,21, 3);
[~, i3] = fcnsigmarejection(h,21, 3);

% sigma rejections on dimensions (per class)
uc=unique(classes(:));
for i = 1:numel(uc)
    j = find(classes==uc(i));
    [~,v] = fcnsigmarejection(area(j),15,3);  i1(j) = i1(j) & v;
    [~,v] = fcnsigmarejection(w(j),15,3);     i2(j) = i2(j) & v;
    [~,v] = fcnsigmarejection(h(j),15,3);     i3(j) = i3(j) & v;
end

% manual dimension requirements
i4 = area >= 20 & w > 4 & h > 4;  

% extreme edges (i.e. don't start an x1 10 pixels from the right side)
i5 = x1 < (image_w-10) & y1 < (image_h-10) & x2 > 10 & y2 > 10;  % border = 5

% cut objects that lost >90% of their area during crop
i6 = (new_area./ area) > 0.10;

% no image dimension nans or infs, or smaller than 32 pix
hw = [image_h image_w];
i7 = ~any(isnan(hw) | isinf(hw) | hw < 32, 2);

% remove invalid classes 75 and 82 ('None' class, wtf?)
i8 = ~any(classes(:) == [75, 82],2);

% remove 18 and 73 (small cars and buildings) as an experiment
%i9 = any(classes(:) == [11, 12, 13, 15, 74, 84],2);  % group 0 aircraft
%i9 = any(classes(:) == [17, 18, 19],2);  % group 1 cars
%i9 = any(classes(:) == [71, 72, 76, 77, 79, 83, 86, 89, 93, 94],2);  % group 2 buildings
%i9 = any(classes(:) == [20, 21, 23, 24, 25, 26, 27, 28, 29, 32, 60, 91],2);  % group 3 trucks
%i9 = any(classes(:) == [33, 34, 35, 36, 37, 38],2);  % group 4 trains
%i9 = any(classes(:) == [40, 41, 42, 44, 45, 47, 49, 50, 51, 52],2);  % group 5 boats
%i9 = any(classes(:) == [53, 54, 55, 56, 57, 59, 61, 62, 63, 64, 65, 66],2);  % group 6 docks

valid = i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7 & i8;
coords = [x1(valid) y1(valid) x2(valid) y2(valid)];
end      


function C = fcn_kmeans(X, n)
rng('default'); % For reproducibility
%X = [randn(100,2)*0.75+ones(100,2);
%    randn(100,2)*0.55-ones(100,2)];

% opts = statset('Display','iter');
%[idx,C, sumd] = kmedoids(X,n,'Distance','cityblock','Options',opts);
X = [X; X(:,[2, 1])];
[idx,C, sumd] = kmeans(X,n,'MaxIter',400,'OnlinePhase','on');
%sumd


fig;
for i = 1:numel(unique(idx))
    plot(X(idx==i,1),X(idx==i,2),'.','MarkerSize',1)
end

plot(C(:,1),C(:,2),'co','MarkerSize',7,'LineWidth',1.5)
legend('Cluster 1','Cluster 2','Medoids','Location','NW');
title('Cluster Assignments and Medoids');
hold off
end


function indices = xview_classes2indices(classes)
% remap xview classes 11-94 to 0-61
indices = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 2.0, 0, 3.0, 0, 4.0, 5.0, 6.0, 7.0, 8.0, 0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0, 0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 0, 23.0, 24.0, 25.0, 0, 26.0, 27.0, 0, 28.0, 0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 0, 0, 0, 0, 46.0, 47.0, 48.0, 49.0, 0, 50.0, 51.0, 0, 52.0, 0, 0, 0, 53.0, 54.0, 0, 55.0, 0, 0, 56.0, 0, 57.0, 0, 58.0, 59.0];
indices = indices(classes);    
end   

function names = xview_names(classes)
x = {'Fixed-wing Aircraft'
'Small Aircraft'
'Cargo Plane'
'Helicopter'
'Passenger Vehicle'
'Small Car'
'Bus'
'Pickup Truck'
'Utility Truck'
'Truck'
'Cargo Truck'
'Truck w/Box'
'Truck Tractor'
'Trailer'
'Truck w/Flatbed'
'Truck w/Liquid'
'Crane Truck'
'Railway Vehicle'
'Passenger Car'
'Cargo Car'
'Flat Car'
'Tank car'
'Locomotive'
'Maritime Vessel'
'Motorboat'
'Sailboat'
'Tugboat'
'Barge'
'Fishing Vessel'
'Ferry'
'Yacht'
'Container Ship'
'Oil Tanker'
'Engineering Vehicle'
'Tower crane'
'Container Crane'
'Reach Stacker'
'Straddle Carrier'
'Mobile Crane'
'Dump Truck'
'Haul Truck'
'Scraper/Tractor'
'Front loader/Bulldozer'
'Excavator'
'Cement Mixer'
'Ground Grader'
'Hut/Tent'
'Shed'
'Building'
'Aircraft Hangar'
'Damaged Building'
'Facility'
'Construction Site'
'Vehicle Lot'
'Helipad'
'Storage Tank'
'Shipping container lot'
'Shipping Container'
'Pylon'
'Tower'};

names = x{classes+1};
end

