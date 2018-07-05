clc; clear; close
load json_data.mat

chip_id = zeros(numel(chips),1);
chip_number = zeros(numel(chips),1);
for i=1:numel(chips)
    chip_id(i) = find(strcmp(chips(i),uchips));
    
    s = chips{i};
    s = s(1:end-4);
    chip_number(i) = str2double(s);
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
fprintf('Images target count range: %g-%g\n',min(n),max(n))
% fig; histogram(n,linspace(0,300,301))
% sortrows([n, uchip_number],-1)

% Target box width and height
w = coords(:,3) - coords(:,1);
h = coords(:,4) - coords(:,2);

% to reject bad box predictions
class_stats = per_class_stats(classes,w,h);
[~,~,~,n] = fcnunique(classes(:));
vpa(n(:)')

%a=class_stats(:,[7 9]); 
%[~,i]=sort(prod(a,2));  a=a(i,:);
%vpa(a(:)',4)

% K-means normalized with and height for 9 points
C = fcn_kmeans([w h], 3);
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
id = single(chip_number);
save('targets_60c.mat','wh','targets','id','class_stats')


function stats = per_class_stats(classes,w,h)
% measure the min and max bbox sizes for rejecting bad predictions
area = h.*w;
uc=unique(classes(:)); 
n = numel(uc);
limits = zeros(n,6); % minmax: [width, height, area]
wh = zeros(n,3); % [class, wh1, wh2, wh3]
for i = 1:n
    j = find(classes==uc(i));
    wj = w(j);  hj = h(j);
    limits(i,:) = [minmax3(wj), minmax3(hj), minmax3(area(j))];
    
    %close all; hist211(wj,hj,{linspace(0,max(wj),40),linspace(0,max(hj),40)}); 
    %title(corr(wj,hj))
    [~,C] = kmeans([wj hj],1,'MaxIter',500,'OnlinePhase','on');
    %plot(C(:,1),C(:,2),'g.','MarkerSize',50)
    wh(i,:) = [i-1, C(1,:)];
end
stats = [limits, wh];
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
[~, i1] = fcnsigmarejection(area,18, 3);
[~, i2] = fcnsigmarejection(w,18, 3);
[~, i3] = fcnsigmarejection(h,18, 3);

% sigma rejections on dimensions (per class)
uc=unique(classes(:));
for i = 1:numel(uc)
    j = find(classes==uc(i));
    [~,v] = fcnsigmarejection(area(j),9,3);  i1(j) = i1(j) & v;
    [~,v] = fcnsigmarejection(w(j),9,3);     i2(j) = i2(j) & v;
    [~,v] = fcnsigmarejection(h(j),9,3);     i3(j) = i3(j) & v;
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
i9 = ~any(classes(:) == [18, 73],2);

valid = i0 & i1 & i2 & i3 & i4 & i5 & i6 & i7 & i8;
coords = [x1(valid) y1(valid) x2(valid) y2(valid)];
end



function C = fcn_kmeans(X, n)
rng('default'); % For reproducibility
%X = [randn(100,2)*0.75+ones(100,2);
%    randn(100,2)*0.55-ones(100,2)];

opts = statset('Display','iter');
%[idx,C, sumd] = kmedoids(X,n,'Distance','cityblock','Options',opts);
[idx,C, sumd] = kmeans(X,n,'MaxIter',400,'OnlinePhase','on','Options',opts);
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