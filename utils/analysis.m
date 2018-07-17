% 659.bmp and 769.bmp are bad pictures, delete

clc; clear; close
load json_data.mat

make_small_chips()

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
sortrows([n, uchip_number],-1)

% Target box width and height
w = coords(:,3) - coords(:,1);
h = coords(:,4) - coords(:,2);

% stats for outlier bbox rejection
class_stats = per_class_stats(classes,w,h);
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


function [] = make_small_chips()
clc; close; clear
load('targets_60c.mat')
path_a = '/Users/glennjocher/downloads/DATA/xview/';

rmdir([path_a 'classes'],'s')
for i=0:59
    mkdir(sprintf([path_a 'classes/%g'],i))
end

uid = unique(id)';
class_count = zeros(1,60);
f_count = 0;  % file count
c_count = 0;  % chip count
X = zeros(650000,32,32,3, 'uint8');
Y = zeros(1,650000,'uint8');
for i = uid
    f_count = f_count+1;
    fprintf('%g/847\n',f_count)
    target_idx = find(id==i)';
    img = imread(sprintf([path_a 'train_images/%g.bmp'],i));
    %fig; imshow(img)

    %fig(4,4)
    for j = target_idx
        c_count = c_count + 1;
        t = targets(j,:); %#ok<*NODEF>
        class = xview_classes2indices(t(1));
        x1=t(2)+1;  y1=t(3)+1;  x2=t(4)+1;  y2=t(5)+1;
        class_count(class+1) = class_count(class+1) + 1;
        w = x2-x1;  h = y2-y1;
        xc = (x2 + x1)/2;  yc = (y2 + y1)/2;
        image_wh = wh(j,:);
        
        % make chip a square
        l = round(max(w,h)*1.1 + 2); if mod(l,2)~=0; l = l + 1; end
        x1 = max(xc-l/2,1); x2 = min(xc+l/2, image_wh(1)); 
        y1 = max(yc-l/2,1); y2 = min(yc+l/2, image_wh(2));
        img1 = img(int16(y1:y2),int16(x1:x2),:);
        img2 = imresize(img1,[32 32], 'bicubic');
        
        Y(c_count) = class;
        X(c_count,:,:,:) = img2;

        % imwrite(img2,sprintf([path_a 'classes/%g/%g.bmp'],class,class_count(class+1)));
        % sca; imshow(img2); axis equal ij; title(class)
    end
end
X=permute(X(1:c_count,:,:,:),[1 4 2 3]); %#ok<*NASGU> permute to pytorch standards
Y=Y(1:c_count);

save('-v6','class_net_data','X','Y')
end


function stats = per_class_stats(classes,w,h)
% measure the min and max bbox sizes for rejecting bad predictions
area = log(h.*w);
uc=unique(classes(:)); 
n = numel(uc);
limits = zeros(n,6); % minmax: [width, height, area]
wh = zeros(n,3); % [class, wh1, wh2, wh3]
for i = 1:n
    j = find(classes==uc(i));
    wj = log(w(j));  hj = log(h(j));  aj = area(j);
    % limits(i,:) = [minmax3(wj), minmax3(hj), minmax3(area(j))];
    limits(i,:) = [mean(wj), std(wj), mean(hj), std(hj), mean(aj), std(aj)];
    [~,C] = kmeans([wj hj],1,'MaxIter',5000,'OnlinePhase','on');
    wh(i,:) = [i-1, exp(C(1,:))];

    % close all; hist211(wj,hj,{linspace(0,max(wj),40),linspace(0,max(hj),40)}); 
    % plot(C(:,1),C(:,2),'g.','MarkerSize',50);     title(corr(wj,hj))
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


function indices = xview_classes2indices(classes)
% remap xview classes 11-94 to 0-61
indices = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 2.0, 0, 3.0, 0, 4.0, 5.0, 6.0, 7.0, 8.0, 0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 0, 0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 0, 23.0, 24.0, 25.0, 0, 26.0, 27.0, 0, 28.0, 0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 0, 0, 0, 0, 46.0, 47.0, 48.0, 49.0, 0, 50.0, 51.0, 0, 52.0, 0, 0, 0, 53.0, 54.0, 0, 55.0, 0, 0, 56.0, 0, 57.0, 0, 58.0, 59.0];
indices = indices(classes);    
end     