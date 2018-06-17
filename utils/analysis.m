clc; clear; close
load xview.mat

chip_id = zeros(numel(chips),1);
for i=1:numel(chips)
    chip_id(i) = find(strcmp(chips(i),uchips));
end

% Target box width and height
w = coords(:,3) - coords(:,1);
h = coords(:,4) - coords(:,2);

% Normalized with and height
wn = w./shapes(chip_id,1);
hn = h./shapes(chip_id,1);
i = wn ~= Inf; wn = wn(i); hn = hn(i);
[~, i] = fcnsigmarejection(wn,6,3);
[~, j] = fcnsigmarejection(hn,6,3); wn = wn(i & j); hn = hn(i & j);

% K-means normalized with and height for 9 points
C = fcn_kmeans([wn hn], 9);
i = sort(C(:,1).*C(:,2))
C = C(i,:)';

% image mean and std
i = ~all(stats==0,2);
shapes=shapes(i,:);
stats=stats(i,:);
stat_means = zeros(1,6);
for i=1:6
    stat_means(i) = mean(fcnsigmarejection(stats(:,i),6,3));
end

% output
mu = stat_means(1:3)   % dataset rgb mean
std = stat_means(4:6)  % dataset rgb std mean
anchor_boxes = C(:)  % anchor boxes




function C = fcn_kmeans(X, n)
rng('default'); % For reproducibility
%X = [randn(100,2)*0.75+ones(100,2);
%    randn(100,2)*0.55-ones(100,2)];

%opts = statset('Display','iter','MaxIter',300);
%[idx,C, sumd] = kmedoids(X,n,'Distance','cityblock','Options',opts);
[idx,C, sumd] = kmeans(X,n,'MaxIter',400,'OnlinePhase','on');


fig;
for i = 1:numel(unique(idx))
    plot(X(idx==i,1),X(idx==i,2),'.','MarkerSize',1)
end

plot(C(:,1),C(:,2),'co','MarkerSize',7,'LineWidth',1.5)
%plot(Cm(:,1),Cm(:,2),'co','MarkerSize',7,'LineWidth',1.5)
legend('Cluster 1','Cluster 2','Medoids','Location','NW');
title('Cluster Assignments and Medoids');
hold off
end