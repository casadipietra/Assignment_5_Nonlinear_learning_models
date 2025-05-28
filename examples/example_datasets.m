%
% This file contains examples on how to use the dataset required for
% this assignment.
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Linear dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = ml_generate_linear_data(2, 1000);
d_1 = X(X(:,1) == 1, 2:end); 
d_2 = X(X(:,1) == -1, 2:end);

figure
scatter(d_1(:,1), d_1(:,2), 'r');
hold on
scatter(d_2(:,1), d_2(:,2), 'b');
grid on
title('Linearly separable Dataset')
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CheckerBoard dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[X, labels] = ml_checkerboard_data(4000);
X1 = X(labels == 1,:);
X2 = X(labels == -1,:);

figure
hold on
scatter(X1(:,1), X1(:,2), 'filled', 'MarkerFaceColor', 'b')
scatter(X2(:,1), X2(:,2), 'filled', 'MarkerFaceColor', 'r')
grid on
title('Checkboard Dataset')
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% "Ripley" dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 1000;
load synth.tr
synth   = synth(randperm(size(synth,1)),:);
X   = synth(1:N,1:2);
t   = synth(1:N,3);
 
COL_data1   = 'k';
COL_data2   = 0.75*[0 1 0];
COL_boundary50  = 'r';
COL_boundary75  = 0.5*ones(1,3);
COL_rv      = 'r';

%
% Plot the training data
% 
figure
whitebg(1,'w')
clf
h_c1 = plot(X(t==0,1),X(t==0,2),'.','MarkerSize',18,'Color',COL_data1);
hold on
h_c2 = plot(X(t==1,1),X(t==1,2),'.','MarkerSize',18,'Color',COL_data2);
box = 1.1*[min(X(:,1)) max(X(:,1)) min(X(:,2)) max(X(:,2))];
axis(box)
set(gca,'FontSize',12)
grid on
title('Ripley Dataset')
