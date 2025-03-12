clc;clear;close all;

%% load data
b_1 = abs(readtable("Predicted_parameters/b_hat_e2e_1.csv"));
b_2 = abs(readtable('Predicted_parameters/b_hat_e2e_2.csv'));
b_3 = abs(readtable('Predicted_parameters/b_hat_e2e_3.csv'));
b_5 = abs(readtable('Predicted_parameters/b_hat_e2e_5.csv'));
b_6 = abs(readtable('Predicted_parameters/b_hat_e2e_6.csv'));

%% Plot
colors = ["#2E86C1","#F39C12","#2ECC71"];
icons = ["o","square",'diamond'];
fontsize=12;
font = 'Arial';


f = figure();
f.Units = "inches";
f.Position = [1 1 12 5.5];

tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% First subplot
nexttile
s1 = scatter3(b_1(1:92,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{1},"MarkerEdgeColor","k");
hold on
s2 = scatter3(b_1(93:142,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{2},"MarkerEdgeColor","k");
s3 = scatter3(b_1(143:end,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{3},"MarkerEdgeColor","k");
s1.Marker = "o";
s2.Marker = "square";
s3.Marker = "diamond";


% Second subplot
nexttile
s1 = scatter3(b_2(1:92,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{1},"MarkerEdgeColor","k");
hold on
s2 = scatter3(b_2(93:142,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{2},"MarkerEdgeColor","k");
s3 = scatter3(b_2(143:end,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{3},"MarkerEdgeColor","k");
s1.Marker = "o";
s2.Marker = "square";
s3.Marker = "diamond";

% Third subplot
nexttile
s1 = scatter3(b_3(1:92,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{1},"MarkerEdgeColor","k");
hold on
s2 = scatter3(b_3(93:142,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{2},"MarkerEdgeColor","k");
s3 = scatter3(b_3(143:end,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{3},"MarkerEdgeColor","k");
s1.Marker = "o";
s2.Marker = "square";
s3.Marker = "diamond";

% Fourth subplot
nexttile
s1 = scatter3(b_5(1:92,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{1},"MarkerEdgeColor","k");
hold on
s2 = scatter3(b_5(93:142,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{2},"MarkerEdgeColor","k");
s3 = scatter3(b_5(143:end,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{3},"MarkerEdgeColor","k");
s1.Marker = "o";
s2.Marker = "square";
s3.Marker = "diamond";
% Fifth subplot
nexttile
h1 = scatter3(b_6(1:92,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{1},"MarkerEdgeColor","k");
hold on
h2 = scatter3(b_6(93:142,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{2},"MarkerEdgeColor","k");
h3 = scatter3(b_6(143:end,:),"Var1","Var2","Var3","filled","MarkerFaceColor",colors{3},"MarkerEdgeColor","k");
h1.Marker = "o";
h2.Marker = "square";
h3.Marker = "diamond";
%
% Set common limits and labels
title_ = {'$w_1-w_0$','$w_2-w_0$','$w_3-w_0$','$w_5-w_0$','$w_6-w_0$'};

for iii=1:5
    nexttile(iii)
    xlim([0,0.02])
    ylim([0,4000])
    zlim([0,1000])
    zticks([0,500,1000])
    set(gca,"FontName",font)
    title(title_{iii},'Interpreter','latex','FontSize',fontsize+5)

    xlabel("$b_{1}$",'Interpreter','latex','FontSize',fontsize+5)
    ylabel("$b_{2}$",'Interpreter','latex','FontSize',fontsize+5)
    zlabel("$b_{3}$",'Interpreter','latex','FontSize',fontsize+5)
end

% Sixth tile for the legend
% nexttile
% axis off  % Turn off the axis for the legend tile
% lgd = legend([h1,h2,h3],{'Training','High-DoD Test','Low-DoD Test'},'FontSize',fontsize,'Location','northeastoutside','FontName',font);
% lgd.Layout.Tile = 'east';  % Assign the legend to the last tile
% % 
% % % Add legend manually with external positioning
lgd = legend('Training','High-DoD Test','Low-DoD Test','Location','northeastoutside','FontSize',fontsize);
lgd.Position = [0.75,0.2,0.16,0.16];  % Adjust position manually
% % % lgd = legend;
% lgd.Layout.Tile = 'east';