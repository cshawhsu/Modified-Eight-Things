%create dendrogram from correlation matrix 
%can use both split half and non split half

%for split half, will average the upper and lower half of matrices
%and set diagonal to 0...

ROI = 'bl VTC DvS facebodyscene';

%[~,mat_corr] = mv_corrcoef([], 1);   %non-split half
[~,mat_corr] = mv_splitHalf([], 1); %split half

%determine if matrix is symmetric (split half or not)
corr_matrix = mat_corr;
if ~issymmetric(corr_matrix) %if not symmetric, average upper and lower half
    corr_matrix_sym = (corr_matrix + transpose(corr_matrix))/2;
    
    corr_matrix_sym_plot = corr_matrix_sym;
    
    %set diagonal to 1
    for i = 1:12
        corr_matrix_sym(i,i) = 1;
    end
    
    corr_method = 'Split Half';
else
    corr_matrix_sym = corr_matrix;
    corr_matrix_sym_plot = corr_matrix_sym;
    corr_method = 'All Runs';
end

plot_title = strcat('Figures/',ROI,'-',corr_method,'.png');

corr_matrix_inv = 1 - corr_matrix_sym;

%find clusters
Z = linkage(squareform(corr_matrix_inv),'average');

inplot_title = strcat(ROI,'-',corr_method);

figure;
subplot(1,2,1)
dendrogram(Z,'Labels',{'D Faces Emo','D Faces Comm','D Bodies Emo','D Hand Comm','D Vehicles','D Scenes','S Faces Emo','S Faces Comm','S Bodies Emo','S Hand Comm','S Vehicles','S Scenes'});
xtickangle(45);
ylabel('mean cluster distance (1 - correlation)');
title(inplot_title);

subplot(1,2,2)
imagesc(corr_matrix_sym_plot);
c_bar = colorbar;
colormap(redblue);
c_bar.Label.String = 'correlation';
caxis([-1 1]);
xticks([1:12]);
xticklabels({'D Faces Emo','D Faces Comm','D Bodies Emo','D Hand Comm','D Vehicles','D Scenes','S Faces Emo','S Faces Comm','S Bodies Emo','S Hand Comm','S Vehicles','S Scenes'});
xtickangle(45);

yticks([1:12]);
yticklabels({'D Faces Emo','D Faces Comm','D Bodies Emo','D Hand Comm','D Vehicles','D Scenes','S Faces Emo','S Faces Comm','S Bodies Emo','S Hand Comm','S Vehicles','S Scenes'});
title(inplot_title);

x0=10;
y0=10;
width=1200;
height=600;
set(gcf,'units','points','position',[x0,y0,width,height])

saveas(gcf,plot_title);
