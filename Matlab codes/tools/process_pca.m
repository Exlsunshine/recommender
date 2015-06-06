
function  [target] = process_pca(rating_matrix, dimension)
    %process PCA to the given dimension
   [res, rec] = pcares(rating_matrix,dimension);
   
   %plot
   if dimension == 2
        target = [rec(:,1), rec(:,2)];
        scatter(target(:,1),target(:,2),'r.');
   end
   
   if dimension == 3
        target = [rec(:,1), rec(:,2), rec(:,3)];
        scatter3(target(:,1),target(:,2),rec(:,3),'b.');
   end
   
end