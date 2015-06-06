function [ user_sim_matrix ] = calculate_simularity( rating_matrix )
    rows = size(rating_matrix, 1);
    columns = size(rating_matrix, 2);
    
    user_sim_matrix = zeros(rows, rows);
    for i = 1 : rows
        for j = i + 1 : rows
            cluster_i = find(rating_matrix(i,:) ~= 0);
            cluster_j = find(rating_matrix(j,:) ~= 0);
            
            if length(cluster_i) == 0 || length(cluster_j) == 0 || length(intersect(cluster_i, cluster_j)) == 0
                user_sim_matrix(i,j) = 0;
                user_sim_matrix(j,i) = 0;
            else
                common_item = intersect(cluster_i, cluster_j);

                rating_i = rating_matrix(i,:);
                rating_j = rating_matrix(j,:);

                user_sim_matrix(i,j) = 0.5 * (cos_sim(rating_i(common_item), rating_j(common_item)) + (length(common_item) / length(union(cluster_i,cluster_j))));
                user_sim_matrix(j,i) = user_sim_matrix(i,j);
            end
        end
        disp([num2str(i / rows * 100) , '%'])
    end
    
%     item_sim_matrix = zeros(columns, columns);
%     for i = 1 : columns
%         for j = 1 : columns
%             cluster_i = find(rating_matrix(:,i) ~= 0);
%             cluster_j = find(rating_matrix(:,j) ~= 0);
%             
%             common_item = intersect(cluster_i, cluster_j);
%             
%             rating_i = rating_matrix(:,i);
%             rating_j = rating_matrix(:,j);
%             
%             item_sim_matrix(i,j) = cos_sim(rating_i(common_item), rating_j(common_item));
%         end
%     end
end