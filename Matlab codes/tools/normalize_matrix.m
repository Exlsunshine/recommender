function [norm] = normalize_matrix(rating_matrix)
    %total number of rows and columns
    rows = size(rating_matrix, 1);
    columns = size(rating_matrix, 2);
    
    %filter out the columns which is not rated by the user yet
    row_none_zero = zeros(rows, 1);
    for i = 1 : rows
        clust = find(rating_matrix(i,:) ~= 0);
        row_none_zero(i) = length(clust);
    end
    row_sum = sum(rating_matrix, 2);
    row_avg = row_sum ./ row_none_zero;
    
    %subtract the avg rating
    norm = rating_matrix;
    for k = 1 : rows
        clust = find(norm(k, :) ~= 0);
        norm(k,clust) = norm(k,clust) - row_avg(k);
    end
end