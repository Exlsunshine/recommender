function [ result ] = cluster_data(data , cluster_num, plot_flag)
    %get data dimension
    data_dimension = size(data, 2);
    
    %process clustering
    clusters = kmeans(data, cluster_num);
    %attach cluster result to original data
    result = [data, clusters];
    
    for i = 1 : cluster_num
        clust = find(clusters == i);
        display([num2str(i), 'cluster contains: ', num2str(length(clust))]);
    end
    
    %plot
    if plot_flag == true
        ptsymb = {'b*','m^','rd','co','g+','ys','wp','kx'};
        for i = 1 : cluster_num
            clust = find(result(:,size(result,2)) == i);

            if data_dimension == 3
                plot3(result(clust,1),result(clust,2),result(clust,3),ptsymb{i});
            end
            if data_dimension == 2
                plot(result(clust,1),result(clust,2),ptsymb{i});
            end
            hold on
        end
        hold off
        if data_dimension == 3
            view(-40,5);
        end
        grid on
    end
end

