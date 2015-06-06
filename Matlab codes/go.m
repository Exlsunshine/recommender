function [ rating_matrix, norm, target, cluster, user_sim_matrix] = go()
    rating_matrix = load_data();
    norm = normalize_matrix(rating_matrix);
    target = process_pca(norm, 2);
    cluster = cluster_data(target , 2, true);
    user_sim_matrix = calculate_simularity(norm);
end