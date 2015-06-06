function [ sim ] = pearson_sim( v1, v2 )
    avg1 = sum(v1) / length(v1);
    avg2 = sum(v2) / length(v2);
    
    numerator = sum((v1 - avg1) .* (v2 - avg2));
    denominator = sqrt(sum((v1 - avg1).^2) * sum((v2 - avg2).^2));
    
    sim = numerator / denominator;
end