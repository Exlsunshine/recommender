function [ sim ] = cos_sim( v1, v2 )
   numerator  = sum (v1 .* v2 );
   denominator = sqrt(sum(v1.^2)) * sqrt(sum(v2.^2));
   sim = numerator / denominator;
end