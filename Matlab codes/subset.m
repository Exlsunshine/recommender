function [ rate ] = subset()
    rating_matrix = load_data();
    
%     o = rating_matrix;
%     for i = 1 : size(rating_matrix, 1)
%         r = o(i,:);
%         r(find(r ~= 0)) = 1;
%         o(i,:) = r;
%     end
    
%     o = sum(o, 1);
%     for i = 1 : size(rating_matrix, 2)
%         disp([num2str(i),' ',num2str(o(1,i))])
%     end
    
%     rated1 = select_column(rating_matrix, 101);
%     rated2 = select_column(rating_matrix, 259);
%     rated3 = select_column(rating_matrix, 51);
%     rated4 = select_column(rating_matrix, 287);
%     rated5 = select_column(rating_matrix, 295);
%     rated6 = select_column(rating_matrix, 99);
%     rated7 = select_column(rating_matrix, 289);
%     rated8 = select_column(rating_matrix, 8);
%     rated9 = select_column(rating_matrix, 182);
%     rated10 = select_column(rating_matrix, 57);
%     
%     disp(rated10)
    %disp(length(intersect(rated4, rated3)))
    
%     disp(rating_matrix(99, 101));
%     disp(rating_matrix(99, 259));
%     disp(rating_matrix(99, 51));
%     disp(rating_matrix(99, 287));
%     disp(rating_matrix(99, 295));
%     disp(rating_matrix(99, 99));
%     disp(rating_matrix(99, 289));
%     disp(rating_matrix(99, 8));
%     disp(rating_matrix(99, 182));
%     disp(rating_matrix(99, 57));


    users = [99,6,8,13,14,21,28,43,56,59];
   
    rate = zeros(length(users),10);
    for i = 1 : length(users)
        rate(i,:) = get_rate(rating_matrix, users(i));
    end
    
    disp(rate)
    %get_rate(rating_matrix, 99);
end

function [rates] = get_rate(rating_matrix, user)
    items = [101,259,51,287,295,99,289,8,182,57];
    
    rates = zeros(1, length(items));
    for i = 1 : length(items)
        rates(1, i) = rating_matrix(user, items(i));
    end
    
    %size(rates)
    %disp(rates)
end


function [ rated ] = select_column(rating_matrix, columns)
    data = rating_matrix(:,columns);
    rated = find(data ~= 0);
end