
    %% 1. read images
    origin1 = imread('book.jpeg');
    origin2 = imread('findbook.png');
    img1=single(rgb2gray(origin1));
    img2=single(rgb2gray(origin2));

    %% 2. compute sift
    [f1,d1]=vl_sift(img1);
    [f2,d2]=vl_sift(img2);
    
    %% 3. compute the distance
    d1 = double(transpose(d1));
    d2 = double(transpose(d2));
    dist = pdist2(d1,d2,'euclidean'); % dist = 336 x 14625
    
    %% 4. get matched points that passed the threshold
    num = size(f1,2); % number of filter interest points
    from = zeros(num,2);
    to = zeros(num,2);
    T = 0.3;

    for i=1:num % number of filter interest points
        idx = find(dist(i,:) == min(dist(i,:)));
        numerator = dist(i,idx(1));
        dist(i,idx) = max(dist(i,:)); % change the minimum to maximum
        denominator = min(dist(i,:));
        if numerator/denominator <= T
            from(i,:) = [f1(1,i) f1(2,i)];
            to(i,:) = [f2(1,idx) f2(2,idx)];
        end
    end
    
    %% 5. remove the points that failed the threshold test
    from( all(~from,2), : ) = [];
    to( all(~to,2), : ) = [];
    
    %% 6. plot the matched points
    figure; ax = axes;
    showMatchedFeatures(origin1,origin2,from,to,'montage','Parent',ax);
    title(ax, 'Candidate point matches');
    legend(ax, 'Matched points 1','Matched points 2');

