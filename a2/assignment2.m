close all;
%% open the image for reading
img = imread('synthetic.png');
img = double(img);
img = mean(img,3);

%% perform base-level smoothing to supress noise
imgS = img;
clear responseDoG responseLoG
k = 1.1;
sigma = 2.0;
s = k.^(1:30)*sigma;
responseLoG = zeros(size(img,1),size(img,2),length(s));

%% Filter over a set of scales
for si = 1:length(s)
    sL = s(si);
    filter = 2*ceil(3*sL)+1;
%     filter = ceil(sL);
    HL = fspecial('log',filter,sL);
    imgFiltL = conv2(imgS,HL,'same');
    %Compute the LoG
    responseLoG(:,:,si)  = (sL^2)*imgFiltL;
end

%% detect and draw interest points
fg = figure;imagesc(img);axis image;hold on;colormap gray;
drawnow;


T = 20;

figure(fg);
for co=3:size(img,2)-2
    fprintf('(row) = (%i) \n',co);
    for row=3:size(img,1)-2
        %% find peak
        f = squeeze(responseLoG(row,co,:));
        [fMax,fmaxLocs] = findpeaks(f);         %maxima
        [fMin,fminLocs] = findpeaks(-f);        %minima
        
        
        %% maxima
        for i = 1:numel(fmaxLocs)
            % threshold test
            if fMax(i) < T
                continue;
            end
            
            % check local extrama
            sc = s(fmaxLocs(i));
            [nx,ny,nz] = meshgrid(co-2:co+2,row-2:row+2,fmaxLocs(i));
            inds = sub2ind(size(responseLoG),ny,nx,nz);
            df = responseLoG(inds(13))-responseLoG(inds);
            df(5)=[];               %don't compare to itself
            if(min(df)>0)
                % plot circle
                xc = sc*sin(0:0.1:2*pi)+co;
                yc = sc*cos(0:0.1:2*pi)+row;
                plot(xc,yc,'r');
            end
        end
        
        
        %% minima
        for i = 1:numel(fminLocs)
            % threshold test
            if fMin(i) < T
                continue;
            end
            
            % check local extrama
            sc = s(fminLocs(i));
            [nx,ny,nz] = meshgrid(co-1:co+1,row-1:row+1,fminLocs(i));
            inds = sub2ind(size(responseLoG),ny,nx,nz);
            df = responseLoG(inds(5))-responseLoG(inds);
            df(5)=[];               %don't compare to itself
            if(max(df) <0)
                % plot circle
                xc = sc*sin(0:0.1:2*pi)+co;
                yc = sc*cos(0:0.1:2*pi)+row;
                plot(xc,yc,'g');
            end
        end
        
    end
end