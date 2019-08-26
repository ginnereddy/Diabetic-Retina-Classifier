function [feature] = find_hm(path)

% Generate a list of all the files
files = dir(sprintf('%s/img*.ppm',path));
num_files = length(files);

feature = zeros([num_files 1]);


for i=1:num_files
    
    img = imread(sprintf('%s/%s',path, files(i).name));
    img=imresize(img,0.5);
    img = double(img);
    r=size(img,1);
    c=size(img,2);
    
    %Creating background mask and subtracting to remove background pixels
    bimg = imread(sprintf('%s/%s',path, files(i).name));
    bimg = imresize(bimg,0.5);
    bimg = medfilt2(bimg(:,:,1),[5 5]);%median filter to remove noise from background
    mx = max(bimg(:));
    bg = bimg<mx*0.15;
    bg_mask = zeros(size(bimg));
    bg_mask(bg) = 1;
    m = max(img(:));
    img = img-bg_mask.*m;
    nz = img<0;
    img(nz)=0;
    
    %Getting ratio of red to green channel to enhance red elements
    img_r = zeros(r,c);
    img_r(:,:) = img(:,:,1)./img(:,:,2);
    %Scaling all pixels to same range
    mn = abs(min(img_r(:)));
    mx = max(img_r(:));
    sc = (img_r+mn)*1./mx;
    na = isnan(sc);%Changing all NaN values to 0
    sc(na)=0;
    
    %Performing a close operation to blur out the details of the blood
    %vessels. This would ensure that we do not consider the blood vessels
    %during extraction.
    %The close operation is a dilation followed by an erode operation.
    sc=imclose(sc,ones(3));%%%%%%%%%%%%%%%
    
    %Using sigmoid function to focus on exudates
    processed_img = zeros(r,c);
    sigma=0.05;
    omega=max(sc(:))*0.8; %Omega and sigma chosen by trial and error to give the best possible result
    processed_img(:,:) = 1-(1./(1+exp((sc(:,:)-omega)./sigma)));
    
    %Binarize using threshold as 50% of max value
    m = max(processed_img(:));
    np = processed_img>m*0.5;
    bin_img = zeros(r,c);
    bin_img(np) = 1;

    %Labeling the connected components in the binary image. Each label or a
    %connected component is a possible candidate for a feature.
    [cc,ccno]=bwlabel(bin_img);%%%%%%%%%%%%
    t=cc;
    m = max(img_r(:));
    %Calculating the roundness of the candidate image. Candidate image is
    %each of the connected component.
    for j=0:ccno-1
        cand=cc==j;
        area = numel(find(cand==1));
        [x,y] = find(cand==1);
        Fs = area/(max(max(x)-min(x),max(y)-min(y))^2);
        %Fs holds the roundness of the candidate. For a perfect round
        %object, Fs is 0.74. I will be thresholding the value at 0.3 to
        %eliminate candidates which are not that round. This would also
        %eliminate any leftover vessels.
        if Fs<0.3
            cc(cc==j)=0;
        end
        %Eliminating candidates based on its area. If the area is small, it
        %might just be some unwanted noise or other elements. If it is too
        %big, it is definitely not a good candidate
        if area<20 || area>100
            cc(cc==j)=0;
        end
    end
    cc(cc>0)=1;
    
    %Counting the number of connected components and using it as feature
    %values.
    comp = bwconncomp(cc);%%%%%%%%%%%%%
    feature(i) = comp.NumObjects;
end

end