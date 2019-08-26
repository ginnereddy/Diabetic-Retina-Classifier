function [feature] = find_exudates(path)

%Generate a list of all the files
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
    %Getting the background using a threshold of 0.15 of max value
    bg = bimg<mx*0.15;
    bg_mask = zeros(size(bimg));
    bg_mask(bg) = 1;
    m = max(img(:));
    img = img-bg_mask.*m;
    nz = img<0;
    img(nz)=0;
    
    %Getting ratio of red to green channel to enhance contrast
    img_r = zeros(r,c);
    img_r(:,:) = img(:,:,1)./img(:,:,2);
    %Scaling all pixels to same range
    mn = abs(min(img_r(:)));
    mx = max(img_r(:));
    sc = (img_r+mn)*1./mx;
    na = isnan(sc);%Changing all NaN values to 0
    sc(na)=0;
    sc = medfilt2(sc,[5 5]);%median filter to remove noise
    
    %Applying a standard deviation filter. I calculates the standard
    %deviation in a default neighborhood of 3x3 and assigns it to the
    %center pixel of the neighborhood. It also implements symmetric
    %padding.
    %I found this useful because it highlights areas where there is a
    %sudden change in intensity values. This means that the region around
    %the exudates would have a higher value in the output image. On top of
    %that, the area without any possible features would be dim in the
    %output. This means we can focus on the important parts.
    sc=stdfilt(sc);
    
    %Performing dilation on binary image of background to increase the  
    %thickness of the edge and then subtracting to remove unnecessary
    %elements in the edge present in the output of the standard deviation
    %filter.
    bg_mask=imdilate(bg_mask,ones(9));
    m = max(sc(:));
    sc = sc-bg_mask.*m;
    nz = sc<0;
    sc(nz)=0;
    
    %Binarize using threshold as 10% of max value
    m = max(sc(:));
    np = sc>m*0.1;
    bin_img = zeros(r,c);
    bin_img(np) = 1;
    
    
    [cc,ccno]=bwlabel(bin_img);%%%%%%%%%%%%%%%%%%%%
    m = max(img_r(:));
    for j=0:ccno-1
        cand=cc==j;
        area = size(find(cand==1),1);
        [x,y] = find(cand==1);
        Fs = area/(max(max(x)-min(x),max(y)-min(y))^2);
        %Fs holds the roundness of the candidate. For a perfect round
        %object, Fs is 0.74. I will be thresholding the value at 0.4 to
        %eliminate candidates which are not that round. This would also
        %eliminate vessels.
        if Fs<0.4
            cc(cc==j)=0;
        end
        %Eliminating candidates based on its area. If the area is small, it
        %might just be some unwanted noise or other elements.
        if area<60
            cc(cc==j)=0;
        end
    end
    cc(cc>0)=1;
    comp = bwconncomp(cc);%%%%%%%%%%%%%%%%%
    
    %Counting the number of connected components and using it as feature values
    feature(i) = comp.NumObjects;
end

end