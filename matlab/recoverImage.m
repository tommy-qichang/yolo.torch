function [rgbimage] =  recoverImage(image,labels,origH, origW, s, isOrig)
[h,w]=size(image);
origImage = imresize(image,[origH,origW]);

recoverLabels = labels;
%labels: 7x7x5
mask = zeros(s,s,2);
%mask(:,:,1) = repmat([0:6],7,1); %add offset x
%mask(:,:,2) = repmat([0:6]',1,7); % add offset y
%mask2 = 1-eq(recoverLabels(:,:,1:2),0);
%recoverLabels(:,:,1:2) = (recoverLabels(:,:,1:2) + mask) .*mask2;
recoverLabels(:,:,1) = recoverLabels(:,:,1) .* (origW/s);
recoverLabels(:,:,2) = recoverLabels(:,:,2) .* (origH/s);
recoverLabels(:,:,3) = (recoverLabels(:,:,3).^2) .* origW;
recoverLabels(:,:,4) = (recoverLabels(:,:,4).^2) .* origH;

recoverLabels(:,:,1:2) = recoverLabels(:,:,1:2) - (recoverLabels(:,:,3:4)./2);

if(isOrig == true)
    
    color = uint8([255 255 0]);
else
    color = uint8([255 0 255]);
end

predict = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom','CustomBorderColor',color);

%rectangle = int32([recoverLabels(5,5,1),recoverLabels(5,5,2),recoverLabels(5,5,3),recoverLabels(5,5,4),]);

labelIdx = squeeze(find(recoverLabels(:,:,6)));
if ~exist('rgbimage','var')
    rgbimage= origImage(:,:,[1 1 1]);
end
for i =1:size(labelIdx,1)
    idx = labelIdx(i);
    y = ceil(idx/s);
    x = idx -(y-1)*s;
    
    
    rectangle = int32([recoverLabels(x,y,1),recoverLabels(x,y,2),recoverLabels(x,y,3),recoverLabels(x,y,4),]);

    rgbimage = step(predict, rgbimage, rectangle);
    
    %position = [23 373;35 185;77 107];
    %box_color = {'red','green','yellow'};

    txt = mat2str((recoverLabels(x,y,5) - recoverLabels(x,y,6)),2);
    
    if(isOrig ~= true)
       txt = strcat('pred:',txt);
       rgbimage = insertText(rgbimage,[recoverLabels(x,y,1),recoverLabels(x,y,2)-20],txt,'FontSize',10,'TextColor','white');
    end
    
    
    
end

    if size(labelIdx,1)==0
        fprintf('no annotate labels');
    end

    %figure,imshow(rgbimage);
ss



end
