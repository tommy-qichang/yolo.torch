function [recoverLabels] =  recoverImage(image,labels,origH, origW, s)
[h,w]=size(image);
origImage = imresize(image,[origH,origW]);

recoverLabels = labels;
%labels: 7x7x5
mask = zeros(s,s,2);
mask(:,:,1) = repmat([0:6],7,1); %add offset x
mask(:,:,2) = repmat([0:6]',1,7); % add offset y
mask2 = 1-eq(recoverLabels(:,:,1:2),0);
recoverLabels(:,:,1:2) = (recoverLabels(:,:,1:2) + mask) .*mask2;
recoverLabels(:,:,1) = recoverLabels(:,:,1) .* (origW/s);
recoverLabels(:,:,2) = recoverLabels(:,:,2) .* (origH/s);
recoverLabels(:,:,3) = (recoverLabels(:,:,3).^2) .* origW;
recoverLabels(:,:,4) = (recoverLabels(:,:,4).^2) .* origH;


yellow = uint8([255 255 0]);
shapeInserter = vision.ShapeInserter('Shape','Rectangles','BorderColor','Custom','CustomBorderColor',yellow);
rectangle = int32([recoverLabels(5,5,1),recoverLabels(5,5,2),recoverLabels(5,5,3),recoverLabels(5,5,4),]);
rgbimage= image2(:,:,[1 1 1]);
J = step(shapeInserter, rgbimage, rectangle);


end
