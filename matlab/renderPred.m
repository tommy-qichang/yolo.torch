idx = 41;
origH = 549;
origW = 512;
s = 1;

id = '12_0.5_IoU';
dataId = '0721_1x1';
outputset = load(strcat('pred_outputset_',id,'.mat'));
outputset = (outputset.x)';

for i=1:size(outputset,1)
    
    pred = outputset(i,:,:,:);
    pred = reshape(pred,1,1,1,6);

    output = load(strcat('pred_output_',id,'.mat'));
    output = (output.x)';
    predlabel = load(strcat('pred_label_',id,'.mat'));
    predlabel = (predlabel.x)';

    images = load(strcat('results/img_',dataId,'.mat'));
    images = images.imagesList;
    image = reshape(images(i,:),448,448)';


    labels = load(strcat('results/label_',dataId,'.mat'));
    labels = labels.imagesLabel;
    label = labels(i,:,:,:);

    targetImg = recoverImage(image,label,origH, origW, s,true);
    predImg = recoverImage(targetImg,pred,origH, origW, s,false);
    
    imwrite(predImg,strcat('predImg/',id,'/pred_',num2str(i),'.jpg'));

end
