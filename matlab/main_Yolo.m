function main_Yolo()

clear ; close all; clc
baseUrl = 'http://128.6.5.14:3300/data/tags/';
pptlist = importdata('newlist.data','\n');
pptlist_test = importdata('newlist_test.data','\n');


% imgWidth = 512;
% imgHeight = 549;
resizeW = 480;
resizeH = 480;
savePrefix = '0705';

trDataPath = strcat('results/img_' , savePrefix);
trLabelPath = strcat('results/label_' , savePrefix);
teDataPath = strcat('results/testimg_' , savePrefix);
teLabelPath = strcat('results/testlabel_' , savePrefix);


genData(pptlist,trDataPath,trLabelPath);

genData(pptlist_test,teDataPath,teLabelPath);



function genData(pptlist, dataPath,labelPath)
    for i=1:size(pptlist,1)
        imageName = pptlist(i);
        fprintf('data:start crop image: %s\n',strcat(baseUrl,'ppt_',char(imageName)));
        
        str = urlread(strcat(baseUrl,'ppt_',char(imageName)));
        
        data = JSON.parse(str);
        resources = data.resources;
        for j=1:size(resources,2)
            ceil = cell2mat(resources(j));
            url = ceil.url;
            publicId = ceil.public_id;
            %     fprintf('===url:%s\n',url);
            tags = ceil.tags;
            loc = '';
            for k = 1:size(tags,2)
                if strncmpi(tags(k),'stroke_loc_',11),
                    strokeArea = char(tags(k));
                    strokeArea = strokeArea(12:end);
                    loc = strsplit(strokeArea,'%2C');
                end
            end
            try
                img = imread(url);
            catch
                print('error reading url:'+url);
                continue;
            end
            
    %         Save images into folders
%             imwrite(img,strcat(trDataPath,'/',publicId,'.jpg'));

    %         Save annotation information into folders.
            if  ~isempty(loc)
                [imgH,imgW,channel]=size(img);
                xmin = round(str2num(char(loc(1))));
                ymin = round(str2num(char(loc(2))));
                width = str2num(char(loc(3)));
                height = str2num(char(loc(4)));
                xmax = round(xmin + width-1);
                ymax = round(ymin + height-1);
                x = (xmin+xmax)/2;
                y = (ymin+ymax)/2;

                relativeX = x/imgW;
                relativeY = y/imgH;
                relativeW = width/imgW;
                relativeH = height/imgH;

%                 orig_file = fopen(strcat(trLabelPath,'/orig_',publicId,'.txt'),'w');
%                 fprintf(orig_file,'1 %1.5f %1.5f %1.5f %1.5f',xmin,ymin,xmax,ymax);
%                 fclose(orig_file);
% 
%                 file = fopen(strcat(trLabelPath,'/',publicId,'.txt'),'w');
%                 fprintf(file,'1 %1.5f %1.5f %1.5f %1.5f',relativeX,relativeY,relativeW,relativeH);
%                 fclose(file);
                
                
                labelLine = zeros(1,5);
                labelLine(1) = 1;
                labelLine(2) = relativeX;
                labelLine(3) = relativeY;
                labelLine(4) = relativeW;
                labelLine(5) = relativeH;
                if exist('imagesLabel','var')
                    imagesLabel = [imagesLabel;labelLine];
                    
                else
                    imagesLabel = labelLine;
                end
            
            else
                
                if exist('imagesLabel','var')
                    imagesLabel = [imagesLabel;zeros(1,5)];
                    
                else
                    imagesLabel = zeros(1,5);
                end
                
                
                
            end
            
            
            
            % transfer image to gray scale and resize image to 480*480.
            grayImg = rgb2gray(imresize(img,[resizeH,resizeW]))';
            
            
            if exist('imagesList','var')
        
                imagesList = [imagesList;reshape(grayImg,1,resizeH*resizeW)];
            else

                imagesList = reshape(grayImg,1,resizeH*resizeW);
                %reshape(imagesWeightList,1,imgW*imgH);
            end


        end
        
        


    end

    
    save(strcat(dataPath,'.mat'),'imagesList');
    save(strcat(labelPath,'.mat'),'imagesLabel');
    
    
end

end

