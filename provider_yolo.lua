--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 10:05 AM
--
require 'nn';
require 'image';
require 'xlua';
require 'math';
local class = require 'class'
require 'mattorch'

local matDataPrefix = '';

Provider = class('Provider');

function Provider:__init(trainSize, testSize)
    print '==> load dataset into trainData/testData'

    allImages = mattorch.load('matlab/results/img_0716.mat');
    allImagesWeight = mattorch.load('matlab/results/label_0716.mat');
    print '==> finish load train data, start load test data...';
    testImages = mattorch.load('matlab/results/testimg_0716.mat');
    testImagesWeight = mattorch.load('matlab/results/testlabel_0716.mat');

    allImages.trData = allImages.imagesList:t();
    allImagesWeight.trWeight = allImagesWeight.imagesLabel:transpose(4,1):transpose(2,3);
    testImages.teData = testImages.imagesList:t();
    testImagesWeight.teWeight = testImagesWeight.imagesLabel:transpose(4,1):transpose(2,3);


    print '==> finish load dataset, start clean data...'
    trainSize = allImages.trData:size(1);
    testSize = testImages.teData:size(1);

    imgWidth = 448;
    imgHeight = 448;
    s = 7;


    -- resize and clean data.

    allImages.trData = allImages.trData:reshape(trainSize,1,imgHeight,imgWidth):float();
    allImagesWeight.trWeight = allImagesWeight.trWeight:float();
    testImages.teData = testImages.teData:reshape(testSize,1,imgHeight,imgWidth):float();
    testImagesWeight.teWeight = testImagesWeight.teWeight:float();


    self.trainData = {
        data = allImages.trData,
        labels = allImagesWeight.trWeight,
        size = function() return trainSize end
    }
    self.testData = {
        data = testImages.teData,
        origData = testImages.teData:clone(),
        labels = testImagesWeight.teWeight,
        size = function() return testSize end
    }
--    allCropImages.trData[{{mask},{}}]

end

function Provider:normalize()
    local trainData = self.trainData;
    local testData = self.testData;

    collectgarbage();
    print '==> pre-processing data'

    local mean = trainData.data:select(2, 1):mean();
    local std = trainData.data:select(2, 1):std();
    trainData.data:select(2, 1):add(-mean);
    trainData.data:select(2, 1):div(std);
    trainData.mean = mean;
    trainData.std = std;

    testData.data:select(2, 1):add(-mean);
    testData.data:select(2, 1):div(std);
    testData.mean = mean;
    testData.std = std;
end




provider = Provider();
provider:normalize();
--print '==> save provider.t7'
--torch.save('provider_yolo.t7', provider);


