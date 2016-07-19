--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 1:03 PM
-- To change this template use File | Settings | File Templates.
--
require 'nn'
require 'cunn'

local vgg = nn.Sequential();

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane)
    vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    vgg:add(nn.ReLU(true))
    return vgg
end

local MaxPooling = nn.SpatialMaxPooling

--input: Bx1x448x448
ConvBNReLU(1,64):add(nn.Dropout(0.3));
ConvBNReLU(64,64);
-- Bx64x448x448
vgg:add(MaxPooling(2, 2, 2, 2):ceil());

-- Bx64x224x224
ConvBNReLU(64, 128):add(nn.Dropout(0.4))
ConvBNReLU(128, 128)
vgg:add(MaxPooling(2, 2, 2, 2):ceil())

-- Bx128x112x112
ConvBNReLU(128, 256):add(nn.Dropout(0.4))
ConvBNReLU(256, 256):add(nn.Dropout(0.4))
ConvBNReLU(256, 256)
vgg:add(MaxPooling(2, 2, 2, 2):ceil())

-- Bx256x56x56
ConvBNReLU(256, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
vgg:add(MaxPooling(2, 2, 2, 2):ceil())

-- Bx512x28x28
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
ConvBNReLU(512, 512)
vgg:add(MaxPooling(2, 2, 2, 2):ceil())

-- Bx512x14x14
ConvBNReLU(512, 512):add(nn.Dropout(0.4))
--ConvBNReLU(1024, 1024):add(nn.Dropout(0.4))
--ConvBNReLU(1024, 1024)
vgg:add(MaxPooling(2, 2, 2, 2):ceil())
-- Bx512x7x7


vgg:add(nn.View(512*7*7))

classifier = nn.Sequential()
classifier:add(nn.Linear(512*7*7, 4096))
classifier:add(nn.BatchNormalization(4096))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 7*7*(6)))
vgg:add(classifier)


-- check that we can propagate forward without errors
-- should get 2x(7*7*6) vector.
--print(vgg);
print(vgg:cuda():forward(torch.CudaTensor(2,1,448,448)))

return vgg