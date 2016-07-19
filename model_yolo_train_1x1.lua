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
function Conv3(nInputPlane, nOutputPlane)
    vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1))
    vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    vgg:add(nn.LeakyReLU(0.1,true))
    return vgg
end
function Conv1(nInputPlane, nOutputPlane)
    vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 1, 1, 1, 1, 0, 0))
    vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    vgg:add(nn.LeakyReLU(0.1,true))
    return vgg
end

function Conv3WithMoreStride(nInputPlane, nOutputPlane)
    vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3, 3, 2, 2, 1, 1))
    vgg:add(nn.SpatialBatchNormalization(nOutputPlane, 1e-3))
    vgg:add(nn.LeakyReLU(0.1,true))
    return vgg
end
function ConvComplex(nInputPlane,nMidPlane, nOutputPlane)
    Conv1(nInputPlane,nMidPlane)
    Conv3(nMidPlane,nOutputPlane)
    return vgg
end



local MaxPooling = nn.SpatialMaxPooling

--input: Bx1x448x448
vgg:add(nn.SpatialConvolution(1, 64, 7, 7, 2, 2, 3, 3))
vgg:add(nn.SpatialBatchNormalization(64, 1e-3))
vgg:add(nn.LeakyReLU(0.1,true))

-- Bx64x224x224

vgg:add(MaxPooling(2, 2, 2, 2):ceil());
vgg:add(nn.SpatialConvolution(64, 192, 3, 3, 1, 1, 1, 1))
vgg:add(nn.SpatialBatchNormalization(192, 1e-3))
vgg:add(nn.LeakyReLU(0.1,true))

-- Bx192x112x112
vgg:add(MaxPooling(2, 2, 2, 2):ceil());

-- Bx192x56x56
ConvComplex(192,128,256)
ConvComplex(256,256,512)
-- Bx512x56x56

vgg:add(MaxPooling(2, 2, 2, 2):ceil());

-- Bx512x28x28
ConvComplex(512,256,512)
ConvComplex(512,256,512)
ConvComplex(512,256,512)
ConvComplex(512,256,512)

ConvComplex(512,512,1024)
--Bx1024x28x28

vgg:add(MaxPooling(2, 2, 2, 2):ceil());
-- Bx1024x14x14
ConvComplex(1024,512,1024)
ConvComplex(1024,512,1024)

-- Bx1024x14x14
Conv3(1024,1024)
Conv3WithMoreStride(1024,1024)

-- Bx1024x7x7
Conv3(1024,1024)
Conv3(1024,1024)


vgg:add(nn.View(1024*7*7))

vgg:add(nn.Linear(1024*7*7,4096))
vgg:add(nn.BatchNormalization(4096))
vgg:add(nn.LeakyReLU(0.1,true))
vgg:add(nn.Dropout(0.5))


vgg:add(nn.Linear(4096,1*1*6))


-- check that we can propagate forward without errors
-- should get 2x(7*7*6) vector.
--print(vgg);
--print(vgg:cuda():forward(torch.CudaTensor(2,1,448,448)))

return vgg