--
--
-- User: changqi
-- Date: 3/14/16
-- Time: 12:25 PM
-- To change this template use File | Settings | File Templates.
require 'nn';
require 'optim'
require 'cunn'
require 'cudnn'
require 'image';
require 'xlua';
require './yoloCriterion/RegionProposalCriterion.lua'
dofile './provider_yolo_1x1.lua'
local class = require 'class'
local c = require 'trepl.colorize'

opt = lapp [[
   -s,--save                  (default "logs")      subdirectory to save logs
   -b,--batchSize             (default 6)          batch size
   -r,--learningRate          (default 1e-3)        learning rate
   --learningRateDecay        (default 1e-7)      learning rate decay
   --weightDecay              (default 0.0005)      weightDecay
   -m,--momentum              (default 0.9)         momentum
   --epoch_step               (default 25)          epoch step
   --model                    (default model_yolo_train)     model name
   --max_epoch                (default 300)           maximum number of iterations
   --backend                  (default nn)            backend
   -i,--log_interval          (default 5.0)           show log interval
   --modelPath                (default logs/model_fcnn.net) exist model
   --trainSize                (default 0) set training and test data size;0->unlimit
   --testSize                (default 0) set training and test data size;0->unlimit
   --criWeight              (default 4) set criterion's weights for region proposal
   --bWeight                (default 0.25) set criterion's weights for balance classification
]]

print(opt)
------------------------------------ loading data----------------------------------------
print(c.blue '==>' .. ' loading data')
--provider = torch.load('provider_yolo.t7')
provider.trainData.data = provider.trainData.data:float()
provider.testData.data = provider.testData.data:float()


function _narrow(trSize,teSize)
    if trSize ~=0 then

        provider.trainData.data = provider.trainData.data:narrow(1,1,trSize);
        provider.trainData.labels = provider.trainData.labels:narrow(1,1,trSize);
    end
    if teSize ~=0 then
        provider.testData.data = provider.testData.data:narrow(1,1,teSize);
        provider.testData.labels = provider.testData.labels:narrow(1,1,teSize);
    end
end

--if opt.trainSize ~=0 then
--    _narrow(opt.trainSize,opt.testSize);
--end


do -- data augmentation module
local BatchFlip, parent = torch.class('nn.BatchFlip', 'nn.Module')

function BatchFlip:__init()
    parent.__init(self)
    self.train = true
end

function BatchFlip:updateOutput(input)
    if self.train then
        local bs = input:size(1)
        local flip_mask = torch.randperm(bs):le(bs / 2)
        for i = 1, input:size(1) do
            if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
        end
    end
    self.output:set(input)
    return self.output
end
end


------------------------------------ configuring----------------------------------------

print(c.red '==>' .. 'configuring model')
local modelPath = opt.modelPath;

local model = nn.Sequential();
--model:add(nn.BatchFlip():float())

model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));

--if modelPath and paths.filep(modelPath) then
--    model:add(torch.load(modelPath));
--    print('==> load exist model:' .. modelPath);
--else
    model:add(dofile(opt.model .. '.lua'):cuda());
--end

--model:get(1).updateGradInput = function(input) return end

----------------------------------- load exist model -------------------------------------





if opt.backend == 'cudnn' then
    require 'cudnn'
    cudnn.convert(model:get(2), cudnn)
end

print(model);


parameters, gradParameters = model:getParameters()

------------------------------------ save log----------------------------------------
print('Will save at ' .. opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames { 'type 2 error', 'type 1 error' }
testLogger.showPlot = false

------------------------------------ set criterion---------------------------------------
print(c.blue '==>' .. ' setting criterion')
--train set:351641088 positive:2344174


criterion = nn.RegionProposalCriterion(opt.criWeight,opt.bWeight,1,true):cuda();
--criterion = cudnn.SpatialCrossEntropyCriterion(torch.Tensor{1.006,150}):cuda();
--criterion = nn.CrossEntropyCriterion(torch.Tensor({1.006,150})):cuda();

confusion = optim.ConfusionMatrix(2);

------------------------------------ optimizer config-------------------------------------
print(c.blue '==>' .. ' configuring optimizer')
optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay,
}

function train()
    model:training();
    epoch = epoch or 1;

    -- drop learning rate every "epoch_step" epochs  ?
    if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate / 2 end

    -- update negative set every 6 epochs.

    print(c.blue '==>' .. " online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')


    local targets = torch.CudaTensor(opt.batchSize,provider.trainData.labels:size(2),provider.trainData.labels:size(3),provider.trainData.labels:size(4));
    -- random index and split all index into batches.
    local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize);
    indices[#indices] = nil;


    local tic = torch.tic();
    for t, v in ipairs(indices) do
        xlua.progress(t, #indices)
        local innerTic = torch.tic();
        local inputs = provider.trainData.data:index(1, v);
        targets:copy(provider.trainData.labels:index(1, v));


        local feval = function(x)
            if x ~= parameters then parameters:copy(x) end
            gradParameters:zero();

--            require('mobdebug').start(nill,8222);
            outputs = model:cuda():forward(inputs)

--            flatOutput = _flatTensor(outputs);
--            flatTargets = targets:reshape(targets:nElement());

            f = criterion:forward(outputs:cuda(), targets:cuda())

            --outputs: Bx2xHxW  target: Bx1xHxW
            df = criterion:backward(outputs:cuda(), targets:cuda());

--            local result = outputs:select(2,2):csub(outputs:select(2,1));
--            result[result:gt(0)]=2;
--            result[result:le(0)]=1;

--            require('mobdebug').start(nill,8222);
--            local final = torch.eq(result,targets);
--            local accuracy = final:sum()/targets:nElement();

            model:backward(inputs:cuda(), df:cuda());

            --outputs: Bx2xHxW  target: BxHxW

            local outputsClasses = outputs:reshape(outputs:size(1),1,1,6):narrow(4,5,2);
            local targetClasses = targets:reshape(targets:size(1),1,1,6):narrow(4,6,1);
            confusion:batchAdd(outputsClasses:reshape(outputs:size(1)*1*1,2), targetClasses:reshape(targets:size(1)*1*1,1)+1);
--            print('losts: '..f..' and accuracy: '..accuracy..'\n');

            return f, gradParameters;
        end

        local x, fx = optim.sgd(feval, parameters, optimState);


        local innerToc = torch.toc(innerTic);
        local function printInfo()
            local tmpl = '---------%d/%d (epoch %.3f), ' ..
                    'train_loss = %6.8f, grad/param norm = %6.4e, ' ..
                    'speed = %5.1f/s, %5.3fs/iter -----------'
            print(string.format(tmpl,
                t, #indices, epoch,
                fx[1], gradParameters:norm() / parameters:norm(),
                opt.batchSize / innerToc, innerToc))
        end

        if t % opt.log_interval == 0 then
            printInfo();
        end
    end

    confusion:updateValids();
--    print(c.red('Train accuracy: ' .. c.cyan '%.2f' .. ' %%\t time: %.2f s'):format(confusion.totalValid * 100, torch.toc(tic)));


    print('Train accuracy:', confusion.totalValid * 100)
    print(confusion)

    confusion:zero()
    epoch = epoch + 1;
end

function _flatTensor(tensor)
    -- from Bx2xHxW to B*H*W x 2
    local subset1 = tensor:select(2,1);
    local outputs1 = subset1:reshape(subset1:nElement());
    local subset2 = tensor:select(2,2);
    local outputs2 = subset2:reshape(subset2:nElement());
    return torch.cat(outputs1,outputs2,2);
end

function _unflatTensor(tensor,origSize)
    local subset1 = tensor:select(2,1);
    local outputs1 = subset1:reshape(origSize);
    local subset2 = tensor:select(2,1);
    local outputs2 = subset2:reshape(origSize);
    return torch.cat(outputs1,outputs2,2);
end



function test()
    -- disable flips, dropouts and batch normalization
    model:evaluate()
    print(c.blue '==>' .. " testing")
    local bs = opt.batchSize;
    len = provider.testData.data:size(1);
    for i = 1, len, bs do
        xlua.progress(i, len)
        if (i + bs) > len then idxEnd = len - i; end
        --        print (('-->testDataSize:%s;i:%s;bs:%s;idxEnd:%s;idxEnd or bs: %s'):format(provider.testData.data:size(1),i,bs,idxEnd,idxEnd or bs))
        local inputs = provider.testData.data:narrow(1, i, idxEnd or bs);
        local targets = provider.testData.labels:narrow(1, i, idxEnd or bs);


        local outputs = model:forward(inputs)
--        local flatOutput = _flatTensor(outputs);
        local outputsClasses = outputs:reshape(outputs:size(1),1,1,6):narrow(4,5,2);
        local targetClasses = targets:reshape(targets:size(1),1,1,6):narrow(4,6,1);
        local flatOutput = outputs;
        local flatTargets = targets;

--        require('mobdebug').start(nill,8222);


        --outputs: Bx2xHxW  target: BxHxW

        confusion:batchAdd(outputsClasses:reshape(outputs:size(1)*1*1,2), targetClasses:reshape(targets:size(1)*1*1,1)+1);
--        local result = outputs:select(2,1):csub(outputs:select(2,2));
--        result[result:gt(0)]=2;
--        result[result:le(0)]=1;
--
--        local targets = provider.testData.labels:narrow(1, i, idxEnd or bs):cuda();
--
--
--        local final = torch.eq(result,targets);
--
--        local accuracy = final:sum()/(6*549*512);
--
--        --            confusion:batchAdd(outputs, targets);
--        print('Accuracy: '..accuracy..'\n');


    end

    confusion:updateValids()
    print('Test accuracy:', confusion.totalValid * 100)
    print(confusion)

    if testLogger then
        paths.mkdir(opt.save)
--        require('mobdebug').start(nill,8222);
        testLogger:add { confusion.valids[1] ,confusion.valids[2]}
        testLogger:style { '+-','+-' }
        testLogger:plot()

        local base64im
        do
            os.execute(('convert -density 200 %s/test.log.eps %s/test.png'):format(opt.save, opt.save))
            os.execute(('openssl base64 -in %s/test.png -out %s/test.base64'):format(opt.save, opt.save))
            local f = io.open(opt.save .. '/test.base64')
            if f then base64im = f:read '*all' end
        end

        local file = io.open(opt.save .. '/report.html', 'w')
        file:write(([[
    <!DOCTYPE html>
    <html>
    <body>
    <title>%s - %s</title>
    <img src="data:image/png;base64,%s">
    <h4>optimState:</h4>
    <table>
    ]]):format(opt.save, epoch, base64im))
        for k, v in pairs(optimState) do
            if torch.type(v) == 'number' then
                file:write('<tr><td>' .. k .. '</td><td>' .. v .. '</td></tr>\n')
            end
        end
        file:write '</table><pre>\n'
        file:write(tostring(confusion) .. '\n')
        file:write(tostring(model) .. '\n')
        file:write '</pre></body></html>'
        file:close()
    end

    -- save model every 5 epochs
    if epoch % 5 == 0 then
        local filename = paths.concat(opt.save, 'model_'..epoch..'.net')
        print('==> saving model to ' .. filename)
        torch.save(filename, model:get(2):clearState())
    end

    confusion:zero()
end


for i = 1, opt.max_epoch do
    train()
    test()
end

-- CUDA_VISIBLE_DEVICES=0 th -i train_yolo.lua --backend=cudnn --save=logs/i1_yoloTest --model=model_yolo