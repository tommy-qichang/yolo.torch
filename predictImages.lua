--
-- Description: ${DESC}
-- User: Qi Chang(tommy) <tommy.qichang@gmail.com>
-- Date: 7/17/16
-- Time: 3:59 PM
-- 
require 'nn';
require 'optim'
require 'cunn'
require 'cudnn'
require 'image';
dofile './provider_yolo_1x1.lua'
require 'mattorch'


--modelPath = 'logs/i5_yoloTest_0.5_1x1/model_50.net';

modelPath = 'logs/i5_yoloTest_0.5_1x1/model_15.net';

model = nn.Sequential();
--model:add(nn.BatchFlip():float())

model:add(nn.Copy('torch.FloatTensor', 'torch.CudaTensor'));

--if modelPath and paths.filep(modelPath) then
    model:add(torch.load(modelPath));
    print('==> load exist model:' .. modelPath);
--else
--model:add(dofile(opt.model .. '.lua'):cuda());
--end
bs = 7;
len = provider.testData.data:size(1);
outputSet = torch.Tensor(provider.testData.data:size(1),6)

print('predict....')
for i = 1, len, bs do
    if (i + bs) > len then idxEnd = len - i; end
    --        print (('-->testDataSize:%s;i:%s;bs:%s;idxEnd:%s;idxEnd or bs: %s'):format(provider.testData.data:size(1),i,bs,idxEnd,idxEnd or bs))
    inputs = provider.testData.data:narrow(1, i, idxEnd or bs);
    targets = provider.testData.labels:narrow(1, i, idxEnd or bs)

    outputs = model:forward(inputs)
    outputs = torch.squeeze(outputs);


    outputSet:narrow(1,i,idxEnd or bs):copy(outputs:float())


end

label = torch.squeeze(provider.testData.labels):narrow(2,6,1);
label = label +1;


outputs_f = outputSet:narrow(2, 6,1) - outputSet:narrow(2, 5,1);

print('save mat....')
mattorch.save('matlab/pred_output.mat', outputs_f);
mattorch.save('matlab/pred_outputset.mat', outputSet);
mattorch.save('matlab/pred_label.mat', label:double());

