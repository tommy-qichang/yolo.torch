--
-- Description: useful criterion for YOLO
-- User: Qi Chang(tommy) <tommy.qichang@gmail.com>
-- Date: 7/7/16
-- Time: 4:15 PM
-- 
--
require 'nn'
RegionProposalCriterion, parent = torch.class('nn.RegionProposalCriterion','nn.Criterion')

-- defult sides:7x7 grid cells for one image.
-- defult each grid propose n proposals.
-- defult class number is 2.
-- v1. first version the defult n = 2;
function RegionProposalCriterion:__init(weights,bWeights,sides,isCuda)
    parent.__init(self)
    if sides ~= nil then
        self.sides = sides
    else
        self.sides = 7
    end
--
--    if n ~= nil then
--        self.n = n
--    else
--        self.n = 2
--    end

    if weights ~= nil then
        assert(type(weights) == 'number', "weights input should be a value apply to coord proposals")
        self.weights = weights
    else
        self.weights = 4
    end
    if bWeights ~= nil then
        assert(type(bWeights) == 'number', "weights input should be a value apply to coord proposals")
        self.bWeights = bWeights
    else
        self.bWeights = 0.25
    end

    if isCuda ~= nil then
        self.isCuda = isCuda;
    else
        self.isCuda = false
    end


end


--v1:
--input: BatchSize x rowNumber x columnNumber x (x,y,w,h,class1,class2))
-- Bx(7*7*(4+2)) = Bx(294)
--target: Bx(7*7*6) (x,y,\sqrt w,\sqrt h,class1,class2)


function RegionProposalCriterion:updateOutput(input, target)
    _input = input:reshape(input:size(1),self.sides,self.sides,6);
    _target = target:reshape(input:size(1),self.sides,self.sides,6);

    assert( input:nElement() == target:nElement(),
        "input and target size mismatch")

    self.buffer = self.buffer or _input.new()

    local buffer = self.buffer
    local weights = self.weights
    local bWeights = self.bWeights
    local output,label
    buffer:resizeAs(_input)

    local mask = _target:narrow(4,6,1):eq(1):double()
--    local mask = _target:narrow(4,6,1):eq(1):double()
--    mask = torch.expand(mask,_input:size())
    mask = torch.expand(mask,_input:narrow(4,1,4):size())
    mask = torch.cat(mask,torch.ones(_target:narrow(4,5,2):size()))

    if self.isCuda == true then
        mask = mask:cuda();
    end

    --set input value into buffer with mask.
    buffer:cmul(_input,mask)
--    _target:narrow(4,1,4):cmul(mask:double())

    -- (x_i - x~_i)^2 + (y_i - y~_i)^2 + (ww_i - ww~_i)^2 + (hh_i - hh~_i)^2 + (p_i - p~_i)^2
    buffer:csub(_target):pow(2)
    --add weights into coordinates proposals.
    if weights ~= nil then
--        bWeight = 0.02
        buffer:narrow(4,1,4):mul(weights)
--        local balanceMask = mask:narrow(4,5,2) + ((1-mask:narrow(4,5,2)):mul(bWeight))
--        buffer:narrow(4,5,2):cmul(balanceMask:cuda())
    end

    -- solve classes unbalance problem by add a \lumbda_noobj
    buffer:narrow(4,5,1):mul(bWeights)


    proposalCost = (torch.sum(buffer:narrow(4,1,4))) / (buffer:narrow(4,1,4):nElement());


    classCost = torch.sum(buffer:narrow(4,6,1)) / (mask:narrow(4,6,1):nElement());

    noobjCost = torch.sum(buffer:narrow(4,5,1)) / (mask:narrow(4,5,1):nElement());

    print(('proposalCost:'..proposalCost..', classCost:'..classCost..', noobjCost:'..noobjCost))
--    require('mobdebug').start(nill,8222)

    output = torch.sum(buffer)
    output = output / _input:nElement()
    self.output = output

    return self.output
end

function RegionProposalCriterion:updateGradInput(input, target)
    _input = input:reshape(input:size(1),self.sides,self.sides,6);
    _target = target:reshape(input:size(1),self.sides,self.sides,6);

    assert( input:nElement() == target:nElement(),
        "input and target size mismatch")

    self.buffer = self.buffer or _input.new()

    local buffer = self.buffer
    local weights = self.weights
    local bWeights = self.bWeights
    local gradInput = self.gradInput
    local label
    gradInput:resizeAs(_input)
    buffer:resizeAs(_input)

    local mask = _target:narrow(4,6,1):eq(1):double()
--    mask = torch.expand(mask,_input:size())
    mask = torch.expand(mask,_input:narrow(4,1,4):size())
    mask = torch.cat(mask,torch.ones(_target:narrow(4,5,2):size()))
    if self.isCuda == true then
        mask = mask:cuda();
    end

    --set input value into buffer with mask.
    buffer:cmul(_input,mask)
--    _target:narrow(4,1,4):cmul(mask:double())

    -- (x~_i - x_i)+(y~_i-y_i)
    gradInput:csub(buffer,_target)

    --add weights into coordinates proposals.
    if weights ~= nil then
--        bWeight = 0.02
        gradInput:narrow(4,1,4):mul(weights)
        --add balanced mask for class unbalance problem
--        local balanceMask = mask:narrow(4,5,2) + ((1-mask:narrow(4,5,2)):mul(bWeight))
--        gradInput:narrow(4,5,2):cmul(balanceMask:cuda())
    end

    gradInput:narrow(4,5,1):mul(bWeights)


    gradInput:div(_target:nElement())



    gradInput:resize(_input:size(1),self.sides*self.sides*6);

    self.gradInput = gradInput*2;
    return self.gradInput
end






