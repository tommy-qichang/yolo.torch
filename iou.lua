--
-- Description: Intersection Over Union
-- a: first box, a Bx7x7x6 tensor. x_min,y_min,w,h
-- b: second box, another Bx7x7x6 tensor.
-- return B*7*7x1 tensor, indicating the IOU of each two boxes.
function iou(ia,ib)
    area_a = torch.cmul(ia:narrow(2,3,1),ia:narrow(2,4,1));
    area_b = torch.cmul(ib:narrow(2,3,1),ib:narrow(2,4,1));



    a_min = torch.div(ia:narrow(2,1,1),)



    area_a = ia:clone():narrow(2,3,1):cmul(ia:narrow(2,4,1));
    area_b = ib:clone():narrow(2,3,1):cmul(ib:narrow(2,4,1));
    a_min = ia:narrow(2,1,2);
    b_min = ib:narrow(2,1,2);
    a_max = a_min:clone():add(ia:narrow(2,3,2));
    b_max = b_min:clone():add(ib:narrow(2,3,2));
    x_max,_ = torch.min(torch.cat(a_max:select(2,1),b_max:select(2,1),2),2);
    x_min,_ = torch.max(torch.cat(a_min:select(2,1),b_min:select(2,1),2),2);
    y_max,_ = torch.min(torch.cat(a_max:select(2,2),b_max:select(2,2),2),2);
    y_min,_ = torch.max(torch.cat(a_min:select(2,2),b_min:select(2,2),2),2);
    x_span = x_max:csub(x_min);
    x_span = x_span:gt(0):double():cmul(x_span);
    y_span = y_max:csub(y_min);
    y_span = y_span:gt(0):double():cmul(y_span);
    --    return {x_span,y_span}
    area_i = x_span:cmul(y_span);
    union = area_a + area_b - area_i;
    return area_i:cdiv(union);
end



--function iou(ia,ib)
--    area_a = ia:clone():narrow(2,3,1):cmul(ia:narrow(2,4,1));
--    area_b = ib:clone():narrow(2,3,1):cmul(ib:narrow(2,4,1));
--    a_min = ia:narrow(2,1,2);
--    b_min = ib:narrow(2,1,2);
--    a_max = a_min:clone():add(ia:narrow(2,3,2));
--    b_max = b_min:clone():add(ib:narrow(2,3,2));
--    x_max,_ = torch.min(torch.cat(a_max:select(2,1),b_max:select(2,1),2),2);
--    x_min,_ = torch.max(torch.cat(a_min:select(2,1),b_min:select(2,1),2),2);
--    y_max,_ = torch.min(torch.cat(a_max:select(2,2),b_max:select(2,2),2),2);
--    y_min,_ = torch.max(torch.cat(a_min:select(2,2),b_min:select(2,2),2),2);
--    x_span = x_max:csub(x_min);
--    x_span = x_span:gt(0):double():cmul(x_span);
--    y_span = y_max:csub(y_min);
--    y_span = y_span:gt(0):double():cmul(y_span);
--    --    return {x_span,y_span}
--    area_i = x_span:cmul(y_span);
--    union = area_a + area_b - area_i;
--    return area_i:cdiv(union);
--end





