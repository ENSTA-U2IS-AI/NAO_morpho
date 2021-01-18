clear all;
path = ('G:\NAO_UNet\data\BSR\BSDS500\data\groundTruth\train\')
d = dir('G:\NAO_UNet\data\BSR\BSDS500\data\groundTruth\train\*.mat')
B=[0 1 0
    1 1 1
    0 1 0];
proportion = 0.0
for i = 1:length(d)
    fileName{i} = d(i).name
    x = load([path,fileName{i}])
    len = size(x.groundTruth,2);
    bmp = x.groundTruth{1, 1}.Boundaries;
    for k=1:len
       bmp = bmp| x.groundTruth{1, k}.Boundaries;
    end
%     bmp = unit8(bmp)
 
    bmp = imdilate(bmp,B)
    imshow(bmp)
    numVal_1 = sum(sum(bmp));
    numVal_0 = length(find(bmp==0));
    num_sum = numVal_1+numVal_0;
    proportion = proportion+ numVal_1/num_sum
    title('bitwise OR Operation')
    newDir = regexp(fileName{i},'.mat','split')
    newDir = [path,string(char(newDir(1))),'.bmp']
    imwrite(bmp,newDir)
end
proportion = proportion/length(d)
% f = struct2cell(x);
% Boundary = x.groundTruth{1, 6}.Boundaries;
% Segmentation = x.groundTruth{1, 6}.Segmentation
% im = cell(1,7)
% len = size(x.groundTruth,2);
% bmp = x.groundTruth{1, 1}.Boundaries;
% for k=1:len
%    bmp = bmp| x.groundTruth{1, k}.Boundaries;
% end
%  imshow(bmp)
%  title('bitwise OR Operation')
%  imwrite(bmp,'./data/BSR/BSDS500/data/groundTruth/train/8049.bmp')
 
% B=[0 1 0
% 1 1 1
% 0 1 0];
% bmp_dilation = imdilate(bmp,B)
% imshow(bmp_dilation)
% title('bitwise OR Operation and dilation operation')
% size =size(x.groundTruth,2)
% imb = imshow(Boundary)