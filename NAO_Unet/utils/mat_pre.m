clear all;
path = ('G:\NAO_UNet\data\BSR\BSDS500\data\groundTruth\test\')
d = dir('G:\NAO_UNet\data\BSR\BSDS500\data\groundTruth\test\*.mat')
for i = 1:length(d)
    fileName{i} = d(i).name
    x = load([path,fileName{i}])
    len = size(x.groundTruth,2);
    bmp = x.groundTruth{1, 1}.Boundaries;
    for k=1:len
       bmp = bmp| x.groundTruth{1, k}.Boundaries;
    end
    imshow(bmp)
    title('bitwise OR Operation')
    newDir = regexp(fileName{i},'.mat','split')
    newDir = [path,string(char(newDir(1))),'.bmp']
    imwrite(bmp,newDir)
end

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