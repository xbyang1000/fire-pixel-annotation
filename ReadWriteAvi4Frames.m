clear;
clc;
%% read video 'AVI' file and save them as RGB images by time interval seconds
% -------------------------------------------------------------------------------------
% k=0;
filename = 'fire4';
classifier = strcat('knnClassifier',filename,'.mat');
load(classifier);% knnClassifier; %从第一帧训练得到，001.jpg

TR_sample = strcat('TrainSample',filename,'.mat');
load(TR_sample);
RandOrder = randperm(size(LabelSample,1))';
LabelSample = LabelSample(RandOrder,:);
Tr = LabelSample(:,1:3);
Y  = LabelSample(:,4);
Y = (Y==1)*2-1; %转为两类
[value,posi] = sort(Y);
Y = Y(posi,:);
Tr= Tr(posi,:);

%训练和测试SVM
SVMModel = fitcsvm(Tr,Y,'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
%[A,score] = predict(SVMModel,Tr);

num = 1;
str_num = sprintf('%03d',num);
ImName = strcat(filename,'AVI',str_num,'.jpg');
Im = imread(ImName);%fire5、7效果不好,10误报高,重新标注识报率显著降低

[row,col,~] = size(Im);
Blank = uint8(ones(2,col*2+2,3)*255);
BlankV = uint8(ones(row,2,3)*255);
AVIobj = strcat(filename,'.avi');
obj = VideoReader(AVIobj);
f = obj.NumberOfFrames;


SelectFrames = [floor([f/4,f/3,f/2]),f];
IMM = [];REIM = [];SVMIM = [];KMED = []; RUIM= [];
for j=1:length(SelectFrames)    
    str_num = sprintf('%03d',SelectFrames(j));
    ImName = strcat(filename,'AVI',str_num,'.jpg');
    Im = imread(ImName);  
    IMM = [IMM Im];
    RGBvec = RGB2vec(Im);
    P_Label = predict(Mdl,RGBvec);
    ind = find(P_Label==1); % 1表示 火
    RGB = zeros(size(RGBvec));
    RGB(ind,:)= RGBvec(ind,:);
    R = reshape(RGB(:,1),[row,col]);
    G = reshape(RGB(:,2),[row,col]);
    B = reshape(RGB(:,3),[row,col]);
    
    ReIm = uint8(cat(3,R,G,B));  %重建视频，只保留有“火”的像素 
    REIM = [REIM ReIm];
    %tmpIm = [Im;Blank;ReIm];     %Blank是两个视频中分隔的白色空白，两行空白
    
   SVMIm = GetSVM_Result(Im,RGBvec,SVMModel);
   SVMIM = [SVMIM SVMIm];    
    
   % k-medois
    [Kmed1,Kmed2] = GetKmedois(Im,RGBvec);
    KIm = [Kmed1 Kmed2];
    f=figure(8);hold on;
    imshow(KIm);
    prompt = '选择有火的聚类，输入1或2 \n';
    n = input(prompt);
    if(n==1)
        KMED = [KMED Kmed1];
    else
        KMED = [KMED Kmed2];
    end
    delete(f);
    %rule
    RuIm = GetRule(Im);
    RUIM = [RUIM RuIm];    
end

test4Flame = [IMM(:,1:col,:) BlankV IMM(:,col+1:2*col,:)];
test4Flame = [test4Flame; Blank];
test4Flame = [test4Flame;IMM(:,2*col+1:3*col,:) BlankV IMM(:,3*col+1:4*col,:)];
figure(1);hold on
imshow(test4Flame);title('origin');

test4Flameknn = [REIM(:,1:col,:) BlankV REIM(:,col+1:2*col,:)];
test4Flameknn = [test4Flameknn; Blank];
test4Flameknn = [test4Flameknn;REIM(:,2*col+1:3*col,:) BlankV REIM(:,3*col+1:4*col,:)];
figure(2);hold on
imshow(test4Flameknn);title('knn');

test4FlameRUIM = [RUIM(:,1:col,:) BlankV RUIM(:,col+1:2*col,:)];
test4FlameRUIM = [test4FlameRUIM; Blank];
test4FlameRUIM = [test4FlameRUIM;RUIM(:,2*col+1:3*col,:) BlankV RUIM(:,3*col+1:4*col,:)];
figure(3);hold on
imshow(test4FlameRUIM);title('rule');

test4FlameKMED = [KMED(:,1:col,:) BlankV KMED(:,col+1:2*col,:)];
test4FlameKMED = [test4FlameKMED; Blank];
test4FlameKMED = [test4FlameKMED;KMED(:,2*col+1:3*col,:) BlankV KMED(:,3*col+1:4*col,:)];
figure(4);hold on
imshow(test4FlameKMED);title('kmedios');

test4FlameSVM = [SVMIM(:,1:col,:) BlankV SVMIM(:,col+1:2*col,:)];
test4FlameSVM = [test4FlameSVM; Blank];
test4FlameSVM = [test4FlameSVM;SVMIM(:,2*col+1:3*col,:) BlankV SVMIM(:,3*col+1:4*col,:)];
figure(5);hold on
imshow(test4FlameSVM);title('svm');

function RuIm = GetRule(Im)
  
    [~,Labels] = GetPixelLabelAndYCbCr(Im);
    P_Label = Labels(:);
    RGBvec = RGB2vec(Im);
    
    ind = find(P_Label==1); % 1表示 火
    [row,col,~] = size(Im);
    RGB = zeros(size(RGBvec));
    RGB(ind,:)= RGBvec(ind,:);
    R = reshape(RGB(:,1),[row,col]);
    G = reshape(RGB(:,2),[row,col]);
    B = reshape(RGB(:,3),[row,col]);
    RuIm = uint8(cat(3,R,G,B));     
end
function [Kmed1,Kmed2] = GetKmedois(Im,RGBvec)
  idx = kmedoids(RGBvec,2);
  [row,col,~] = size(Im);
  label = unique(idx);
  ind1 = find(idx==label(1));
  ind2 = find(idx==label(2));
    
  RGB = zeros(size(RGBvec));
  RGB(ind1,:)= RGBvec(ind1,:);
  R = reshape(RGB(:,1),[row,col]);
  G = reshape(RGB(:,2),[row,col]);
  B = reshape(RGB(:,3),[row,col]);
  Kmed1 = uint8(cat(3,R,G,B)); 
  
  RGB = zeros(size(RGBvec));
  RGB(ind2,:)= RGBvec(ind2,:);
  R = reshape(RGB(:,1),[row,col]);
  G = reshape(RGB(:,2),[row,col]);
  B = reshape(RGB(:,3),[row,col]);
  Kmed2 = uint8(cat(3,R,G,B));     
end

function SVMIm = GetSVM_Result(Im,RGBvec,SVMModel)
    [P_Label,score] = predict(SVMModel,RGBvec);
    ind = find(P_Label==1); % 1表示 火
    [row,col,~] = size(Im);
    RGB = zeros(size(RGBvec));
    RGB(ind,:)= RGBvec(ind,:);
    R = reshape(RGB(:,1),[row,col]);
    G = reshape(RGB(:,2),[row,col]);
    B = reshape(RGB(:,3),[row,col]);
    SVMIm = uint8(cat(3,R,G,B)); 
end

   
    






