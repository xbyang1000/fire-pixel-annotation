function InteractiveAnnotion()
clc;
clear;
close all;
num = 12;
filename = strcat('fire',num2str(num),'.jpg');
Im = imread(filename);
h = figure(1);hold on;
imshow(Im);
set(h,'Visible','off');
% if (~exist('ClassAnno','var'))
%     prompt = 'input the number of classes (n), n>=2 \n';
%     n = input(prompt);
% end

%set(h,'Visible','off');
fprintf('select 2 interest regions by click, end with "Enter" \n');
set(h,'Visible','on');
impixelinfo(h);% (x,y,r,g,b),where (x,y) means image position，and（r,g,b), pixel values.
hold on;
[ClassAnno,NumOfEachClass1] = GetConvexHull(Im,2);
% set(h,'Visible','off');
%load TempConvexVertex;
LabelSample = GetSampleLabel(ClassAnno,NumOfEachClass1,Im);

MD = TrainClassifier(LabelSample);
[ReIm,Label] = GetAnnoIm(Im,MD);
% figure;hold on;
% imshow(Im);title('origin');
% hold off;
set(h,'Visible','on');title('origin');
hold off;
g=figure(2);hold on;
imshow(uint8(ReIm));title('reload');
set(g,'Visible','on');
prompt = 'Fire well-detected,Yes (y) or No (n)?\n';
answer = input(prompt,'s');

NumNewClass = 0;
if (lower(answer)=='y')
    while(1)
        
        prompt = 'Low false warning,Yes (y) or No (n)?\n';
        answ = input(prompt,'s')
    
        if (lower(answ)=='y')
            SaveFiles(num,ReIm,Label,LabelSample,MD);
            break;
        else
            NumNewClass=NumNewClass+1;
            delete(h);
            fprintf('add a convex hull for new class\n');
            set(g,'Visible','on');
           
            %[ClassAnno,NumOfEachClass1] = GetConvexHull(Im,n);
            [tmpClassAn,tmpNumOfEachClass1] = GetConvexHull1(Im,NumNewClass);
           
            ClassAnno = [ClassAnno;tmpClassAn];
            NumOfEachClass1 = [NumOfEachClass1;tmpNumOfEachClass1];
            LabelSample = GetSampleLabel(ClassAnno,NumOfEachClass1,Im);
            MD = TrainClassifier(LabelSample);
            [ReIm,Label] = GetAnnoIm(Im,MD);            
            imshow(uint8(ReIm));title('reload');                     
        end
    end

else
    fprintf('restart the program to make all fire to be detected!\n');
end 

end
function SaveFiles(num,ReIm,Label,TrSample,KdKnnClassifier)
    ReFileName = strcat('Refire',num2str(num),'.jpg');
    imwrite(uint8(ReIm),ReFileName);
    MatFileName = strcat(num2str(num),'.mat');
    save(MatFileName); 
end

function Md =TrainClassifier(LabelSample)
Y = LabelSample(:,end);
X = LabelSample(:,1:end-1);
[Y,ind]=sort(Y);
Y = Y(ind,1);
X = X(ind,:);
% Pclass = [];
% ClassNum = length(unique(Y));
% for i=1:ClassNum
%     Pclass = [Pclass sum(Y==i)];
% end
% [MinOfClass,posi] = min(Pclass);
Md = fitcknn(X,Y,'NumNeighbors',3);


end

function RGBvec=RGB2vec(Im)
Im = double(Im);
R=Im(:,:,1);G=Im(:,:,2);B=Im(:,:,3);
R = R(:);G=G(:);B=B(:);
RGBvec = [R G B];
end

function [ReloadIm,P_Label] = GetAnnoIm(Im,Mdl)
RGBvec = RGB2vec(Im);
P_Label = predict(Mdl,RGBvec);
P_Label = (P_Label==1);% 1对应火

% Reload to image
[row,col,~] = size(Im);
PP_Label = reshape(P_Label,[row,col]);%恢复为图像大小的矩阵
ReloadIm = zeros(size(Im));%重建图像矩阵初始化，RGB分量全部取值为255代表白色
for i=1:row
    for j=1:col
        if (PP_Label(i,j)==1)  %像素标记为1
            ReloadIm(i,j,:)=Im(i,j,:); %原图像在（i,j)位置的像素复制到重建图像对应位置，替换初始化时的白色
        end
    end
end
end

%% 1 生成凸壳，用以标注样本 
% figure(1),imshow(Im);
% hold on;
% prompt = '请输入准备标注的类别数,1-6范围内\n';
% n = input(prompt);
% if(n<1 || n>6)
%     fprint('输入有误');
%     return;
% end
% 
% ClassAnno = [];
% NumOfEachClass1 = zeros(n,1);
% impixelinfo;%在图像上显示像素信息：（x,y,r,g,b）,(x,y)为坐标，（r,g,b)为三个颜色像素值，右键复制到剪粘板
% for i=1:n   
%     fprintf('准备标注第%d类,回车表示输入结束\n',i);
%     xx=[];yy=[];
%     [xx,yy]=ginput;%输入Enter结束选择，支持任意多个点,4表示选择4个点
%     xx = [xx;xx(1)];%将最后一点坐标与起始坐标连接起来，使选种区域封闭。
%     yy = [yy;yy(1)];
%     str = strcat(format(i),'-');
%     NumOfEachClass1(i,1)=size(xx,1);
%     plot(xx,yy,str,'LineWidth',2);
%     ClassAnno = [ClassAnno;yy xx];
% end
% save TempConvexVertex ClassAnno NumOfEachClass1  -V6;%存入mat文件

%% 2 标注样本
% load TempConvexVertex
% n= length(NumOfEachClass1);
% EachClassNum = zeros(size(NumOfEachClass1,1)+1,1);
% EachClassNum(2:end,1) = cumsum(NumOfEachClass1);
% %figure(1);hold on;
% I = double(Im);
% 
% k=0;
% for i=1:n
%     ConvexVertex = [];Index = [];Label = [];ind = [];
%     Index = EachClassNum(i)+1:EachClassNum(i+1);
%     ConvexVertex = ClassAnno(Index,:);
%     str = strcat(format(i),'-'); 
%    % plot(ConvexVertex(:,1),ConvexVertex(:,2),str,'LineWidth',2); 
%     x1 = [];y1=[];Region=[];
%    [x1,y1] = meshgrid(min(ConvexVertex(:,1)):1:max(ConvexVertex(:,1)),min(ConvexVertex(:,2)):1:max(ConvexVertex(:,2)));
%    Region = [x1(:),y1(:)];
%    Label = GetLabelFromConvexRegion(ConvexVertex,Region);
%    Region = floor(Region);
%    ind = find(Label==1);
%    for j=1:length(ind)
%        k=k+1;
%        PixelValue = [I(Region(ind(j),2),Region(ind(j),1),1) I(Region(ind(j),2),Region(ind(j),1),2),I(Region(ind(j),2),Region(ind(j),1),3)];
%        LabelSample(k,:) = [PixelValue i];    
%    end   
% end
% save TrainSample LabelSample -V6


%% 3 训练分类器
% load TrainSample
% Y = LabelSample(:,end);
% X = LabelSample(:,1:end-1);
% [Y,ind]=sort(Y);
% Y = Y(ind,1);
% X = X(ind,:);
% % Pclass = [];
% % ClassNum = length(unique(Y));
% % for i=1:ClassNum
% %     Pclass = [Pclass sum(Y==i)];
% % end
% % [MinOfClass,posi] = min(Pclass);
% Mdl = fitcknn(X,Y,'NumNeighbors',3);
% 
% save knnClassifier Mdl -V6
%% 标记图像并显示
% %load knnClassifier;
% %load knnClassifier5;
% load(Classifier)
% RGBvec = RGB2vec(Im);
% P_Label = predict(Mdl,RGBvec);
% P_Label = (P_Label==1);% 1对应火
% 
% % Reload to image
% [row,col,~] = size(Im);
% PP_Label = reshape(P_Label,[row,col]);%恢复为图像大小的矩阵
% ReloadIm = zeros(size(Im));%重建图像矩阵初始化，RGB分量全部取值为255代表白色
% for i=1:row
%     for j=1:col
%         if (PP_Label(i,j)==1)  %像素标记为1
%             ReloadIm(i,j,:)=Im(i,j,:); %原图像在（i,j)位置的像素复制到重建图像对应位置，替换初始化时的白色
%         end
%     end
% end
% figure;hold on
% subplot(1,2,1);imshow(Im);title('原图');
% subplot(1,2,2);imshow(uint8(ReloadIm));title('KNN标记图');
function LabelSample = GetSampleLabel(ClassAnno,NumOfEachClass1,Im)
n= length(NumOfEachClass1);
EachClassNum = zeros(size(NumOfEachClass1,1)+1,1);
EachClassNum(2:end,1) = cumsum(NumOfEachClass1);
%figure(1);hold on;
I = double(Im);
k=0;
for i=1:n
    ConvexVertex = [];Index = [];Label = [];ind = [];
    Index = EachClassNum(i)+1:EachClassNum(i+1);
    ConvexVertex = ClassAnno(Index,:);
    LineColor='rbgcmy';
    str = strcat(LineColor(i),'-'); 
   % plot(ConvexVertex(:,1),ConvexVertex(:,2),str,'LineWidth',2); 
    x1 = [];y1=[];Region=[];
   [x1,y1] = meshgrid(min(ConvexVertex(:,1)):1:max(ConvexVertex(:,1)),min(ConvexVertex(:,2)):1:max(ConvexVertex(:,2)));
   Region = [x1(:),y1(:)];
   Label = GetLabelFromRegion(ConvexVertex,Region);
   Region = floor(Region);
   ind = find(Label==1);
   for j=1:length(ind)
       k=k+1;
       % 此处坐标分量交换了，注意检查       
       PixelValue = [I(Region(ind(j),1),Region(ind(j),2),1) I(Region(ind(j),1),Region(ind(j),2),2),I(Region(ind(j),1),Region(ind(j),2),3)];
       LabelSample(k,:) = [PixelValue i];    
   end   
end
end

function [ClassAnno NumOfClass]=GetConvexHull1(OriginIm,n)
    %set(h,'Visible','on');
    LineColor='rbgcmy';
    ClassAnno = [];
    NumOfClass = zeros(n,1);
    %impixelinfo;%在图像上显示像素信息：（x,y,r,g,b）,(x,y)为坐标，（r,g,b)为三个颜色像素值，右键复制到剪粘板
        xx=[];yy=[];
        [xx,yy]=ginput;%输入Enter结束选择，支持任意多个点,4表示选择4个点
        xx = [xx;xx(1)];%将最后一点坐标与起始坐标连接起来，使选种区域封闭。
        yy = [yy;yy(1)];
        str = strcat(LineColor(n+2),'-');
        NumOfClass=size(xx,1);
        plot(xx,yy,str,'LineWidth',2);
        ClassAnno = [ClassAnno;yy xx];    
end

function [ClassAnno NumOfEachClass1]=GetConvexHull(OriginIm,n)
    %set(h,'Visible','on');
    LineColor='rbgcmy';
    ClassAnno = [];
    NumOfEachClass1 = zeros(n,1);
    %impixelinfo;%在图像上显示像素信息：（x,y,r,g,b）,(x,y)为坐标，（r,g,b)为三个颜色像素值，右键复制到剪粘板
    for i=1:n   
        xx=[];yy=[];
        [xx,yy]=ginput;%输入Enter结束选择，支持任意多个点,4表示选择4个点
        xx = [xx;xx(1)];%将最后一点坐标与起始坐标连接起来，使选种区域封闭。
        yy = [yy;yy(1)];
        str = strcat(LineColor(i),'-');
        NumOfEachClass1(i,1)=size(xx,1);
        plot(xx,yy,str,'LineWidth',2);
        ClassAnno = [ClassAnno;yy xx];
    end   
end


function Label = GetLabelFromRegion(ConvexPosi,PosiRegion)
%% 纠正顶点集次序
X = ConvexPosi(1:end-1,:);%去掉最后一个重复节点
vertex_set = GetConVertex(ConvexPosi(1:end-1,:));%顶点集下标
convex_vertex = X(vertex_set',:);%按生成凸壳顺序存储顶点
%hold on;
%LineVertexBoundary(X,vertex_set,'r');

%% 
% [sign1,center_proj norm_vectors]  = GetSignBarycenter_NormalOfLineSegment(convex_vertex,vertex_set);
% n1 = sum(sign1);
row = size(convex_vertex,1);
center_proj = zeros(row,size(convex_vertex,2));
m = mean(convex_vertex);%center
convex_vertex = [convex_vertex;convex_vertex(1,:)];
for i=1:row
    % temp = v2-v1;
    temp = convex_vertex((i+1),:)-convex_vertex((i),:);
   
    % 过v1和v2点的直线方程为： (m-mp)*(v2-v1)=0,mp为m在该直线的投影，
    % 则向量长度 |mp-v1| = ( (m-v1)*(v2-v1)/||v2-v1|| )
    %   向量方向取 v2-v1的单位向量： (v2-v1)/||v2-v1||
    k  = ((m-convex_vertex((i),:))*temp')/(temp*temp');
    mp = convex_vertex((i),:)+k*temp;
    center_proj(i,:) = mp;
end

% 测试投影是否落在各线段上
% hold on;
% plot(center_proj(:,1),center_proj(:,2),'y*');
% plot(m(:,1),m(:,2),'r*');

P = repmat(m,row,1) - center_proj;
num = size(PosiRegion,1);
Label = zeros(num,1);
for i=1:num
    tmp = PosiRegion(i,:);
    Q = repmat(tmp,row,1) - convex_vertex(1:end-1,:);
    value = diag(P*Q');
    lamda = min(value);
    if (lamda>=0),   Label(i,1)=1;    end
end

end
function vertex_set = GetConVertex(trainx)
maxdist = 0;
n = size(trainx,1);
for i=1:n
    for j=i+1:n
        temp = trainx(i,:)-trainx(j,:);
        if temp*temp' > maxdist
            first = i;
            second = j;
            maxdist = temp*temp';
        end
    end
end


% ================= 初始化 ========================
w = trainx(first,:)-trainx(second,:);% 方向指向点1
b = w*trainx(second,:)';             % 点2为定点

vertex_set = [];
ind = 1:size(trainx,1);
set = [ind];
fixpoint = second;

anoth_onLine = w\b;
fix_direct = anoth_onLine'-trainx(second,:);
fix_direct = fix_direct/norm(fix_direct);
 j = 1;
while ~ismember(fixpoint,vertex_set)
    left_set = setdiff(set,fixpoint);
    vertex_set = [vertex_set fixpoint];
    tempx = trainx(left_set',:)-repmat(trainx(fixpoint,:),length(left_set),1);
    
    % 行归一化
    for i=1:size(tempx,1),  tempx(i,:)=tempx(i,:)/norm(tempx(i,:));end
    cos_angle = tempx*fix_direct';
    [value,posi] = max(cos_angle);
    
    fixpoint = left_set(posi);
    anoth_point = vertex_set(1,length(vertex_set));
    fix_direct = trainx(fixpoint,:)-trainx(anoth_point,:);
    fix_direct = fix_direct/norm(fix_direct);
    j = j+1;
end
end
