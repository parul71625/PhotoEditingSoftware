function varargout = PhotoEditingSoftware(varargin)
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @PhotoEditingSoftware_OpeningFcn, ...
                   'gui_OutputFcn',  @PhotoEditingSoftware_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
end

% --- Executes just before PhotoEditingSoftware is made visible.
function PhotoEditingSoftware_OpeningFcn(hObject, eventdata, handles, varargin)
handles.output = hObject;
guidata(hObject, handles);
end

% --- Outputs from this function are returned to the command line.
function varargout = PhotoEditingSoftware_OutputFcn(hObject, eventdata, handles) 
varargout{1} = handles.output;
end




% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    % Red-Eye Removal Feature
    I=imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/redeye.jpg');
    plot(6,6)
    image(I);
    P = ginput(4)';
    Q = ginput(4)';
    red_threshold = 100;
    J=I;
    for i = int16(min(P(2,:))):1:int16(max(P(2,:)))
    for j = int16(min(P(1,:))):1:int16(max(P(1,:)))
    c = I(i,j,:);
    red=c(1);
    if red > red_threshold
    J(i,j,:)=0;
    end
    end
    end
    for i = int16(min(Q(2,:))):1:int16(max(Q(2,:)))
    for j = int16(min(Q(1,:))):1:int16(max(Q(1,:)))
    c = I(i,j,:);
    red=c(1);
    if red > red_threshold
    J(i,j,:)=0;
    end
    end
    end
    subplot(2,1,1)
    imshow(I);title('Red Eye');
    subplot(2,1,2)
    imshow(J);title('No Red Eye');
end


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
    % Adjust Brightness and Contrast
    C = get(handles.edit1,'String'); 
    C = str2num(C);
     if isempty(C)
         h = msgbox('Specify adjustment values (Eg1:[.2,.3]  Eg2:[.2 .3 0; .6 .7 1])');
         return
     end    
    A =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg');
    I = rgb2gray(A);
    I = imadjust(I);
    subplot(3,3,[1,2])
    imshow(A);title('Original Image');
    subplot(3,3,[4,5])
    imshow(I);title('Adjusted Image 1 (B/W)');
    I2 =  imadjust(A, C, []);
    subplot(3,3,[7,8])
    imshow(I2);title('Adjusted Image 2 (Colored)');
end



% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
    % Sharpening
    T = get(handles.edit1,'String');
    
     if isempty(T)
         h = msgbox('Specify type of sharpening you want to apply (Eg:average, disk, gaussian, laplacian, log, motion, prewitt, sobel)');
         return
     end       
    f1 =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg');
    f1 = rgb2gray(f1);
    subplot(2,5,[1,1.99])
    imshow(f1); title('Original image');
    subplot(2,5,[3.21,4])
    imhist(f1); title('Original image histogram');
    w4=fspecial(T);
    f2=im2double(f1);
    g4=f2-imfilter(f2, w4, 'replicate');
    subplot(2,5,[6,6.99])
    imshow(g4); title('Sharpened Image');
    subplot(2,5,[8.21,9])
    imhist(g4); title('Sharpened image histogram');
end



% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
    %Softening
    V = get(handles.edit1,'String');
    V = str2num(V);
     if isempty(V)
         h = msgbox('Specify values for softening (Eg1:[10,10])  (Eg2: [30,30])');
         return
     end       
    f1 =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg');
    f1 = rgb2gray(f1);
    subplot(2,5,[1,1.99])
    imshow(f1); title('Original image');
    subplot(2,5,[3.21,4])
    imhist(f1); title('Original image histogram');
    K = wiener2(f1,V);
    subplot(2,5,[6,6.99])
    imshow(K); title('Softened Image');
    subplot(2,5,[8.21,9])
    imhist(K); title('Softened image histogram');
end



% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
    % Special Effects
    option = get(handles.edit1,'String');
    n = str2num(option);
     if isempty(option)
         h = msgbox({'1. Flip' '2. Rotate|AOR' '3. Negative' '4. Collage' '5. Glassy Effect'});
         return
     end       
    I =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg');
    if strcmp(option,'Flip') | n==1
        I2 = flipdim(I,2);
        I3 = flipdim(I,1);
        I4 = flipdim(I3,2);
        subplot(2,3,1)
        imshow(I); title('Original Image');
        subplot(2,3,2)
        imshow(I2); title('Flipped Image Horizontally');
        subplot(2,3,4)
        imshow(I3); title('Flipped Image Vertically');
        subplot(2,3,5)
        imshow(I4); title('Flipped Image Vertically/Horizontally');
    end

    if strfind(option, 'Rotate')
        str = regexp(option,'\|','split');
        angle = str2double(str(2));
        I2 = imrotate(I,angle);
        subplot(2,3,[1,2])
        imshow(I); title('Original Image');
        subplot(2,3,[4,5])
        imshow(I2); title('Rotated image');
    end
 
 if strcmp(option,'Negative') | n==3
    A =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg');
    negimg = imcomplement(A);
    subplot(2,3,[1,2])
    imshow(A); title('Original Image');
    subplot(2,3,[4,5])
    imshow(negimg); title('Negative image');
 end
 
 if strcmp(option,'Collage') | n==4
    I2 =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird2.jpg');
    I3 =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird3.jpg');
    I4 =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird4.jpg');
    plotSize = [256,256];
    B = plotSize./[2,1];
    C = plotSize./[2,2];
    D = plotSize./[2,2];
    collImg = [imresize(I2,B);imresize(I3,C),imresize(I4,D)];
    imshow(collImg);title('Collage');
 end
 
 if strcmp(option,'Glassy Effect') | n==5
     A =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg');
    m=6;                                                                   
    n=7;
    Image=uint8(zeros([size(A,1)-m,size(A,2)-n,3]));
    for i=1:size(A,1)-m
        for j=1:size(A,2)-n
            mymask=A(i:i+m-1,j:j+n-1,:);
            %Select a pixel value from the neighborhood.
            x2=ceil(rand(1)*m);
            y2=ceil(rand(1)*n);
            Image(i,j,:)=mymask(x2,y2,:);
        end
    end
    subplot(2,3,[1,2])
    imshow(A); title('Original Image');
    subplot(2,3,[4,5])
    imshow(Image); title('Glassy image');
 end

end



% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
    % Straighten a picture
    %Straightening 1st image
    I = imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/rect.jpg');
    I=rgb2gray(I);
    subplot(2,3,1)
    imshow(I); title('Tilted Rectangle 1');
    I2 = imclearborder(im2bw(I));
    [y,x] = find(I2);
    [~,loc] = min(y+x);
    C = [x(loc),y(loc)];
    [~,loc] = min(y-x);
    C(2,:) = [x(loc),y(loc)];
    [~,loc] = max(y+x);
    C(3,:) = [x(loc),y(loc)];
    [~,loc] = max(y-x);
    C(4,:) = [x(loc),y(loc)];
    L = mean(C([1 4],1));
    R = mean(C([2 3],1));
    U = mean(C([1 2],2));
    D = mean(C([3 4],2));
    C2 = [L U; R U; R D; L D];
    T = cp2tform(C ,C2,'projective');
    IT = imtransform(im2bw(I),T);
    subplot(2,3,4)
    imshow(IT);title('Straight Rectangle 1');

    %Straightening 2nd image
    I = imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/rect2.jpg');
    I=rgb2gray(I);
    subplot(2,3,2)
    imshow(I); title('Tilted Rectangle 2');
    I2 = imclearborder(im2bw(I));
    [y,x] = find(I2);
    [~,loc] = min(y+x);
    C = [x(loc),y(loc)];
    [~,loc] = min(y-x);
    C(2,:) = [x(loc),y(loc)];
    [~,loc] = max(y+x);
    C(3,:) = [x(loc),y(loc)];
    [~,loc] = max(y-x);
    C(4,:) = [x(loc),y(loc)];
    L = mean(C([1 4],1));
    R = mean(C([2 3],1));
    U = mean(C([1 2],2));
    D = mean(C([3 4],2));
    C2 = [L U; R U; R D; L D];
    T = cp2tform(C ,C2,'projective');
    IT = imtransform(im2bw(I),T); %IM2BW is not necessary
    subplot(2,3,5)
    imshow(IT); title('Straight Rectangle 2');
end



% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
    % Crop a picture    
    C = get(handles.edit1,'String');    
    C = str2num(C);
     if isempty(C)
         h = msgbox('Enter the coordinates for cropping   (Eg: [75 68 130 112])');
         return
     end    
    i=imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg');
    subplot(2,3,[1,2])
    imshow(i);title('Original image');
    [I2, rect] = imcrop(i);
    subplot(2,3,4)
    imshow(I2);title('Cropped Image 1');
    I3 = imcrop(i,C);
    subplot(2,3,5)
    imshow(I3);title('Cropped Image 2');

end



% --- Executes on button press in pushbutton8.
function pushbutton8_Callback(hObject, eventdata, handles)
    % Change the background of image
    color = get(handles.edit1,'String');
    color = str2num(color);
     if isempty(color)
         h = msgbox('Enter the coordinates of color you want to change background color to  (Eg: [255 128 127])');
         return
     end           
    
    A = imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/black.jpg');
    subplot(2,3,1)
    imshow(A); title('Black background');
    [nRow,nCol,nColor] = size(A);
    for i = 1:nRow
    for j = 1:nCol
    c = A(i,j,:);
    if c==0  | c==255 %Background color is originally black or white
    A(i,j,1)=color(1);
    A(i,j,2)=color(2);
    A(i,j,3)=color(3);
    end
    end
    end
    subplot(2,3,4)
    imshow(A); title('Background Change');

    A = imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/white.jpg');
    subplot(2,3,2)
    imshow(A); title('White background');
    [nRow,nCol,nColor] = size(A);
    for i = 1:nRow
    for j = 1:nCol
    c = A(i,j,:);
    if c==0  | c==255 %Background color is originally black or white
    A(i,j,1)=color(1);
    A(i,j,2)=color(2);
    A(i,j,3)=color(3);
    end
    end
    end
    subplot(2,3,5)
    imshow(A); title('Background Change');
end



% --- Executes on button press in pushbutton9.
function pushbutton9_Callback(hObject, eventdata, handles)
    % Add a Caption
    str = get(handles.edit1,'String');
     if isempty(str)
         h = msgbox('Enter Caption|Color|Size|Position  (Eg: [Title|1.0 0.9 0.8|127|45 34])');
         return
     end       
    str = regexp(str,'\|','split');
    A = im2double((imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg')));
    subplot(2,3,[1,2])
    imshow(A); title('Image without Caption');
    Text = str{1};
    H = vision.TextInserter(Text);  
    H.Color = str2num(str{2});
    H.FontSize = str2double(str{3});
    H.Location = str2num(str{4});
    %Birdieeee|0.9 0.7 0.4|123|45 67
    %H = vision.TextInserter('Birdieeee');  
    %H.Color = [0.9 0.7 0.4];
    %H.FontSize = 123;
    %H.Location = [45 67];
    I = im2double((imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg')));
    CI = step(H, I);
    subplot(2,3,[4,5])
    imshow(CI); title('Image with Caption');

end



% --- Executes on button press in pushbutton10.
function pushbutton10_Callback(hObject, eventdata, handles)
    % Noise Reduction
    A =imread('C:/Users/Parul/Desktop/8th Sem/MATLAB/GUI MATLAB/Images/bird.jpg');
    A = rgb2gray(A);
    subplot(2,3,[1.56,2])
    imshow(A);title('Original Image');
    N = imnoise(A,'gaussian',0,0.01);
    subplot(2,3,[4,4.30])
    imshow(N);title('Image with noise');
    NR = wiener2(N,[5,5]);
    subplot(2,3,[5.22,5.51])
    imshow(NR); title('Image after Noise Reduction');
end



function edit1_Callback(hObject, eventdata, handles)
end


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
end


% --- Executes during object creation, after setting all properties.
function axes6_CreateFcn(hObject, eventdata, handles)
end