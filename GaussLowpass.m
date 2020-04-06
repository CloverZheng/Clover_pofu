I=round((fspecial('gaussian',[15,15],2.9))*10000);
%xlswrite('GaussM',I);
ratio=sum(I(:));   %这个是模板前的因子
image=imread('Moc_gray.jpg');
image=rgb2gray(image);  %转成灰度图
[height,width]=size(image);
padding_image=zeros(height+14,width+14);    %边界拓延选择最简单的0拓延
padding_image(8:height+7,8:width+7)=image;
new_image=zeros(height,width);
load=0;
for i=8:height+7
    for j=8:width+7
       
            for m=i-7:i+7
                for n=j-7:j+7
                   load=load+padding_image(m,n)*I(m-i+8,n-j+8);
                end
            end
        load=load/ratio;
        new_image(i-7,j-7)= round(load);
        load=0;
    end
end
imwrite(mat2gray(new_image),'new_image.jpg');