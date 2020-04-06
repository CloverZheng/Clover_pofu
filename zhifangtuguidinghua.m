%原始图像，像素的最高亮度级为160
R=[1 7 8 9 10 11 14;
    5 2 6 7 14 12 15;
    3 4 7 8 6 9 11;
    2 1 4 7 8 8 9;
    8 4 5 9 11 12 10;
    8 10 11 15 16 10 13;
    13 6 9 16 13 12 10]*10;
%统计像素矩阵中各像素出现的次数
R1=tabulate(R(:));  %返回：像素值 出现次数 占比%
R2=R1(:,1); %I1的第一行，0到160
R3=R1(:,2)/49;  %占比，相当于I3=I1(:,3)
[m_r,n_r]=size(R2); %[160,1]
r=[];pr=[]; %for循环后，筛选出图像中存在的亮度级，r装像素亮度级，pr装亮度级对应的占比
for i=1:m_r
    if R3(i,1)~=0
        r(end+1)=R2(i,1);
        pr(end+1)=R3(i,1);
    end
end
figure
subplot(3,1,1);
stem(r,pr);
axis([0 255 0 0.3]);
title("原图像的灰度直方图");
xlabel("r");ylabel("pr");

%参考图像
Z=[10 9 11 12 10 11 10;
    9 10 11 12 11 12 11;
    9 11 10 12 13 14 13;
    12 12 10 13 14 15 13;
    11 13 14 16 16 14 15;
    14 12 12 15 16 15 14;
    13 12 13 16 13 14 15]*11;
Z1=tabulate(Z(:));
Z2=Z1(:,1);
Z3=Z1(:,2)/49;
[m_z,n_z]=size(Z2);
z=[];pz=[];
for i=1:m_z
    if Z3(i,1)~=0
        z(end+1)= Z2(i,1);
        pz(end+1)= Z3(i,1);
    end
end
subplot(3,1,2);
stem(z,pz);
axis([0 255 0 0.3]);
title("参考图像的灰度直方图");
xlabel("z");ylabel("pz");

[m_i1,n_i1]=size(pr);   %返回[1,16]
[m_j1,n_j1]=size(pz);   %返回[1,8]

ratio=0.75;  %控制阈值，如果当前的temp_v还够放90%的temp_r，那就塞进去
v=[];

num_r=1;num_z = 1;  %“指针”初始化，指向第一个位置
temp_r=pr(num_r);temp_v=pz(num_z);  %初始化，r和z都指向第一个填充位
num_r=num_r+1;num_z=num_z+1;    %指针指向下一个位置

while (num_z<=n_j1)
    if(temp_v<(temp_r*ratio))   %z在这个填充位的容量太小，不适合放东西，跳到下一个
        v=[v 0];    %v可以不打分号，看看输出和自己计算的一不一样
        temp_v=temp_v+pz(num_z);
        num_z =num_z+ 1;
    end
    if(temp_v>=(temp_r*ratio))
        if(temp_v==(temp_r*ratio))  %刚好匹配
            v=[v temp_r];
            temp_r=pr(num_r);
            num_r=num_r+1;
            temp_v=pz(num_z);
            num_z =num_z+1;
        else    %这个填充位置容量大，看看能不能多放几个
            
            load=temp_r;
            while(temp_v>=(load*ratio))
                temp_r=load;
                if(num_r<=n_i1) %如果r还有下一位
                    load=temp_r+pr(num_r);
                    num_r=num_r+1;
                end
            end
            num_r=num_r-1;  %加多了，回退一位
            v=[v temp_r];
            temp_v=temp_v-temp_r;
            if(num_r<=n_i1)
                temp_r=pr(num_r);
                num_r=num_r+1;
            end
            temp_v=temp_v+pz(num_z);
            num_z=num_z+1;
        end
        
    end
    
end
v=[v temp_r];


subplot(3,1,3);stem(z,v);
axis([0 255 0 0.3]);
title("原图像经过直方图规定化的灰度直方图");
xlabel("v");ylabel("pv");
