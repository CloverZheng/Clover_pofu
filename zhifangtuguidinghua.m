%ԭʼͼ�����ص�������ȼ�Ϊ160
R=[1 7 8 9 10 11 14;
    5 2 6 7 14 12 15;
    3 4 7 8 6 9 11;
    2 1 4 7 8 8 9;
    8 4 5 9 11 12 10;
    8 10 11 15 16 10 13;
    13 6 9 16 13 12 10]*10;
%ͳ�����ؾ����и����س��ֵĴ���
R1=tabulate(R(:));  %���أ�����ֵ ���ִ��� ռ��%
R2=R1(:,1); %I1�ĵ�һ�У�0��160
R3=R1(:,2)/49;  %ռ�ȣ��൱��I3=I1(:,3)
[m_r,n_r]=size(R2); %[160,1]
r=[];pr=[]; %forѭ����ɸѡ��ͼ���д��ڵ����ȼ���rװ�������ȼ���prװ���ȼ���Ӧ��ռ��
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
title("ԭͼ��ĻҶ�ֱ��ͼ");
xlabel("r");ylabel("pr");

%�ο�ͼ��
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
title("�ο�ͼ��ĻҶ�ֱ��ͼ");
xlabel("z");ylabel("pz");

[m_i1,n_i1]=size(pr);   %����[1,16]
[m_j1,n_j1]=size(pz);   %����[1,8]

ratio=0.75;  %������ֵ�������ǰ��temp_v������90%��temp_r���Ǿ�����ȥ
v=[];

num_r=1;num_z = 1;  %��ָ�롱��ʼ����ָ���һ��λ��
temp_r=pr(num_r);temp_v=pz(num_z);  %��ʼ����r��z��ָ���һ�����λ
num_r=num_r+1;num_z=num_z+1;    %ָ��ָ����һ��λ��

while (num_z<=n_j1)
    if(temp_v<(temp_r*ratio))   %z��������λ������̫С�����ʺϷŶ�����������һ��
        v=[v 0];    %v���Բ���ֺţ�����������Լ������һ��һ��
        temp_v=temp_v+pz(num_z);
        num_z =num_z+ 1;
    end
    if(temp_v>=(temp_r*ratio))
        if(temp_v==(temp_r*ratio))  %�պ�ƥ��
            v=[v temp_r];
            temp_r=pr(num_r);
            num_r=num_r+1;
            temp_v=pz(num_z);
            num_z =num_z+1;
        else    %������λ�������󣬿����ܲ��ܶ�ż���
            
            load=temp_r;
            while(temp_v>=(load*ratio))
                temp_r=load;
                if(num_r<=n_i1) %���r������һλ
                    load=temp_r+pr(num_r);
                    num_r=num_r+1;
                end
            end
            num_r=num_r-1;  %�Ӷ��ˣ�����һλ
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
title("ԭͼ�񾭹�ֱ��ͼ�涨���ĻҶ�ֱ��ͼ");
xlabel("v");ylabel("pv");
