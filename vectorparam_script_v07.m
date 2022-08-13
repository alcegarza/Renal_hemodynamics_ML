% author: Rocío ÁlvarezCedrón
% date: 2022.01.26 v.01
% date: 2022.01.29 v.02
% date: 2022.02.01 v.03
% date: 2022.02.19 v.04
% date: 2022.02.21 v.05 --> we are going to try the complex poles first
% (only twice and then the real poles 8 times)
% date:2022.02.21 v.06 --> loop to obtain error_list with all the errors of
% the different files of recordings
% date: 2022.04.04 v.07 --> read the excel with the list of the recordings
% we want, and compute the vector parameter and save it

clear all, close all
disp(['Principio'])
%% Pole location

T = 1/2; %sampling rate 2 Hz

%complex conjugate poles
%sampling frequency and damping ratio
w1 = 0.22*2*pi;  d_ratio1 = 0.14;
w2 = 0.17*2*pi;  d_ratio2 = 0.23;
w3 = 0.15*2*pi; d_ratio3 = 0.32;
%poles conjugate pair
z1 = exp(T*(-d_ratio1*w1 + 1i*w1*sqrt(1-d_ratio1^2)));
z2 = exp(T*(-d_ratio2*w2 + 1i*w2*sqrt(1-d_ratio2^2)));
z3= exp(T*(-d_ratio3*w3 + 1i*w3*sqrt(1-d_ratio3^2)));

%real poles
%sampling frequency and damping ratio
w4 = 0.035*2*pi;  d_ratio4 = 1;
w5 = 0.024*2*pi;  d_ratio5 = 1;
w6 = 0.027*2*pi;  d_ratio6 = 1;
%poles
z4 = exp(T*(-d_ratio4*w4 + 1i*w4*sqrt(1-d_ratio4^2)));
z5 = exp(T*(-d_ratio5*w5 + 1i*w5*sqrt(1-d_ratio5^2)));
z6 = exp(T*(-d_ratio6*w6 + 1i*w6*sqrt(1-d_ratio6^2)));


%plot poles
figure
plot([z1 z2 z3 z4 z5 z6], 'x')
zgrid()
axis([0 1.2 -0.1 1])
clear w1 w2 w3 w4 w5 w6 ....
    d_ratio1 d_ratio2 d_ratio3 d_ratio4 d_ratio5 d_ratio6



%% Average poles

pr = mean([z4 z5 z6]); %real pole
pc = mean([z1 z2 z3]); %complex pole

clear z1 z2 z3 z4 z5 z6


%% Subsampling
% 30 min recording at 200Hz
% Need to subsample at 2Hz

%% read file
fid=fopen('files.txt');
tline = fgetl(fid);
tlines = cell(0,1);
nofile = [];
while ischar(tline)
    tlines{end+1,1} = tline;
    tline = fgetl(fid);
    if(tline>0)
        files = dir(strcat('database/', tline,'*.mat'));
    end 
    a = size(files);
    tline2=tline;
    if a(1) == 0
        b=size(tline);
        if b(2) < 15
            tline2= strcat(tline,'!');
        end
        if b(2) > 15
            tline2= tline(1:15);
        end
        if b(2) > 1
        nofile = [nofile;tline2];
        end
    end 

    for index = 1:size(files)
        load(fullfile('database',files(index).name))



        %%
        %load('database/D200_2012_01_27_1749.mat')
        t= [0:359999]/(60*200);
        t1=t(1:100:length(t));

        bp_20Hz = decimate(double(bp),10);
        %bp_2Hz = decimate(bp_20Hz,10);
        bp_2Hz = decimate(bp_20Hz,10);
        bp_2Hz = bp_2Hz - mean(bp_2Hz);

        rbf_20Hz = decimate(double(rbf),10);
        rbf_2Hz = decimate(rbf_20Hz,10);
        rbf_2Hz = rbf_2Hz - mean(rbf_2Hz);

        % delete the variables that are not useful
        clear SSARI_value %rbf_20Hz bp_20Hz






        %% vector parameter

        %x = [1 , zeros(1,3599)]; %impulse signal to check orthonormality
        x = bp_2Hz; %input signal

        % System

        %complex pair pole
        alpha11 = sqrt( (1+pc)*(1+conj(pc))*(1-pc*conj(pc)) / 2);
        alpha21 = sqrt( (1-pc)*(1-conj(pc))*(1-pc*conj(pc)) / 2);

        %complex pair pole tf
        bc = [abs(pc)^2 -2*real(pc) 1; 0 alpha11 -alpha11; 0 alpha21 alpha21];
        ac = [1 -2*real(pc) abs(pc)^2];

        %complex pair pole signal
        Hc1 = tf(bc(1,:) , ac,T);
        Hc2 = tf(bc(2,:) , ac,T);
        Hc3 = tf(bc(3,:) , ac,T);

        y1c = lsim(Hc1,x);
        x1c = lsim(Hc2,x);
        x2c = lsim(Hc3,x);

        y2c = lsim(Hc1,y1c);
        x3c = lsim(Hc2,y1c);
        x4c = lsim(Hc3,y1c);

        %real pole
        br = [pr -1; 0 sqrt(1-pr^2)];
        ar = [1 -pr];

        %real pole tf
        Hr1 = tf([br(1,:)] , ar,T);
        Hr2 = tf([br(2,:)] , ar,T);

        %real pole signal
        y1r = lsim(Hr1,y2c);
        x1r = lsim(Hr2,y2c);

        y2r = lsim(Hr1,y1r);
        x2r = lsim(Hr2,y1r);


        y3r = lsim(Hr1,y2r);
        x3r = lsim(Hr2,y2r);

        y4r = lsim(Hr1,y3r);
        x4r = lsim(Hr2,y3r);

        y5r = lsim(Hr1,y4r);
        x5r = lsim(Hr2,y4r);

        y6r = lsim(Hr1,y5r);
        x6r = lsim(Hr2,y5r);

        y7r = lsim(Hr1,y6r);
        x7r = lsim(Hr2,y6r);

        y8r = lsim(Hr1,y7r);
        x8r = lsim(Hr2,y7r);

        a =[x1c  x2c x3c x4c x1r x2r  x3r x4r x5r  x6r  x7r x8r];
        A = transpose(a)*a;

        %corr = corrcoef([x1r  x2r  x3r x4r  x1c  x2c x3c x4c x5c x6c x7c x8c]);


        %%
        inputs = [x1c  x2c x3c x4c x1r x2r  x3r x4r x5r x6r x7r x8r];


        s_order = [];
        t_order = [];

        for i = 1:12
            for j = i:12
                s_order = [s_order  inputs(1:3600,i).*inputs(1:3600,j)];
                for k = j:12
                    t_order = [t_order  inputs(1:3600,i).*inputs(1:3600,j).*inputs(1:3600,k)];
                end
            end
        end
        z_order = bp_2Hz';
        z_order_ones = ones(3600,1);
        phi = [z_order z_order_ones inputs s_order t_order]';

        %theta_0 = randn(456,1);
        %rbf_true = theta_0'*phi;

        theta = (phi * phi') \ (phi * rbf_2Hz');

        save(['15/', files(index).name],'theta')
        
        %theta = (phi * phi') \ (phi * rbf_true');

        %check = norm(theta - theta_0)


        rbf_estimate = phi' * theta;





        error = sum((rbf_2Hz - rbf_estimate').^2)/sum(rbf_2Hz.^2); %error energy
        disp(['Error energy complex first: ',num2str(error)])

    end
end

fclose(fid);

%%
save('nofiles_1.mat','nofile')