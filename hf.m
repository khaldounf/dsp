clear all;
close all;
clc;


Rs = 10; % symbol rate kHz
sps = 1; % digital samples per symbol
Fs = sps*Rs; % samplig rate
T = 1/Fs; % sample_time, ms

% num_rays = 2; % 2 rays
% s = [80 100]; % us/MHz
% rays_delay = [0 0.015]; % ms
% rays_delay_samples = round(rays_delay./T);

doppler_spread = 1.0; %Hz;
HF_Chan_1 = stdchan('iturHFMM', Fs * 1.0e+3, doppler_spread);
release(HF_Chan_1);
set(HF_Chan_1, 'PathGainsOutputPort', true);
max_delay_samples = max(HF_Chan_1.PathDelays)/(T/1000);


num_packets = 100;
num_training_syms = 1000;
num_data_syms = 3000;
num_packet_syms = num_training_syms + num_data_syms;
% number of generated symbol and bits
signal_length_ms = (num_data_syms/Rs);

% 1 - BPSK, 2 - QPSK, 3 - 8-PSK, 4 - QAM-16
modulation_type = 2;

% snr
EbN0dB = 20;
EbN0 = 10.^(EbN0dB/10);

% bps = bit per symbol
switch (modulation_type)
    case 1  
        bps = 1;
    case 2
        bps = 2;
    case 3
        bps = 3;
    case 4
        bps = 4;
end

EsN0 = EbN0*bps;
EsN0dB = 10*log10(EsN0);

M = 2^bps;


% number of generated bits
num_bits = bps * num_packet_syms;

% bit data to transmit
bit_data_tr = randi([0, 1], num_bits, 1);
% bps-bit symbols to transmit
syms_data_tr = reshape(bit_data_tr, bps, num_packet_syms );

switch (modulation_type)
    case 1
        temp1 = (0:M-1)';
        temp2 = pskmod(temp1, M);
        Es = mean(abs(temp2).^2);

        sig_tr = pskmod(syms_data_tr, M, 'InputType', 'bit');
    case 2
        temp1 = (0:M-1)';
        temp2 = pskmod(temp1, M, pi/4);
        Es = mean(abs(temp2).^2);

        sig_tr = pskmod(syms_data_tr, M, pi/4, 'InputType', 'bit');
    case 3
        temp1 = (0:M-1)';
        temp2 = pskmod(temp1, M, pi/8);
        Es = mean(abs(temp2).^2);

        sig_tr = pskmod(syms_data_tr, M, pi/8, 'InputType', 'bit');
    case 4

        temp1 = (0:M-1)';
        temp2 = qammod(temp1, M);
        Es = mean(abs(temp2).^2);

        sig_tr = qammod(syms_data_tr, M, 'InputType', 'bit');
end


N0 = Es/(EsN0);  %white noise noise power density (one sided)
sigma_noise = sqrt(N0/2);  % noise stddev 
sig_tr = sig_tr.';
scatterplot(sig_tr);
title('Pure constellation')

%% сигнал + шум
noise = sigma_noise*(randn(size(sig_tr)) + 1i*randn(size(sig_tr)));

sig_rec_awgn = sig_tr + noise;
scatterplot(sig_rec_awgn);
title('Constellation in AWGN')


%%  сигнал в канале + шум
[sig_rec_chan, chan_path_gains] = HF_Chan_1(sig_tr);


% h_chan = [1, [(randn(1,3) + 1i*randn(1,3))*0.2]];
% h_chan = sort(h_chan,   'descend');
% h_chan = upsample(h_chan, sps);
% n = (0:length(h_chan)-1)';
% 
% % power normalizing 
mean_gain = sum(mean(abs(chan_path_gains).^2));
chan_path_gains = chan_path_gains/sqrt(mean_gain);

h_chan = [chan_path_gains(1,1), zeros(1, max_delay_samples-2), chan_path_gains(1,2), zeros(1, max_delay_samples-2)];
n = (0:length(h_chan)-1)';
% power normalizing for plot
% h_chan = h_chan/abs(sum(h_chan));

figure
zplane(h_chan,1);

Nfft =  2^14;
h_chan_sp = fftshift(fft(h_chan, Nfft));
f = (-Nfft/2:Nfft/2-1)'*Fs/Nfft;
figure
plot(f, 20*log10(abs(h_chan_sp)), 'LineWidth', 2);
grid on;
xlabel('f, kHz');
ylabel('Channel response, dB');

figure
stem(n, abs(h_chan), 'LineWidth', 2);
title('Channel pulse responce h(n)');
grid on;
xlabel('n');
ylabel('h(n)');
% sum(mean(abs(chan_path_gains).^2))
% 
% figure
% stem(n, abs(h_chan), 'LineWidth', 2);
% title('Channel pulse responce h(n)');
% grid on;
% xlabel('n');
% ylabel('h(n)');
% 
% sig_rec_chan = conv(sig_tr, h_chan);
% sig_rec_channel = sig_rec_chan(1:length(sig_tr));

v1 = var(sig_tr);
v2 = var(sig_rec_chan);

noise2 = sigma_noise*(randn(size(sig_rec_chan)) + 1i*randn(size(sig_rec_chan)));

sig_rec_chan_noise = sig_rec_chan + noise2;
scatterplot(sig_rec_chan_noise);
title('Constellation after channel and AWGN')
%% LMS
% sig_tr = sig_tr.';

eqlms = comm.LinearEqualizer( ...
    'Algorithm', 'LMS', ...
    'NumTaps', 3*max_delay_samples, ...
    'StepSize', 0.05, ...
    'InputSamplesPerSymbol', 1,...,
    'Constellation', temp2.', ...
    'TrainingFlagInputPort',true,...
    'ReferenceTap', 1.5*max_delay_samples);

eqrls = comm.LinearEqualizer( ...
    'Algorithm', 'RLS', ...
    'NumTaps', 3*max_delay_samples, ...
    'ForgettingFactor', 0.97,...
    'InputSamplesPerSymbol', 1,...,
    'Constellation', temp2.', ...
    'TrainingFlagInputPort',true,...
    'ReferenceTap', 1.5*max_delay_samples);


release(eqlms);
mxStep = maxstep(eqlms, sig_rec_chan);
set(eqlms, 'StepSize', 0.5*mxStep);
% num_samples_to_train = 500;
% train_flag = (((1:length(sig_tr))') <= num_samples_to_train);


eqdfelms = comm.DecisionFeedbackEqualizer( ...
    'Algorithm', 'LMS', ...
    'NumForwardTaps', 2*max_delay_samples, ...
    'NumFeedbackTaps', max_delay_samples, ...
    'StepSize', 0.05, ...
    'InputSamplesPerSymbol', 1,...,
    'Constellation', temp2.', ...
    'TrainingFlagInputPort',true,...
    'ReferenceTap', max_delay_samples);

release(eqdfelms);
mxStep = maxstep(eqdfelms, sig_rec_chan);

set(eqdfelms, 'StepSize', 0.5*mxStep);
set(eqdfelms, 'AdaptAfterTraining', true);

eqdferls = comm.DecisionFeedbackEqualizer( ...
    'Algorithm', 'RLS', ...
    'NumForwardTaps', 2*max_delay_samples, ...
    'NumFeedbackTaps', max_delay_samples, ...
    'ForgettingFactor', 0.97,...
    'InputSamplesPerSymbol', 1,...,
    'Constellation', temp2.', ...
    'TrainingFlagInputPort',true,...
    'ReferenceTap', max_delay_samples);

release(eqdferls);
set(eqdferls, 'AdaptAfterTraining', true);

input_data = sig_rec_chan_noise;
train_data = sig_tr;

equalized_data_lms1 = zeros(length(num_training_syms), 1);
err_lms1 = zeros(length(num_training_syms), 1);
% coeff_lms1 = zeros(length(num_training_syms)*3*max_delay_samples, 1);
% size(coeff_lms1)

equalized_data_lms2 = zeros(length(num_training_syms), 1);
err_lms2 = zeros(length(num_training_syms), 1);
% coeff_lms2 = zeros(length(num_training_syms)*3*max_delay_samples, 1);

equalized_data_rls1 = zeros(length(num_training_syms), 1);
err_rls1 = zeros(length(num_training_syms), 1);
equalized_data_rls2 = zeros(length(num_training_syms), 1);
err_rls2 = zeros(length(num_training_syms), 1);

equalized_data_dfelms1 = zeros(length(num_training_syms), 1);
err_dfelms1 = zeros(length(num_training_syms), 1);
equalized_data_dfelms2 = zeros(length(num_training_syms), 1);
err_dfelms2 = zeros(length(num_training_syms), 1);

equalized_data_dferls1 = zeros(length(num_training_syms), 1);
err_dferls1 = zeros(length(num_training_syms), 1);
equalized_data_dferls2 = zeros(length(num_training_syms), 1);
err_dferls2 = zeros(length(num_training_syms), 1);


for j1 = 1:num_training_syms
    [ttt1,ttt2, ttt3 ]  = eqlms(input_data(j1), train_data(j1), 1);
    equalized_data_lms1(j1) = ttt1;
    err_lms1(j1) = ttt2;
%     size(ttt3)
% op1=j1 * (3*max_delay_samples) - (3*max_delay_samples) +1
% op2=j1+(3*max_delay_samples)-1
    coeff_lms1(j1 * (3*max_delay_samples) - (3*max_delay_samples) +1 :(j1 * (3*max_delay_samples) - (3*max_delay_samples) +1)+3*max_delay_samples-1) = ttt3';
    

     [ttt1,ttt2, ttt3 ]  = eqrls(input_data(j1), train_data(j1), 1);
    equalized_data_rls1(j1) = ttt1;
    err_rls1(j1) = ttt2;
%     wts1(j1) = tt3;

    [ttt1,ttt2, ttt3 ]  = eqdfelms(input_data(j1), train_data(j1), 1);
    equalized_data_dfelms1(j1) = ttt1;
    err_dfelms1(j1) = ttt2;

     [ttt1,ttt2, ttt3 ]  = eqdferls(input_data(j1), train_data(j1), 1);
    equalized_data_dferls1(j1) = ttt1;
     err_dferls1(j1) = ttt2;
end

for j1 = 1:num_data_syms
    [ttt1,ttt2, ttt3]  = eqlms(input_data(j1+num_training_syms), train_data(j1+num_training_syms), 0);
    equalized_data_lms2(j1) = ttt1;
    err_lms2(j1) = ttt2;
    coeff_lms2(j1 * (3*max_delay_samples) - (3*max_delay_samples) +1 :(j1 * (3*max_delay_samples) - (3*max_delay_samples) +1)+3*max_delay_samples-1) = ttt3';

     [ttt1,ttt2, ttt3]  = eqrls(input_data(j1+num_training_syms), train_data(j1+num_training_syms), 0);
    equalized_data_rls2(j1) = ttt1;
    err_rls2(j1) = ttt2;

     [ttt1,ttt2, ttt3]  = eqdfelms(input_data(j1+num_training_syms), train_data(j1+num_training_syms), 0);
    equalized_data_dfelms2(j1) = ttt1;
    err_dfelms2(j1) = ttt2;

     [ttt1,ttt2, ttt3]  = eqdferls(input_data(j1+num_training_syms), train_data(j1+num_training_syms), 0);
    equalized_data_dferls2(j1) = ttt1;
    err_dferls2(j1) = ttt2;

end


% 
% input_data = sig_rec_chan(num_training_syms+1:end);
% train_data = sig_tr(num_training_syms+1:end);
% % 
% % release(eqlms);
% [equalized_data2, err2, wts2]  = eqlms(input_data, train_data, 0);


eqInfo = info(eqlms);
lat1 = eqInfo.Latency;

% sig_rec_chan_equalized = eqlms(input_data, train_data);
tx = (0:length(sig_tr())-1)'*T*sps;

% equalized_data = circshift(equalized_data, -lat1);
% 
% figure
% plot(1:num_samples_to_train, real(train_data), 1:num_samples_to_train, real(equalized_data) , 'LineWidth' , 2);

figure
plot(tx/T, abs(chan_path_gains(:,1)), tx/T, abs(chan_path_gains(:,2)), 'LineWidth', 2);
xlabel('Symbols')
ylabel('Channel path gains')
legend({'First ray', 'Second ray'});
grid on;

figure
plot(tx/T, abs(chan_path_gains(:,1)).^2 + abs(chan_path_gains(:,2)).^2, 'LineWidth', 2);
xlabel('Symbols')
ylabel('Overall power')
legend({'Overall power'});
grid on;

scatterplot(equalized_data_lms2);
title('Constellation after LMS Linear')
scatterplot(equalized_data_rls2);
title('Constellation after RLS Linear')

scatterplot(equalized_data_dfelms2);
title('Constellation after LMS DFE')
scatterplot(equalized_data_dferls2);
title('Constellation after RLS DFE')

figure
plot(tx/T, abs([err_lms1 err_lms2]), tx/T, abs([err_rls1 err_rls2]), tx/T, abs([err_dfelms1 err_dfelms2]), tx/T, abs([err_dferls1 err_dferls2]), 'LineWidth', 2);
xlabel('Symbols')
ylabel('Error Magnitude')
title('Equalizer Error Signal')
legend({'LMS Linear', 'RLS Linear', 'LMS DFE', 'RLS DFE'})
grid on;

% 
% figure
% plot(1:num_data_syms, real(train_data(num_training_syms+1:end)), 1:num_data_syms, real(equalized_data2) , 'LineWidth' , 2);
% figure
% plot(1:num_samples_to_train, real(input_data), 1:num_samples_to_train, real(train_data), 1:num_samples_to_train, real(equalized_data) , 'LineWidth' , 2);

% figure
% plot(tx(1:100), real(sig_tr(1:100)), tx(1:100), real(sig_rec_chan_equalized(1:100)) , 'LineWidth' , 2);
% 
% figure
% plot(tx(1:100*sps)/sps, real(sig_rec_chan(1:100*sps)) , 'LineWidth' , 2);

%% packet_transmission
% 
% for j1 =  1:num_packets
%     % bit data to transmit
%     bit_data_tr = randi([0, 1], num_bits, 1);
%     % bps-bit symbols to transmit
%     syms_data_tr = reshape(bit_data_tr, bps, num_packet_syms );
%     
%     switch (modulation_type)
%         case 1
%             temp1 = (0:M-1)';
%             temp2 = pskmod(temp1, M);
%             Es = mean(abs(temp2).^2);
%     
%             sig_tr = pskmod(syms_data_tr, M, 'InputType', 'bit');
%         case 2
%             temp1 = (0:M-1)';
%             temp2 = pskmod(temp1, M, pi/4);
%             Es = mean(abs(temp2).^2);
%     
%             sig_tr = pskmod(syms_data_tr, M, pi/4, 'InputType', 'bit');
%         case 3
%             temp1 = (0:M-1)';
%             temp2 = pskmod(temp1, M, pi/8);
%             Es = mean(abs(temp2).^2);
%     
%             sig_tr = pskmod(syms_data_tr, M, pi/8, 'InputType', 'bit');
%         case 4
%     
%             temp1 = (0:M-1)';
%             temp2 = qammod(temp1, M);
%             Es = mean(abs(temp2).^2);
%     
%             sig_tr = qammod(syms_data_tr, M, 'InputType', 'bit');
%     end
% 
%     if(j1 == 1)
%         
%     end
% 
% end

% [y, err, weights] = eq(rx,tx(1:numTrainingSymbols));