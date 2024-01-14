coeff_1 = reshape(coeff_lms1.', 30, []).';
coeff_2 = reshape(coeff_lms2.', 30, []).';
cooef = [coeff_1; coeff_2];
error = [err_lms1'; err_lms2'];


sig_tr_mag = abs(sig_tr); 
sig_tr_angle = angle(sig_tr); 
sig_tr_ = [sig_tr_mag, sig_tr_angle];

error_mag = abs(error); 
error_angle = angle(error); 
error_ = [error_mag, error_angle];

cooef_mag = abs(cooef); 
cooef_angle = angle(cooef); 
cooef = [cooef_mag, cooef_angle];

% Generate some sample data
XTrain = [sig_tr_(1:3000,:), error_(1:3000,:)];
YTrain = [cooef(1:3000,:)];
XTest = [sig_tr_(3001:4000,:), error_(3001:4000,:)];
YTest = [cooef(3001:4000,:)];

% Save the data to a file
save('data.mat', 'XTrain', 'YTrain', 'XTest', 'YTest');
