clc;
% clearvars
% close all
dir = "home/nathangray/PycharmProjects/dopf_trans/gridlabd/30_ders/output";
sub_power = import_sub_power(dir + "/sub_power.csv");

inv = [3, 9, 15, 20, 23, 25, 30, 37, 44, 49, 50, 51, 56, 59, 62, 65, 67, 78, 88, 93, 95, 99, 101, 102, 103, 107, 110, 119, 120, 122];
data = 0;
for i=1:length(inv)
    data = data - import_3ph_data(dir + "/inv_"+inv(i)+"_meter.csv");
end
gen = real(data(:, 1) + data(:, 2) + data(:, 3));
load = ones(size(sub_power))*3570000;

loss = (sub_power + gen - load)/1e3;

% nocomm = (nocomm + gen - load)/1e3;

t = 0:1:57;
plot(t,[loss],'MarkerSize',8,'LineWidth',2);
% createfigure_3lines_powerloss(t, [n1_losses0, n1_losses1, n1_losses2]);
% createfigure_3lines_powerloss(t, [n2_losses0, n2_losses1, n2_losses2]);
% magnifyOnFigure;