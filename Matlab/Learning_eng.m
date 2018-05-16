%English
%Improved vs %Baseline
samples = [5000,10000,15000,20000,25000,27299];


figure
subplot(2,1,1)

baseline_lr = [0.68,0.67,0.68,0.68,0.68,0.69];
baseline_RF = [0.71,0.71,0.71,0.72,0.72,0.73];
baseline_K= [0.69,0.69,0.7,0.7,0.7,0.7];
baseline_DT = [0.71,0.71,0.71,0.72,0.72,0.72];
baseline_ET= [0.71,0.71,0.71,0.72,0.72,0.73];
baseline_Adaboost = [0.71,0.71,0.7,0.7,0.71,0.71];

plot(samples, baseline_lr,'--*','LineWidth',1)
grid on
hold on
ylim([0.65,0.75])
xlim([1000,30000])
plot(samples, baseline_RF,'--*','LineWidth',1)
plot(samples, baseline_DT,'--*','LineWidth',1)
plot(samples, baseline_K,'--*','LineWidth',1)
plot(samples, baseline_ET,'--*','LineWidth',1)
plot(samples, baseline_Adaboost,'--*','LineWidth',1)
legend('Logisitic Regression','Random Forest','K neighbours','Decision Trees','Extra Trees','Adaboost')

xlabel('No of Samples in Training Data')
ylabel('F1 Score')
title('Baseline System English')

subplot(2,1,2)
imp_lr = [0.69,0.69,0.7,0.7,0.7,0.71];
imp_rf = [0.77,0.79,0.79,0.8,0.8,0.8];
imp_k = [0.73,0.79,0.8,0.81,0.82,0.82];
imp_dt = [0.77,0.78,0.79,0.79,0.79,0.8];
imp_ET = [0.76,0.77,0.78,0.78,0.78,0.78];
imp_Adahoost_spanish=[0.76,0.76,0.76,0.76,0.75,0.75];
Hybrid_Model_s=[0.89,0.89,0.9,0.91,0.91,0.91];

plot(samples, imp_lr_spanish,'--*','LineWidth',1)
grid on
hold on
ylim([0.65,1])
xlim([1000,30000])
plot(samples, imp_rf_spanish,'--*','LineWidth',1)
plot(samples, imp_k_spanish,'--*','LineWidth',1)
plot(samples, imp_dt_spanish,'--*','LineWidth',1)
plot(samples, imp_ET_spanish,'--*','LineWidth',1)
plot(samples, imp_Adahoost_spanish,'--*','LineWidth',1)
plot(samples, Hybrid_Model_s,'--*','LineWidth',1)
legend('Logisitic Regression','Random Forest','K neighbours','Decision Trees','Extra Trees','Adaboost','Hybrid Model')

xlabel('No of Samples in Training Data')
ylabel('F1 Score')
title('Improved System English')
