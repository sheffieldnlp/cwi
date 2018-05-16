%Spanish
%Improved vs %Baseline
samples = [1500,3000,6000,9000,12000,13750];

figure
subplot(2,1,1)

baseline_lr_spanish = [0.7,0.72,0.7,0.7,0.7,0.7];
baseline_RF_spanish = [0.71,0.71,0.7,0.7,0.7,0.7];
baseline_K_spanish = [0.7,0.69,0.69,0.68,0.69,0.68];
baseline_DT_spanish = [0.7,0.71,0.69,0.7,0.7,0.7];
baseline_ET_spanish = [0.72,0.72,0.68,0.7,0.7,0.7];
baseline_Adaboost_Spanish = [0.71,0.72,0.67,0.7,0.7,0.7];

plot(samples, baseline_lr_spanish,'--*','LineWidth',2)
grid on
hold on
ylim([0.65,0.75])
xlim([1000,14000])
plot(samples, baseline_RF_spanish,'--*','LineWidth',2)
plot(samples, baseline_DT_spanish,'--*','LineWidth',2)
plot(samples, baseline_K_spanish,'--*','LineWidth',2)
plot(samples, baseline_ET_spanish,'--*','LineWidth',2)
plot(samples, baseline_Adaboost_Spanish,'--*','LineWidth',2)
legend('Logisitic Regression','Random Forest','K neighbours','Decision Trees','Extra Trees','Adaboost')

xlabel('No of Samples in Training Data')
ylabel('F1 Score')
title('Baseline System Spanish')

subplot(2,1,2)
imp_lr_spanish = [0.72,0.75,0.71,0.75,0.74,0.72];
imp_rf_spanish = [0.75,0.76,0.76,0.77,0.77,0.76];
imp_k_spanish = [0.72,0.72,0.73,0.73,0.72,0.72];
imp_dt_spanish = [0.73,0.73,0.76,0.76,0.76,0.75];
imp_ET_spanish = [0.72,0.73,0.73,0.75,0.74,0.74];
imp_Adahoost_spanish=[0.75,0.75,0.75,0.74,0.75,0.74];
Hybrid_Model_s=[0.87,0.88,0.87,0.87,0.87,0.88];
plot(samples, imp_lr_spanish,'--*','LineWidth',2)
grid on
hold on
ylim([0.65,0.95])
xlim([1000,14000])
plot(samples, imp_rf_spanish,'--*','LineWidth',2)
plot(samples, imp_k_spanish,'--*','LineWidth',2)
plot(samples, imp_dt_spanish,'--*','LineWidth',2)
plot(samples, imp_ET_spanish,'--*','LineWidth',2)
plot(samples, imp_Adahoost_spanish,'--*','LineWidth',2)
plot(samples, Hybrid_Model_s,'--*','LineWidth',2)
legend('Logisitic Regression','Random Forest','K neighbours','Decision Trees','Extra Trees','Adaboost','Hybrid Model')

xlabel('No of Samples in Training Data')
ylabel('F1 Score')
title('Improved System Spanish')
