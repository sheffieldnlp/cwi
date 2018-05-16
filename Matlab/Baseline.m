F1_dev_eng = [0.72,0.7,0.73,0.71,0.73,0.79,0.69];
F1_dev_span=[0.72	0.7	0.72	0.72	0.72	0.8	0.72];
F1_test_eng=[0.73	0.74	0.73	0.71	0.73	0.79	0.7];
F1_test_span=[0.71	0.69	0.71	0.71	0.71	0.78	0.71];

subplot(2,1,1)
plot(F1_dev_eng,'--o','LineWidth',2)
set(gca,'xtick',1:7,'xticklabel',{'DT','K-n','RF','AB','ET','HM','LG'})
grid on
hold on
ylim([0.68 0.81])
plot(F1_test_eng,'--o','LineWidth',2)
legend('Baseline Dev English','Baseline Test English')


subplot(2,1,2)
plot(F1_dev_span,'--o','LineWidth',2)
set(gca,'xtick',1:7,'xticklabel',{'DT','K-n','RF','AB','ET','HM','LG'})
grid on
hold on
ylim([0.68 0.81])
plot(F1_test_span,'--o','LineWidth',2)
legend('Baseline Dev Spanish','Baseline Test Spanish')
