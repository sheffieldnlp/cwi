F1_dev_eng = [0.8	0.82	0.8	0.74	0.79	0.9	0.72];
F1_dev_span=[0.76	0.73	0.77	0.74	0.74	0.86	0.71];
F1_test_eng=[0.79	0.8	0.8	0.77	0.76	0.91	0.72];
F1_test_span=[0.77	0.77	0.77	0.74	0.74	0.87	0.7];

subplot(2,1,1)
plot(F1_dev_eng,'--o','LineWidth',2)
set(gca,'xtick',1:7,'xticklabel',{'DT','K-n','RF','AB','ET','HM','LG'})
grid on
hold on
ylim([0.7 1])
plot(F1_test_eng,'--o','LineWidth',2)
legend('Improved Dev English','Improved Test English')


subplot(2,1,2)
plot(F1_dev_span,'--o','LineWidth',2)
set(gca,'xtick',1:7,'xticklabel',{'DT','K-n','RF','AB','ET','HM','LG'})
grid on
hold on
ylim([0.7 1])
plot(F1_test_span,'--o','LineWidth',2)
legend('Improved Dev Spanish','Improved Test Spanish')
