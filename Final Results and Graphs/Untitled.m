clc
clear all
close all
n=[0.74	0.81
0.68	0.7
0.73	0.81
0.73	0.81	
0.73	0.82];
	
	
bar_handle =bar(n,'grouped')
grid on
set(gca,'XTickLabel',{'A','B','C','D','E'},'fontweight','b',...
	'fontsize',10);
ylabel('F1 Score','fontsize',12,'fontweight','b','color','k');
title('Feature Selection: K nearest Neighbours',...
	'fontsize',12,'fontweight','b','color','k')
legend('Spanish','English')
xlabel('Feature Category')
set(bar_handle(1),'FaceColor',[0.5,0.75,1])
set(bar_handle(2),'FaceColor','k')
%txt = 'Week2 Average: 23.0108 Litres';
%text(1.5,52,txt,'fontweight','b','fontsize',9,'BackgroundColor',[1 1 1])
%txt = 'Week1 Average: 13.9588 Litres';
%text(1.5,55,txt,'fontweight','b','fontsize',9,'BackgroundColor',[1 1 1])
