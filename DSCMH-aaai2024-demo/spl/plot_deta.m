% P=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];
% q=[0:0.01:4];
% tau=0.5;
% for i=1:length(P)
%     y=[];
%     x=[0:1:length(q)];
%     for j=1:length(q)
%         y_tmp=(P(i))^(q(j)-1)/tau;
%         y=[y,y_tmp];
%         
%     end
% %     xlim([0 4])
%     plot(q,y)
%     hold on
% end
% legend( 'P^+=0.1', 'P^+=0.2', 'P^+=0.3', 'P^+=0.4', 'P^+=0.5', 'P^+=0.6', 'P^+=0.7', 'P^+=0.8', 'P^+=0.9');
% set(gca,'FontSize',15); % 设置文字大小，同时影响坐标轴标注、图例、标题等。
% xlabel('q','FontSize',20);
% ylabel('\Delta','rotation',0,'FontSize',20);

% q=[0.5,0.8,1,2,4];
% P=[0:0.01:1];
% tau=0.5;
% for i=1:length(q)
%     y=[];
%     x=[0:1:length(P)];
%     for j=1:length(P)
%         y_tmp=(P(j))^(q(i)-1)/tau;
%         y=[y,y_tmp];
%         
%     end
% %     xlim([0 4])
%     plot(P,y)
%     hold on
% end
% legend( 'q=0.5', 'q=0.8', 'q=1', 'q=2', 'q=4');
% 
% set(gca,'FontSize',15); % 设置文字大小，同时影响坐标轴标注、图例、标题等。
% 
% xlabel('P^+','FontSize',25);
% ylabel('\Delta','rotation',0,'FontSize',25);



q=[0.5,2];
P=[0:0.01:1];
tau=0.5;
for i=1:length(q)
    y=[];
    x=[0:1:length(P)];
    for j=1:length(P)
        y_tmp=(1-P(j)^q(i))/q(i);
        y=[y,y_tmp];
        
    end
%     xlim([0 4])
    plot(P,y)
    hold on
end
% legend( 'q=0.5', 'q=0.8', 'q=1', 'q=2', 'q=4');

set(gca,'FontSize',15); % 设置文字大小，同时影响坐标轴标注、图例、标题等。

xlabel('P^+','FontSize',20);
ylabel('loss','rotation',0,'FontSize',20);