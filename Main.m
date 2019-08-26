

path = 'D:/Academics UWM/Spring 2019/CS 567/Final/Retina images/';

feature(:,1) = find_exudates(path);
feature(:,2) = find_hm(path);

label = [0*(1:18) 0*(1:18)+1]';

figure
plot(feature(label==0,1), feature(label==0,2), 'g.', 'markersize', 20)
hold on
plot(feature(label==1,1), feature(label==1,2), 'r.', 'markersize', 20)
hold off




%% Creating groups
num_labels1=randperm(18);
num_labels2=randperm(18)+18;
    
cv_groups = [num_labels1(1:3),num_labels2(1:3);
    num_labels1(4:6),num_labels2(4:6);
    num_labels1(7:9),num_labels2(7:9);
    num_labels1(10:12),num_labels2(10:12);
    num_labels1(13:15),num_labels2(13:15);
    num_labels1(16:18),num_labels2(16:18)];


%% 6-fold CV Logistic regression

pred = zeros(size(label)); % vector to hold predictions

for fold=1:6
  test = feature(cv_groups(fold,:),:);  
  
  train_fold=1:6;
  t=find(train_fold~=fold);
  train = feature(cv_groups(train_fold,:),:);
 
  labels_train = label(cv_groups(train_fold,:));
  
  %Normalization
  nfeat = size(train, 2);
        for n=1:nfeat
            mn_train = mean(train(:,n));
            sd_train = std(train(:,n));
            train(:,n) = (train(:,n)-mn_train)/sd_train;
            test(:,n) = (test(:,n)-mn_train)/sd_train;
        end
  
  ntest = size(test, 1);
  ntrain = size(train, 1);
  pred_test = zeros(1, ntest);
  
  % Train the classifier (logistic model fit
  beta = glmfit(train, label, 'binomial', 'link', 'logit');
  
  % Need to use the inverse logit to get the probabilities for test
  xb = [ones(size(test,1), 1), test]*beta;
  prob_test = exp(xb)./(1+exp(xb));
  pred_test = 1*prob_test>.5;
  
  pred(cv_groups(fold,:)) = pred_test;
end

match = label == pred;
accuracy_healthy_log = mean(match(label == 0))
accuracy_unnhealthy_log = mean(match(label == 1))




%% 6-fold CV Knn
pred = zeros(size(label)); % vector to hold predictions
k = 4;

for fold=1:6
  test = feature(cv_groups(fold,:),:);  
  
  train_fold=1:6;
  t=find(train_fold~=fold);
  train = feature(cv_groups(train_fold,:),:);
 
  labels_train = label(cv_groups(train_fold,:));
  
  %Normalization
  nfeat = size(train, 2);
        for n=1:nfeat
            mn_train = mean(train(:,n));
            sd_train = std(train(:,n));
            train(:,n) = (train(:,n)-mn_train)/sd_train;
            test(:,n) = (test(:,n)-mn_train)/sd_train;
        end
        
  
  ntest = size(test, 1);
  ntrain = size(train, 1);
  pred_test = zeros(1, ntest);
  for i=1:ntest
      dist_from_train = sqrt(sum((ones(ntrain,1)*test(i,:)-train).^2, 2));
      [reord, ord] = sort(dist_from_train);
      knn = labels_train(ord(1:k));
      p_g1 = mean(knn == 0);
      p_g2 = mean(knn == 1);
      if (p_g2<p_g1)
          pred_test(i)=0;
      elseif (p_g1<p_g2)
          pred_test(i)=1;
      else
          pred_test(i)=round(rand); 
      end   
  end
  pred(cv_groups(fold,:)) = pred_test;
end

match = label == pred;
accuracy_healthy_knn = mean(match(label == 0))
accuracy_unnhealthy_knn = mean(match(label == 1))
