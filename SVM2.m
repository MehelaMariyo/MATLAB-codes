%% Load the dataset
load ionosphere

%% Define 'g' as possitive class
negativeClass = 'b';
possitiveClass = 'g';
holdOut_ratio = 0.15;

%%
x_stand = zscore(X);    


%% Split the dataset into train and test
cv = cvpartition(Y, 'HoldOut', holdOut_ratio);
x_train = x_stand(training(cv), :);
y_train = Y(training(cv));

x_test = x_stand(test(cv),:);
y_test = Y(test(cv));

%%
svmStruct = fitcsvm(x_train, y_train, 'KernelFunction','linear','ClassNames',{negativeClass, possitiveClass});

%%
predictions = predict(svmStruct, x_test);

%% 
predictions = categorical(predictions, {negativeClass, possitiveClass});
y_test = categorical(y_test, {negativeClass, possitiveClass});


%%
cm = confusionmat(y_test, predictions);
disp(cm);
