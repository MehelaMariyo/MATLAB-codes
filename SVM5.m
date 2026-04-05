%%
load ionosphere

%%
X = zscore(X);  
negativeClass = 'b';
possitiveClass = 'g';
testPartion = 0.2;

%% 
cv = cvpartition(Y,'HoldOut',testPartion);
x_train = X(training(cv), :);
y_train = Y(training(cv));

x_test = X(test(cv),:);
y_test = Y(test(cv));

%%
svmStruct = fitcsvm(x_train, y_train, 'KernelFunction','rbf','ClassNames',{negativeClass, possitiveClass});

%%
predictions = predict(svmStruct, x_test);

%% 
predictions = categorical(predictions, {negativeClass, possitiveClass});
y_test = categorical(y_test, {negativeClass, possitiveClass});


%%
cm = confusionmat(y_test, predictions);
disp(cm);
cf = confusionchart(y_test,predictions);