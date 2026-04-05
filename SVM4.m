%%
load fisheriris

%%
X = meas;
Y = species;
testPropotion = 0.15;

%%
cv = cvpartition(Y, 'HoldOut', testPropotion);
x_train = X(training(cv), :);
y_train = Y(training(cv));

x_test = X(test(cv),:);
y_test = Y(test(cv));

%%
kernelList = {'linear','ploy','rbf','sigmoid'};

%%
accuracies = zeros(size(kernelList));

%%
for k=1:length(kernelList)
    kernel = kernelList(k);
    
    SVModel1 = fitcecoc(x_train,y_train);
    
    cvModell = crossval(SVModel1);
    cvAccuracies = 1 - kfoldLoss(cvModell, 'LossFun', 'ClassifError');
    
    accuracies(k) = mean(cvAccuracies);
    accuracies(k)*100
end
