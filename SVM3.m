%%
data = readtable('Parts.xlsx');
head(data)

%% 
X = data(:,1:4);
Y = data(:,5);

%%
cv = cvpartition(Y, 'HoldOut', 0.15);
x_train = X(training(cv), :);
y_train = Y(training(cv));

x_test = X(test(cv),:);
y_test = Y(test(cv));
