%% Load the data set
data = readtable('Dataset_01.xlsx');
head(data)

%% Encoding the categorical varaibles
Gender = grp2idx(data.Gender);
Married = grp2idx(data.Married);
Education = grp2idx(data.Education);
SelfEmployed = grp2idx(data.SelfEmployed);
PropertyArea = grp2idx(data.PropertyArea);
CreditHistory = grp2idx(data.CreditHistory);

%% Define feature matrix and target vector
X = [Gender Married Education SelfEmployed data.ApplicantIncome data.CoapplicantIncome data.LoanAmount data.LoanAmountTerm CreditHistory PropertyArea];
y = data.LoanStatus;

%% Spliting the dataset into train and test sets
holdOutRatio = 0.2;
cv = cvpartition(length(y),'HoldOut',holdOutRatio);

X_train = X(training(cv),:);
y_train = y(training(cv));

X_test = X(test(cv),:);
y_test = y(test(cv));

%% Fit the model 

T = ind2vec(y_train);
X_train = ind2vec(X_train);
spread = 1;
net = newpnn(X_train,T,spread);


%% Performace evaluation 

% MSE
se = 0;
for k=1:length(y)
    err = y_test(k)-y(k);
    se = se + (err)^2;
end
mse = se/length(y);
%RMSE
rmse = sqrt(mse);

%MAE
ae = 0;
for k=1:length(y)
    err = y_test(k)-y(k);
    ae = ae + abs(err);
end
mae = ae/length(y);

fprintf("MSE of predictions = %0.3f\n",mse);
fprintf("RMSE of predictions = %0.3f\n",rmse);
fprintf("MAE of predictions = %0.3f\n",mae);