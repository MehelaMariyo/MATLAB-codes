%% Load the data set
data = readtable('Dataset_02.xlsx');
head(data)

%% Encoding the categorical varaibles
Weather = grp2idx(data.Weather);
Traffic_Level = grp2idx(data.Traffic_Level);
Time_of_Day = grp2idx(data.Time_of_Day);
Vehicle_Type = grp2idx(data.Vehicle_Type);
Courier_Experience_yrs = grp2idx(data.Courier_Experience_yrs);

%% Define feature matrix and target vector
X = [data.Distance_km Weather Traffic_Level Time_of_Day Vehicle_Type data.Preparation_Time_min Courier_Experience_yrs];
y = data.Delivery_Time_min;

vars = {'Distance_km' 'Weather' 'Traffic_Level' 'Time_of_Day' 'Vehicle_Type' 'Preparation_Time_min' 'Courier_Experience_yrs'};
catVars = {'Weather' 'Traffic_Level' 'Time_of_Day' 'Vehicle_Type' 'Courier_Experience_yrs'};

%% Spliting the dataset into train and test sets
holdOutRatio = 0.2;
cv = cvpartition(length(y),'HoldOut',holdOutRatio);

X_train = X(training(cv),:);
y_train = y(training(cv));

X_test = X(test(cv),:);
y_test = y(test(cv));

%% Fit the model 

model_DT = fitrtree(X_train,y_train, 'PredictorNames',vars,'CategoricalPredictors',catVars);
y_pred = predict(model_DT, X_test);
view(model_DT, 'mode', 'graph')

%% Performace evaluation 
% MSE
se = 0;
for k=1:length(y_pred)
    err = y_test(k)-y_pred(k);
    se = se + (err)^2;
end
mse = se/length(y_pred);
%RMSE
rmse = sqrt(mse);

%MAE
ae = 0;
for k=1:length(y_pred)
    err = y_test(k)-y_pred(k);
    ae = ae + abs(err);
end
mae = ae/length(y_pred);

fprintf("MSE of predictions = %0.3f\n",mse);
fprintf("RMSE of predictions = %0.3f\n",rmse);
fprintf("MAE of predictions = %0.3f\n",mae);

