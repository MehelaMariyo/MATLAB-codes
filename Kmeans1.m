%% Load the data set
data = readtable('Dataset_01.xlsx');
head(data)

%% Define feature matrix
X = [data.ApplicantIncome data.CreditHistory];

%%
figure;
plot(X(:,1), X(:,2),'.')

%%
[idx2,C2] = kmeans(X,2,'Distance','cityblock','Display','final','Replicates',5);
[idx3,C3] = kmeans(X,3,'Distance','cityblock','Display','final','Replicates',5);
[idx4,C4] = kmeans(X,4,'Distance','cityblock','Display','final','Replicates',5);
[idx5,C5] = kmeans(X,5,'Distance','cityblock','Display','final','Replicates',5);
[idx6,C6] = kmeans(X,6,'Distance','cityblock','Display','final','Replicates',5);
[idx7,C7] = kmeans(X,7,'Distance','cityblock','Display','final','Replicates',5);

%%
figure
[silh7,h] = silhouette(X,idx7,'cityblock');
[silh2,h] = silhouette(X,idx2,'cityblock');
[silh3,h] = silhouette(X,idx3,'cityblock');
[silh4,h] = silhouette(X,idx4,'cityblock');
[silh5,h] = silhouette(X,idx5,'cityblock');
[silh6,h] = silhouette(X,idx6,'cityblock');
h = gca;
h.Children.EdgeColor = [.8 .8 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'
%%
figure;
gscatter(X(:,1), X(:,2), idx2);
figure;
gscatter(X(:,1), X(:,2), idx3);

figure;
gscatter(X(:,1), X(:,2), idx4);
figure;
gscatter(X(:,1), X(:,2), idx5);

figure;
gscatter(X(:,1), X(:,2), idx6);
figure;
gscatter(X(:,1), X(:,2), idx7);

%%
mean(silh7)
mean(silh2)
mean(silh3)
mean(silh4)
mean(silh5)
mean(silh6)

