    %% load the dataset
    load fisheriris

    %% Split the data set to features and target
    x = meas(51:end,3:4);
    y = species(51:end);

    %% Fit SVM
    SVmodel1 = fitcsvm(x,y);

    %% Extract support vectors
    sv = SVmodel1.SupportVectors;

    %% Create a grid to plot decition boundry
    x1range = linspace(min(x(:,1)), max(x(:,1)),100);
    x2range = linspace(min(x(:,2)), max(x(:,2)),100);

    %% meshgrid creates 2D cordinates matrix (x1, x2)
    [x1, x2] = meshgrid(x1range, x2range);
    xGrid = [x1(:), x2(:)];

    %% Predict labels for grid
    %[label ,score] = predict(SVmodel1, xGrid));
    [~, score] = predict(SVmodel1, xGrid);

    %% Plot the data points and desition boundry
    figure;
    gscatter(x(:,1), x(:,2), y);
    hold on;
    plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize',10);
    contour(x1, x2, reshape(score(:,2), size(x1)), [0,0], 'k'); %Decision boundry
    xlabel('Patal length');
    ylabel('Patal width');
    legend('Versicolor', 'Verginica','Sipport vector');
    hold off;

    %%
    xNew = [5 2; 4,1.5];
    x = [x;xNew];
    species = predict(SVmodel1, xNew);
    
    %% struct - string concat
    speciesNew = strcat(species, '(Predicted)');
    
    %%
    y = [y;speciesNew];
    figure;
    gscatter(x(:,1), x(:,2), y);
    hold on;
    plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize',10);
    plot(xNew(:,1), xNew(:,2), 'ko');
    contour(x1, x2, reshape(score(:,2), size(x1)), [0,0], 'k'); %Decision boundry
    xlabel('Patal length');
    ylabel('Patal width');
    legend('Versicolor', 'Verginica','Predicted','Predicted','Sipport vector');
    hold off;
    
    
    
    
