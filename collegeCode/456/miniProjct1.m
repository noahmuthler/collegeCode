clear; clc; close all;

% input size 6000x2
% target value 6000x1

% load('DataSet1_MP1.mat');
% load('DataSet2_MP1.mat');
% 
% % X = DataSet1;
% % Y = DataSet1_targets;
% 
% X = DataSet2;
% Y = DataSet2_targets;
% 
% n = size(X,1);
% 
% X(:,3) = Y;
% P = randperm(n);
% X = X(P,:);
% 
% CV_size = ceil(n*.2);
% 
% CV_X = X(n-CV_size+1:n,:);
% CV_Y = CV_X(:,3);
% CV_X(:,3) = [];
% 
% X(n-CV_size+1:n,:) = [];
% 
% n = size(X,1);
% 
% Y = X(:,3);
% X(:,3) = [];

% to use DataSet1 comment out test_train_2
% to use DataSet2 comment out test_train_1
%load('test_train_1.mat');
load('test_train_2.mat');

n = size(X,1);

% input layer = 2 neurons
% hidden layer = 20 neurons
% output layer = 1 neuron
% activation function = hyperbolic tangent

alpha = -0.1 + (.2)*rand([20 2]);
b_alpha = -0.1 + (.2)*rand([20 1]);

W = -0.1 + (.2)*rand([20 1]);
b_W = -0.1 + (.2)*rand();

learning_rate_0 = 0.1;
learning_rate = learning_rate_0;
learning_rate_end = 10^-5;
number_of_dropouts = 3;

k = 1;
epoch = 1;
max_epoch = 1000;

cost = zeros([max_epoch 1]);
CV_cost = zeros([max_epoch 1]);

loss = 0;
CV_loss = 0;

while epoch < max_epoch
    i = k - ceil(k/n - 1) * n;

    drop_out = ones([size(W,1) 1]);
    drop_out_vals = randperm(size(W,1),number_of_dropouts);
    drop_out(drop_out_vals) = 0;
    
    % forward prop
    % hidden layer
    hidden_z_in = alpha .* drop_out * X(i,:)' + b_alpha .* drop_out;
    hidden_z_inter = tanh(hidden_z_in);
    
    % output layer
    %out_y_in = W' * hidden_z_out + b_W;
    out_y_in = (W .* drop_out)' * hidden_z_inter + b_W;
    out_y_inter = tanh(out_y_in);
    y_predict = out_y_inter;

    loss = loss + (Y(i) - y_predict)^2;
    %loss = loss + (Y(i) - y_predict)^2 + (lambda/2) * (sum(W .* W) + sum(sum(alpha .* alpha)));

    % back prop
    % output layer
    derivative_out_vals = 1/2*(1 + out_y_inter)*(1 - out_y_inter);
    delta_out = (Y(i) - y_predict)*derivative_out_vals;
    %delta_W = learning_rate*delta_out*hidden_z_out;
    delta_W = learning_rate*delta_out*hidden_z_inter;
    %delta_W = learning_rate*(delta_out + lambda*(sum(sum(W .* W)) + sum(sum(alpha .* alpha))))*hidden_z_inter;
    delta_b_W = learning_rate*delta_out;

    % hidden layer
    %delta_hidden_in = delta_out * (alpha(:,1)  + alpha(:,2)) * size(alpha,2);
    delta_hidden_in = delta_out * (W .* drop_out);
    delta_hidden = delta_hidden_in * 1/2 .* (1 + hidden_z_inter) .* (1 - hidden_z_inter);
    delta_alpha = learning_rate * delta_hidden * X(i,:);
    delta_b_alpha = learning_rate * delta_hidden;

    W = W + delta_W;
    b_W = b_W + delta_b_W;

    alpha = alpha + delta_alpha;
    b_alpha = b_alpha + delta_b_alpha;
    
    if(i == n)
        cost(epoch) = loss / n;
        loss = 0;
        
        for j = 1:size(CV_X,1)
            hidden_z_in_CV = alpha * CV_X(j,:)' + b_alpha;
            hidden_z_inter_CV = tanh(hidden_z_in_CV);
            
            % output layer
            %out_y_in_CV = W' * hidden_z_out_CV + b_W;
            out_y_in_CV = W' * hidden_z_inter_CV + b_W;
            out_y_inter_CV = tanh(out_y_in_CV);
            y_predict_CV = out_y_inter_CV;

            CV_loss = CV_loss + (CV_Y(j) - y_predict_CV)^2;
        end
        
        CV_cost(epoch) = CV_loss / size(CV_X,1);
        CV_loss = 0;

        X(:,3) = Y;
        P = randperm(n);
        X = X(P,:);
        Y = X(:,3);
        X(:,3) = [];

        % update learning rate
        learning_rate = learning_rate_0 - (learning_rate_0 - learning_rate_end) / max_epoch * epoch;
        epoch = epoch + 1;
    end

    k = k + 1;
end

%% testing training data
X_correct = 0;

for i = 1:4800
    % forward prop
    % hidden layer
    hidden_z_in = alpha * X(i,:)' + b_alpha;
    hidden_z_inter = tanh(hidden_z_in);
    
    % output layer
    out_y_in = W' * hidden_z_inter + b_W;
    out_y_inter = tanh(out_y_in);
    y_predict = out_y_inter;

    if(y_predict == Y(i))
        X_correct = X_correct + 1;
    end
end
%% testing training data
X_correct = 0;

for i = 1:4800
    % forward prop
    % hidden layer
    hidden_z_in = alpha * X(i,:)' + b_alpha;
    hidden_z_inter = tanh(hidden_z_in);
    
    % output layer
    out_y_in = W' * hidden_z_inter + b_W;
    out_y_inter = tanh(out_y_in);
    y_predict = out_y_inter;
    %y_predict = bipolar(out_y_inter);

    if(y_predict > 0 && Y(i) == 1)
        X_correct = X_correct + 1;
    elseif(y_predict < 0 && Y(i) == -1)
        X_correct = X_correct + 1;
    end
end
%% testing CV data
CV_correct = 0;

for i = 1:1200
    % forward prop
    % hidden layer
    hidden_z_in = alpha * CV_X(i,:)' + b_alpha;
    hidden_z_inter = tanh(hidden_z_in);
    
    % output layer
    out_y_in = W' * hidden_z_inter + b_W;
    out_y_inter = tanh(out_y_in);
    y_predict = out_y_inter;
    %y_predict = bipolar(out_y_inter);

    if(y_predict > 0 && CV_Y(i) == 1)
        CV_correct = CV_correct + 1;
    elseif(y_predict < 0 && CV_Y(i) == -1)
        CV_correct = CV_correct + 1;
    end
end


%% plotting decision boundaries
[meshX, meshY] = meshgrid(-15:0.1:25,-10:0.1:15);
test_data = [meshX(:) meshY(:)];

output = zeros([size(test_data,1) 1]);

% forward prop
% hidden layer

for i = 1:size(test_data,1)
    hidden_z_in = alpha * test_data(i,:)' + b_alpha;
    hidden_z_inter = tanh(hidden_z_in);
    %hidden_z_out = bipolar(hidden_z_inter);
    
    % output layer
    %out_y_in = W' * hidden_z_out + b_W;
    out_y_in = W' * hidden_z_inter + b_W;
    out_y_inter = tanh(out_y_in);
    output(i) = out_y_inter;
    %output(i) = bipolar(out_y_inter);
end

figure;
scatter(meshX(:), meshY(:), 10, output)
colormap hot;
hold on
temp = (Y + 1) / 2;
temp_X = X .* temp;
temp = nonzeros(temp);
temp_X = nonzeros(temp_X);
temp_X = reshape(temp_X, size(temp_X,1)/2,2);
plot(temp_X(:,1), temp_X(:,2), 'r+')
temp = (Y - 1) / 2;
temp_X = X .* temp;
temp = nonzeros(temp);
temp_X = nonzeros(temp_X);
temp_X = reshape(temp_X, size(temp_X,1)/2,2);
plot(-1*temp_X(:,1), -1*temp_X(:,2), 'g+')
t = strcat("Training data, Correct: ", string(X_correct/size(X,1) * 100), "%");
title(t);


figure;
scatter(meshX(:), meshY(:), 10, output)
colormap hot;
hold on
temp = (CV_Y + 1) / 2;
temp_X = CV_X .* temp;
temp = nonzeros(temp);
temp_X = nonzeros(temp_X);
temp_X = reshape(temp_X, size(temp_X,1)/2,2);
plot(temp_X(:,1), temp_X(:,2), 'r+')
temp = (CV_Y - 1) / 2;
temp_X = CV_X .* temp;
temp = nonzeros(temp);
temp_X = nonzeros(temp_X);
temp_X = reshape(temp_X, size(temp_X,1)/2,2);
plot(-1*temp_X(:,1), -1*temp_X(:,2), 'g+')
t = strcat("Testing data, Correct: ", string(CV_correct/size(CV_X,1)*100), "%");
title(t);

%% plotting training and CV MSE
figure;
plot(1:max_epoch,cost, 'x')
hold on
plot(1:max_epoch,CV_cost, 'x')
legend(["Cost", "CV cost"])
xlabel('Number of epochs')
ylabel('Cost')


