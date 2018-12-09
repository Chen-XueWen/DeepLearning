clear;
load iris.mat; 
%MAT made of X = 4 and Y = (setosa = 1,setosa = 0)
%https://archive.ics.uci.edu/ml/machine-learning-databases/iris/

%Randomize Dataset
rows = randperm(size(Iris_Dataset,1));
Iris_Dataset = Iris_Dataset(rows, :);
[sets,~] = size(Iris_Dataset);

%Split X/Y
X_Data = Iris_Dataset(:,1:4);
Y_Data = Iris_Dataset(:,5);

%Normalization
XM = mean(X_Data,2);
XSd = std(X_Data,0,2);
X_Data = (X_Data - XM)./(XSd);

%Split up data into train & test & val(optional)
X_train = X_Data((1:0.8*sets),(1:4));
X_train = X_train.';
Y_train = Y_Data((1:0.8*sets));
Y_train = Y_train.';

X_test = X_Data((0.8*sets+1:sets),(1:4));
X_test = X_test.';
Y_test = Y_Data(0.8*sets+1:sets);
Y_test = Y_test.';


iterations = 20000;
learning_rate = 0.005;

%Calculate Weight and Biased (aka Fit Model)
[w,b] = LogisticRegression(X_train, Y_train, iterations, learning_rate);

%Prediction
Y_prediction_train = HelperFunc.predict(w, b, X_train);
Train_Acc = (1 - mean(abs(Y_prediction_train - Y_train)));
disp("train accuracy: " + Train_Acc*100 + "%")

Y_prediction_test = HelperFunc.predict(w, b, X_test);
Test_Acc = (1 - mean(abs(Y_prediction_test - Y_test)));
disp("test accuracy: " + Test_Acc*100 + "%")

