clear;
load ChronicKidneyDisease.mat; 
%MAT made of X = 24 and Y = 25(ckd = 1,non-ckd = 0)

%Randomize Dataset
rows = randperm(size(ChronicKidneyDisease,1));
ChronicKidneyDisease = ChronicKidneyDisease(rows, :);
[sets,~] = size(ChronicKidneyDisease);

%Split X/Y
X_Data = ChronicKidneyDisease(:,1:24);
Y_Data = ChronicKidneyDisease(:,25);
X_Data = X_Data.';
Y_Data = Y_Data.';
%Normalization
XM = mean(X_Data,2);
XSd = std(X_Data,0,2);
X_Data = (X_Data - XM)./(XSd);

%Split up data into train & test & val(optional)
X_train = X_Data((1:24),(1:0.8*sets));
Y_train = Y_Data((1:0.8*sets));

X_test = X_Data((1:24),(0.8*sets+1:sets));
Y_test = Y_Data(0.8*sets+1:sets);


iterations = 20000;
learning_rate = 0.01;

%Calculate Weight and Biased (aka Fit Model)
[w,b] = LogisticRegression(X_train, Y_train, iterations, learning_rate);

%Prediction
Y_prediction_train = HelperFunc.predict(w, b, X_train);
Train_Acc = (1 - mean(abs(Y_prediction_train - Y_train)));
disp("train accuracy: " + Train_Acc*100 + "%")

Y_prediction_test = HelperFunc.predict(w, b, X_test);
Test_Acc = (1 - mean(abs(Y_prediction_test - Y_test)));
disp("test accuracy: " + Test_Acc*100 + "%")