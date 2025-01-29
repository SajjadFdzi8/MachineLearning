data = readtable('AirQualityUCI.xlsx');

data(:, [1, 2]) = []; 

inputData = data(:, 1:end-1);
outputData = data(:, end);

invalidData = any(inputData{:,:} == -200, 2);
cleanedData = data(~invalidData, :);

cleanedInputData = cleanedData(:, 1:end-1);
cleanedOutputData = cleanedData(:, end);

disp('Number of remaining data after removing -200:');
disp(sum(~invalidData));

disp('Number of valid data in input columns:');
disp(sum(~invalidData));

disp('Number of valid data in output column:');
disp(sum(~invalidData));

cleanedInputData = table2array(cleanedInputData);
cleanedOutputData = table2array(cleanedOutputData);

cleanedInputData = (cleanedInputData - min(cleanedInputData)) ./ (max(cleanedInputData) - min(cleanedInputData));
cleanedOutputData = (cleanedOutputData - min(cleanedOutputData)) ./ (max(cleanedOutputData) - min(cleanedOutputData));

totalData = height(cleanedData);

trainSize = round(0.6 * totalData);
testSize = round(0.2 * totalData);
validationSize = totalData - trainSize - testSize;

trainData = cleanedData(1:trainSize, :);
testData = cleanedData(trainSize+1:trainSize+testSize, :);
validationData = cleanedData(trainSize+testSize+1:end, :);

disp('Total number of training data:');
disp(trainSize);

disp('Total number of testing data:');
disp(testSize);

disp('Total number of validation data:');
disp(validationSize);

training_data = [cleanedInputData, cleanedOutputData];

spread = 0.5;
fis = genfis2(cleanedInputData, cleanedOutputData, spread);

MaxEpoch = 150;
ErrorGoal = 0;
InitialStepSize = 0.01;
StepSizeDecreaseRate = 0.9;
StepSizeIncreaseRate = 1.1;
TrainOptions = [MaxEpoch, ErrorGoal, InitialStepSize, StepSizeDecreaseRate, StepSizeIncreaseRate];

DisplayInfo = true;
DisplayError = true;
DisplayStepSize = true;
DisplayFinalResult = true;
DisplayOptions = [DisplayInfo, DisplayError, DisplayStepSize, DisplayFinalResult];

OptimizationMethod = 1;

[fis, trainError] = anfis(training_data, fis, TrainOptions, DisplayOptions, [], OptimizationMethod);

disp('Trained FIS:');
disp(fis);

testInputData = cleanedInputData(trainSize + 1:trainSize + testSize, :);
validationInputData = cleanedInputData(trainSize + testSize + 1:end, :);

testActualOutput = cleanedOutputData(trainSize + 1:trainSize + testSize);
validationActualOutput = cleanedOutputData(trainSize + testSize + 1:end);

testPredictedOutput = evalfis(testInputData, fis);
validationPredictedOutput = evalfis(validationInputData, fis);

testRMSE = sqrt(mean((testActualOutput - testPredictedOutput).^2));

validationRMSE = sqrt(mean((validationActualOutput - validationPredictedOutput).^2));

disp('Test RMSE:');
disp(testRMSE);

disp('Validation RMSE:');
disp(validationRMSE);

figure;
plot(1:length(testActualOutput), testActualOutput, 'b', 'LineWidth', 1.5);
hold on;
plot(1:length(testPredictedOutput), testPredictedOutput, 'r--', 'LineWidth', 1.5);
xlabel('Sample');
ylabel('Output');
legend('Actual Output', 'Predicted Output');
title('Actual vs Predicted (Test Data)');
grid on;

figure;
plot(1:length(validationActualOutput), validationActualOutput, 'b', 'LineWidth', 1.5);
hold on;
plot(1:length(validationPredictedOutput), validationPredictedOutput, 'r--', 'LineWidth', 1.5);
xlabel('Sample');
ylabel('Output');
legend('Actual Output', 'Predicted Output');
title('Actual vs Predicted (Validation Data)');
grid on;

figure;
plot(1:MaxEpoch, trainError, 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Training RMSE');
title('Training RMSE for FIS');
grid on;
