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

cleanedInputData = table2array(cleanedInputData);
cleanedOutputData = table2array(cleanedOutputData);

cleanedInputData = (cleanedInputData - min(cleanedInputData)) ./ (max(cleanedInputData) - min(cleanedInputData));
cleanedOutputData = (cleanedOutputData - min(cleanedOutputData)) ./ (max(cleanedOutputData) - min(cleanedOutputData));

totalData = height(cleanedData);

trainSize = round(0.6 * totalData);
testSize = round(0.2 * totalData);
validationSize = totalData - trainSize - testSize;

trainInput = cleanedInputData(1:trainSize, :);
trainOutput = cleanedOutputData(1:trainSize, :);
testInput = cleanedInputData(trainSize+1:trainSize+testSize, :);
testOutput = cleanedOutputData(trainSize+1:trainSize+testSize, :);
validationInput = cleanedInputData(trainSize+testSize+1:end, :);
validationOutput = cleanedOutputData(trainSize+testSize+1:end, :);

spread = 0.5;
goal = 0.001;  
maxNeurons = 50;
displayFreq = 1;

net = newrb(trainInput', trainOutput', goal, spread, maxNeurons, displayFreq);

testPredicted = sim(net, testInput');
validationPredicted = sim(net, validationInput');

testRMSE = sqrt(mean((testOutput - testPredicted').^2));
validationRMSE = sqrt(mean((validationOutput - validationPredicted').^2));

disp('Test RMSE:');
disp(testRMSE);
disp('Validation RMSE:');
disp(validationRMSE);

figure;
plot(1:length(testOutput), testOutput, 'b', 'LineWidth', 1.5);
hold on;
plot(1:length(testPredicted), testPredicted, 'r--', 'LineWidth', 1.5);
xlabel('Sample');
ylabel('Output');
legend('Actual Output', 'Predicted Output');
title('Actual vs Predicted Outputs (Test Data)');
grid on;

figure;
plot(1:length(validationOutput), validationOutput, 'b', 'LineWidth', 1.5);
hold on;
plot(1:length(validationPredicted), validationPredicted, 'r--', 'LineWidth', 1.5);
xlabel('Sample');
ylabel('Output');
legend('Actual Output', 'Predicted Output');
title('Actual vs Predicted Outputs (Validation Data)');
grid on;

neurons = 1:maxNeurons;
rmseValues = zeros(size(neurons));

for i = 1:maxNeurons
    tempNet = newrb(trainInput', trainOutput', goal, spread, i, displayFreq);
    tempPredicted = sim(tempNet, testInput');
    rmseValues(i) = sqrt(mean((testOutput - tempPredicted').^2));
end

figure;
plot(neurons, rmseValues, 'LineWidth', 1.5);
xlabel('Number of Neurons');
ylabel('RMSE');
title('RMSE vs Number of Neurons');
grid on;
