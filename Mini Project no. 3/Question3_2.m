filename = 'SteamGeneratorData.xlsx';
data = readmatrix(filename);

data(:, 1) = [];

inputs = data(:, 1:4);
outputs = data(:, 5:8);

inputs = normalize(inputs, 'range');
outputs = normalize(outputs, 'range');

trainRatio = 0.7;
numSamples = size(data, 1);
numTrain = floor(trainRatio * numSamples);

trainInputs = inputs(1:numTrain, :);
trainOutputs = outputs(1:numTrain, :);
testInputs = inputs(numTrain+1:end, :);
testOutputs = outputs(numTrain+1:end, :);

MaxEpoch = 150;
ErrorGoal = 0;
InitialStepSize = 0.01;
StepSizeDecreaseRate = 0.9;
StepSizeIncreaseRate = 1.1;
TrainOptions = [MaxEpoch, ErrorGoal, InitialStepSize, StepSizeDecreaseRate, StepSizeIncreaseRate];
DisplayOptions = [true, true, true, true];
OptimizationMethod = 1; 

TestRMSEs = zeros(1, size(outputs, 2));

for i = 1:size(outputs, 2)
    output_train = trainOutputs(:, i);
    output_test = testOutputs(:, i);
    
    training_data = [trainInputs, output_train];
    
    spread = 0.5;
    fis = genfis2(trainInputs, output_train, spread);
    
    [fis, trainError] = anfis(training_data, fis, TrainOptions, DisplayOptions, [], OptimizationMethod);
    
    TestOutputs = evalfis(testInputs, fis);
    
    TestErrors = output_test - TestOutputs;
    TestMSE = mean(TestErrors(:).^2);
    TestRMSEs(i) = sqrt(TestMSE);
    
    figure;
    plot(1:MaxEpoch, trainError, 'LineWidth', 1.5);
    xlabel('Epoch');
    ylabel('Training RMSE');
    title(['Training RMSE for Output y', num2str(i)]);
    grid on;
    
    figure;
    plot(1:length(output_test), output_test, 'b', 'LineWidth', 1.5);
    hold on;
    plot(1:length(TestOutputs), TestOutputs, 'r--', 'LineWidth', 1.5);
    xlabel('Sample');
    ylabel('Output');
    legend('Actual Output', 'Predicted Output');
    title(['Comparison of Actual and Predicted Outputs for y', num2str(i)]);
    grid on;
    
    [x, y] = meshgrid(linspace(0, 1, 50), linspace(0, 1, 50));
    z = evalfis([x(:), y(:), zeros(size(x(:))), zeros(size(x(:)))], fis);
    z = reshape(z, size(x));
    figure;
    surf(x, y, z);
    xlabel('Input 1');
    ylabel('Input 2');
    zlabel('Output');
    title(['Surface Plot of Fuzzy Model for Output y', num2str(i)]);
    grid on;
end

for i = 1:size(outputs, 2)
    disp(['Output y', num2str(i), ' Test RMSE: ', num2str(TestRMSEs(i))]);
end
