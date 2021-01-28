#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_ROW    3
#define INPUT_COLUMN 1
#define INPUT_NUM    INPUT_ROW* INPUT_COLUMN
#define OUTPUT_NUM   8

#define HIDDEN_LAYER_NUM         2
#define HIDDEN_LAYER_NEURONS_NUM 8
#define LEARNING_RATE            0.2

#define RANDOM_ALGORITHM (rand() % 1000) * 1.0 / RAND_MAX

typedef struct InputSample
{
    double inputs[INPUT_NUM];
    double labels[OUTPUT_NUM];
} Sample; // Inputs as all the input data and labels as the correct ans/label of the sample/data set

typedef struct Input
{
    double activation;
} Input; // Input, each input has an activation

typedef struct FirstHiddenLayerNeuron
{
    double activation;
    double weights[INPUT_NUM];
    double bias;
} NeuronFHL; // First hidden layer neuron - each neuron's weights array has the size depends on the input layer

typedef struct Neuron
{
    double activation;
    double weights[HIDDEN_LAYER_NEURONS_NUM];
    double bias;
} Neuron; // Neuron, each neuron has the size of the weights array depends on the total number of neurons in each layer

typedef struct BackpropagationData
{
    double deltaFirstHiddenWeights[HIDDEN_LAYER_NEURONS_NUM][INPUT_NUM];
    double deltaHiddenBias[HIDDEN_LAYER_NUM][HIDDEN_LAYER_NEURONS_NUM];
    double deltaHiddenWeights[HIDDEN_LAYER_NUM - 1][HIDDEN_LAYER_NEURONS_NUM][HIDDEN_LAYER_NEURONS_NUM];
    double deltaOutputWeights[OUTPUT_NUM][HIDDEN_LAYER_NEURONS_NUM];
    double deltaOutputBias[OUTPUT_NUM];
} BackpropagationData;

typedef struct NeuralNetwork
{
    Input               inputLayer[INPUT_NUM];
    NeuronFHL           firstHiddenLayer[HIDDEN_LAYER_NEURONS_NUM];
    Neuron              hiddenLayers[HIDDEN_LAYER_NUM - 1][HIDDEN_LAYER_NEURONS_NUM];
    Neuron              outputLayer[OUTPUT_NUM];
    double              anticipation[OUTPUT_NUM];
    BackpropagationData deltaDataset;
} NeuralNetwork; // The neural network, contains an input layer, hidden layers (first + the rest), an output, and anticipations for a specific sample and a set of delta data

NeuralNetwork AI;

int Random(int min, int max)
{
    if (max < min) return 0;
    if (max == min) return max;
    return rand() % (max - min + 1) + min;
}

double Logistic(double input)
{
    return 1 / (1 + exp(-input));
}

double dSigmoid(double input)
{
    return input * (1 - input);
}

// Initialize all the weights and Bias with random numbers
void Initialize()
{
    //First hidden layer
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
    {
        AI.firstHiddenLayer[i].bias = RANDOM_ALGORITHM;
        for (int j = 0; j < INPUT_NUM; j++)
            AI.firstHiddenLayer[i].weights[j] = RANDOM_ALGORITHM;
    }

    //Rest of the hidden layers
    for (int i = 0; i < HIDDEN_LAYER_NUM - 1; i++)
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
        {
            AI.hiddenLayers[i][j].bias = RANDOM_ALGORITHM;
            for (int k = 0; k < HIDDEN_LAYER_NEURONS_NUM; k++)
                AI.hiddenLayers[i][j].weights[k] = RANDOM_ALGORITHM;
        }

    //Output Layer
    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        AI.outputLayer[i].bias = RANDOM_ALGORITHM;
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.outputLayer[i].weights[j] = RANDOM_ALGORITHM;
    }
}

void FeedForward(const Sample* sample)
{
    //Collect data from sample
    for (int i = 0; i < INPUT_NUM; i++)
        AI.inputLayer[i].activation = sample->inputs[i];
    for (int i = 0; i < OUTPUT_NUM; i++)
        AI.anticipation[i] = sample->labels[i];

    //Calculate the activation value for all the neurons in the FIRST hidden layer
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
    {
        for (int j = 0; j < INPUT_NUM; j++)
            AI.firstHiddenLayer[i].activation += AI.inputLayer[j].activation * AI.firstHiddenLayer[i].weights[j];
        AI.firstHiddenLayer[i].activation = Logistic(AI.firstHiddenLayer[i].activation + AI.firstHiddenLayer[i].bias);
    }

    //Calculate the activation value for all the neurons in the SECOND hidden layer
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.hiddenLayers[0][i].activation += AI.firstHiddenLayer[j].activation * AI.hiddenLayers[0][i].weights[j];
        AI.hiddenLayers[0][i].activation = Logistic(AI.hiddenLayers[0][i].activation + AI.hiddenLayers[0][i].bias);
    }

    //Calculate the activation value for all the neurons in the REMAINING hidden layers
    for (int i = 1; i < HIDDEN_LAYER_NUM - 1; i++)
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
        {
            for (int k = 0; k < HIDDEN_LAYER_NEURONS_NUM; k++)
                AI.hiddenLayers[i][j].activation += AI.hiddenLayers[i - 1][k].activation * AI.hiddenLayers[i][j].weights[k];
            AI.hiddenLayers[i][j].activation = Logistic(AI.hiddenLayers[i][j].activation + AI.hiddenLayers[i][j].bias);
        }

    //Calculate the activation value for all the neurons in the OUTPUT layer
    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.outputLayer[i].activation += AI.hiddenLayers[HIDDEN_LAYER_NUM - 2][j].activation * AI.outputLayer[i].weights[j];
        AI.outputLayer[i].activation = Logistic(AI.outputLayer[i].activation + AI.outputLayer[i].bias);
    }
}

void Backpropagation()
{
    // Format the data set
    memset(&AI.deltaDataset, 0x0, sizeof(BackpropagationData));

    // Output layer
    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        AI.deltaDataset.deltaOutputBias[i] = (AI.outputLayer[i].activation - AI.anticipation[i]) * dSigmoid(AI.outputLayer[i].activation);
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
        {
            AI.deltaDataset.deltaOutputWeights[i][j] = AI.deltaDataset.deltaOutputBias[i] * AI.hiddenLayers[HIDDEN_LAYER_NUM - 2][j].activation;
            AI.deltaDataset.deltaHiddenBias[HIDDEN_LAYER_NUM - 1][j] += AI.deltaDataset.deltaOutputBias[i] * AI.outputLayer[i].weights[j]
                                                                        * dSigmoid(AI.hiddenLayers[HIDDEN_LAYER_NUM - 2][j].activation);
        }
    }

    // Remaining hidden layers except the First hidden layer
    for (int i = HIDDEN_LAYER_NUM - 2; i > 0; i--)
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            for (int k = 0; k < HIDDEN_LAYER_NEURONS_NUM; k++)
            {
                AI.deltaDataset.deltaHiddenWeights[i][j][k] = AI.deltaDataset.deltaHiddenBias[i + 1][j] * AI.hiddenLayers[i - 1][k].activation;
                AI.deltaDataset.deltaHiddenBias[i][k] += AI.deltaDataset.deltaHiddenBias[i + 1][j] * AI.hiddenLayers[i][j].weights[k]
                                                         * dSigmoid(AI.hiddenLayers[i - 1][k].activation);
            }
    // First hidden layer
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
        {
            AI.deltaDataset.deltaHiddenWeights[0][i][j] = AI.deltaDataset.deltaHiddenBias[1][i] * AI.firstHiddenLayer[j].activation;
            AI.deltaDataset.deltaHiddenBias[0][j] += AI.deltaDataset.deltaHiddenBias[1][i] * AI.hiddenLayers[0][i].weights[j]
                                                     * dSigmoid(AI.firstHiddenLayer[j].activation);
        }
    // First hidden layer weights
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
        for (int j = 0; j < INPUT_NUM; j++)
            AI.deltaDataset.deltaFirstHiddenWeights[i][j] = AI.deltaDataset.deltaHiddenBias[0][i] * AI.inputLayer[j].activation;
}

double Cost()
{
    double cost = 0;
    for (int i = 0; i < OUTPUT_NUM; i++)
        cost += pow((AI.outputLayer[i].activation - AI.anticipation[i]), 2);

    return cost;
}

void Training(const Sample* sampleSet, int length)
{
    BackpropagationData totalDataset;
    memset(&totalDataset, 0x0, sizeof(BackpropagationData));

    // Collect delta data from each sample and add them up into total which will be used to adjust the network
    for (int i = 0; i < length; i++)
    {
        FeedForward(&sampleSet[i]);
        Backpropagation();

        for (int i = 0; i < HIDDEN_LAYER_NUM - 1; i++)
            for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
                totalDataset.deltaHiddenBias[i][j] += AI.deltaDataset.deltaHiddenBias[i][j];

        for (int i = 0; i < OUTPUT_NUM; i++)
        {
            totalDataset.deltaOutputBias[i] += AI.deltaDataset.deltaOutputBias[i];
            for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
                totalDataset.deltaOutputWeights[i][j] += AI.deltaDataset.deltaOutputWeights[i][j];
        }
        for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
            for (int j = 0; j < INPUT_NUM; j++)
                totalDataset.deltaFirstHiddenWeights[i][j] += AI.deltaDataset.deltaFirstHiddenWeights[i][j];
        for (int i = 0; i < HIDDEN_LAYER_NUM - 1; i++)
            for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
                for (int k = 0; k < HIDDEN_LAYER_NEURONS_NUM; k++)
                    totalDataset.deltaHiddenWeights[i][j][k] += AI.deltaDataset.deltaHiddenWeights[i][j][k];
    }

    // <Learning process> Adjust all weights and biases depends on the delta dataset
    for (int i = 0; i < HIDDEN_LAYER_NUM - 1; i++)
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.hiddenLayers[i][j].bias -= totalDataset.deltaHiddenBias[i + 1][j] * LEARNING_RATE;

    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        AI.outputLayer[i].bias -= totalDataset.deltaOutputBias[i] * LEARNING_RATE;
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.outputLayer[i].weights[j] -= totalDataset.deltaOutputWeights[i][j] * LEARNING_RATE;
    }
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
    {
        AI.firstHiddenLayer[i].bias -= totalDataset.deltaHiddenBias[0][i] * LEARNING_RATE;
        for (int j = 0; j < INPUT_NUM; j++)
            AI.firstHiddenLayer[i].weights[j] -= totalDataset.deltaFirstHiddenWeights[i][j] * LEARNING_RATE;
    }
    for (int i = 0; i < HIDDEN_LAYER_NUM - 1; i++)
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            for (int k = 0; k < HIDDEN_LAYER_NEURONS_NUM; k++)
                AI.hiddenLayers[i][j].weights[k] -= totalDataset.deltaHiddenWeights[i][j][k] * LEARNING_RATE;
}

int IndexOfDesiredOutput(const Sample* data)
{
    for (int i = 0; i < OUTPUT_NUM; i++)
        if (data->labels[i] == 1)
            return i;
    return 0;
}

int output(const Sample* const input)
{
    FeedForward(input);
    int    index = 0;
    double max   = AI.outputLayer[0].activation;
    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        if (AI.outputLayer[i].activation > max)
        {
            index = i;
            max   = AI.outputLayer[i].activation;
        }
    }
    return index;
}

void consoleTest(const Sample* sample)
{
    int    index = 0;
    double max   = 0;
    FeedForward(sample);

    printf("Inputs: ");
    for (int i = 0; i < INPUT_NUM; i++)
        printf("%.3f\t", sample->inputs[i]);
    printf("\nActual/Label: ");
    for (int i = 0; i < OUTPUT_NUM; i++)
        printf("%.3f\t", sample->labels[i]);

    printf("\nDesired Index: %d\n\033[%dmCost: %f\n\033[0m", IndexOfDesiredOutput(sample), Cost() > 0.5 ? 31 : 0, Cost());
    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        if (AI.outputLayer[i].activation > max)
        {
            index = i;
            max   = AI.outputLayer[i].activation;
        }
        printf("\033[%dm%d.Output: %.10f <> Desired: %.1f\n\033[0m", AI.anticipation[i] - AI.outputLayer[i].activation >= 0 ? 34 : 0, i, AI.outputLayer[i].activation, AI.anticipation[i]);
    }
    printf("\033[%dmNeural Network's output: %d <> Actual: %d\n\033[0m", index != IndexOfDesiredOutput(sample) ? 31 : 32, index, IndexOfDesiredOutput(sample));
}