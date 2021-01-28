// #include <math.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>

// #define RANDOM_ALGORITHM (rand() % 1000) * 1.0 / RAND_MAX

// typedef struct Neuron
// {
//     double activation;
//     double weights;
//     double bias;
// } Neuron;

// double Logistic(double input)
// {
//     return 1 / (1 + pow(Euler_Number, -input));
// }

// double input;
// Neuron neuron1, neuron2, output;

// int main()
// {
//     srand((unsigned)time(NULL));
//     neuron1.bias = RANDOM_ALGORITHM;
//     neuron1.weights = RANDOM_ALGORITHM;
//     neuron2.bias = RANDOM_ALGORITHM;
//     neuron2.weights = RANDOM_ALGORITHM;

// }

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_ROW    1
#define INPUT_COLUMN 1
#define INPUT_NUM    INPUT_ROW *INPUT_COLUMN

#define OUTPUT_NUM               1
#define HIDDEN_LAYER_NUM         2
#define HIDDEN_LAYER_NEURONS_NUM 1

#define Euler_Number     2.71828
#define RANDOM_ALGORITHM (rand() % 1000) * 1.0 / RAND_MAX

typedef struct InputSample
{
    double input[INPUT_NUM];
    double label[OUTPUT_NUM];
} Sample;

typedef struct FirstHiddenLayerNeuron
{
    double activation;
    double weights[INPUT_NUM];
    double bias;
} NeuronFHL;

typedef struct Neuron
{
    double activation;
    double weights[HIDDEN_LAYER_NEURONS_NUM];
    double bias;
} Neuron;

typedef struct Input
{
    double activation;
} Input;

typedef struct NeuralNetwork
{
    Input inputLayer[INPUT_NUM];
    NeuronFHL firstHiddenLayer[HIDDEN_LAYER_NEURONS_NUM];
    Neuron hiddenLayers[HIDDEN_LAYER_NUM - 1][HIDDEN_LAYER_NEURONS_NUM];
    Neuron outputLayer[OUTPUT_NUM];
    int anticipation[OUTPUT_NUM];
} NeuralNetwork;

NeuralNetwork AI;

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

double Logistic(double input)
{
    return 1 / (1 + pow(Euler_Number, -input));
}

double Logistic2(double input)
{
    return 1 / (1 + exp(-input));
}

void FeedForward(const Sample *picture)
{
    //Collect data from sample picture
    for (int i = 0; i < INPUT_NUM; i++)
        AI.inputLayer[i].activation = picture->input[i];
    for (int i = 0; i < OUTPUT_NUM; i++)
        AI.anticipation[i] = picture->label[i];

    //Calculate the activation value for all the neurons in the FIRST hidden layer
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++) //
    {
        for (int j = 0; j < INPUT_NUM; j++)
            AI.firstHiddenLayer[i].activation += (AI.inputLayer[j].activation * AI.firstHiddenLayer[i].weights[j]) + AI.firstHiddenLayer[i].bias;
        AI.firstHiddenLayer[i].activation = Logistic(AI.firstHiddenLayer[i].activation);
    }

    //Calculate the activation value for all the neurons in the SECOND hidden layer
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.hiddenLayers[0][i].activation += (AI.firstHiddenLayer[j].activation * AI.hiddenLayers[0][i].weights[j]) + AI.hiddenLayers[0][i].bias;
        AI.hiddenLayers[0][i].activation = Logistic(AI.hiddenLayers[0][i].activation);
    }

    //Calculate the activation value for all the neurons in the REMAINING hidden layers
    for (int i = 1; i < HIDDEN_LAYER_NUM - 1; i++)
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
        {
            for (int k = 0; k < HIDDEN_LAYER_NEURONS_NUM; k++)
                AI.hiddenLayers[i][j].activation += (AI.hiddenLayers[i - 1][k].activation * AI.hiddenLayers[i][j].weights[k]) + AI.hiddenLayers[i][j].bias;
            AI.hiddenLayers[i][j].activation = Logistic(AI.hiddenLayers[i][j].activation);
        }

    //Calculate the activation value for all the neurons in the OUTPUT layer
    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.outputLayer[i].activation += (AI.hiddenLayers[HIDDEN_LAYER_NUM - 2][j].activation * AI.outputLayer[i].weights[j]) + AI.outputLayer[i].bias;
        AI.outputLayer[i].activation = Logistic(AI.outputLayer[i].activation);
    }
}

double Cost()
{
    double cost = 0;
    for (int i = 0; i < OUTPUT_NUM; i++)
        cost += pow(AI.outputLayer[i].activation - AI.anticipation[i], 2);
    return cost;
}

void BackPropagation()
{
}

void Learn(const Sample *picture, const int num)
{
}

int main()
{
    Sample picture = {0, 1};
    srand((unsigned)time(NULL));
    Initialize();
    for (int i = 0; i < 200; i++)
    {
        system("cls");
        FeedForward(&picture);
        printf("Old:\n");
        printf("Cost/wL = %f\n", 2 * (AI.outputLayer[0].activation - AI.anticipation[0]) * Logistic(AI.hiddenLayers[0][0].activation * AI.outputLayer[0].weights[0] + AI.outputLayer[0].bias) * AI.hiddenLayers[0][0].activation);
        printf("Cost/biasL = %f\n", 2 * (AI.outputLayer[0].activation - AI.anticipation[0]) * Logistic(AI.hiddenLayers[0][0].activation * AI.outputLayer[0].weights[0] + AI.outputLayer[0].bias) * 1);
        printf("Cost/activationL-1 = %f\n", 2 * (AI.outputLayer->activation - AI.anticipation[0]) * Logistic(AI.hiddenLayers[0][0].activation * AI.outputLayer[0].weights[0] + AI.outputLayer[0].bias) * AI.outputLayer->weights[0]);
        printf("Cost = %f\n", Cost());
        printf("AI's Output = %f <> Real target = %d\n", AI.outputLayer[0].activation, AI.anticipation[0]);

        AI.outputLayer[0].bias -= (2 * (AI.outputLayer->activation - AI.anticipation[0]) * Logistic(AI.hiddenLayers[0][0].activation * AI.outputLayer[0].weights[0] + AI.outputLayer[0].bias) * AI.hiddenLayers[0][0].activation);
        AI.outputLayer[0].weights[0] -= (2 * (AI.outputLayer->activation - AI.anticipation[0]) * Logistic(AI.hiddenLayers[0][0].activation * AI.outputLayer[0].weights[0] + AI.outputLayer[0].bias) * 1);

        FeedForward(&picture);

        printf("New:\n");
        printf("Cost/wL = %f\n", 2 * (AI.outputLayer->activation - AI.anticipation[0]) * Logistic(AI.hiddenLayers[0][0].activation * AI.outputLayer[0].weights[0] + AI.outputLayer[0].bias) * AI.hiddenLayers[0][0].activation);
        printf("Cost/biasL = %f\n", 2 * (AI.outputLayer->activation - AI.anticipation[0]) * Logistic(AI.hiddenLayers[0][0].activation * AI.outputLayer[0].weights[0] + AI.outputLayer[0].bias) * 1);
        printf("Cost/activationL-1 = %f\n", 2 * (AI.outputLayer->activation - AI.anticipation[0]) * Logistic(AI.hiddenLayers[0][0].activation * AI.outputLayer[0].weights[0] + AI.outputLayer[0].bias) * AI.outputLayer->weights[0]);
        printf("Cost = %f\n", Cost());
        printf("AI's Output = %f <> Real target = %d\n", AI.outputLayer[0].activation, AI.anticipation[0]);
    }

}