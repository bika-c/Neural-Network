#include "..\include\NeuralNetwork.h"
#include <string.h>

NeuralNetwork AI;

static int    index = 0;
static Sample queue[QUEUE_LENGTH];
Sample        data[NUM_TRAIN];

void swop(Sample* sample1, Sample* sample2)
{
    Sample temp;
    temp     = *sample1;
    *sample1 = *sample2;
    *sample2 = temp;
}

void randomArray(Sample sampleArray[], int length)
{
    int j = 0;
    for (int i = 0; i < length; i++)
    {
        j = rand() % length;
        swop(&sampleArray[i], &sampleArray[j]);
    }
}

int Random(int min, int max)
{
    if (max < min) return 0;
    if (max == min) return max;
    return rand() % (max - min + 1) + min;
}

double logistic(double input)
{
    return 1.0 / (1.0 + exp(-input));
}

double dSigmoid(double input)
{
    return input * (1.0 - input);
}

// Initialize all the weights and Bias with random numbers
void initializeNetwork()
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

void feedForward(const Sample* sample)
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
        AI.firstHiddenLayer[i].activation = logistic(AI.firstHiddenLayer[i].activation + AI.firstHiddenLayer[i].bias);
    }

    //Calculate the activation value for all the neurons in the SECOND hidden layer
    for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.hiddenLayers[0][i].activation += AI.firstHiddenLayer[j].activation * AI.hiddenLayers[0][i].weights[j];
        AI.hiddenLayers[0][i].activation = logistic(AI.hiddenLayers[0][i].activation + AI.hiddenLayers[0][i].bias);
    }

    //Calculate the activation value for all the neurons in the REMAINING hidden layers
    for (int i = 1; i < HIDDEN_LAYER_NUM - 1; i++)
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
        {
            for (int k = 0; k < HIDDEN_LAYER_NEURONS_NUM; k++)
                AI.hiddenLayers[i][j].activation += AI.hiddenLayers[i - 1][k].activation * AI.hiddenLayers[i][j].weights[k];
            AI.hiddenLayers[i][j].activation = logistic(AI.hiddenLayers[i][j].activation + AI.hiddenLayers[i][j].bias);
        }

    //Calculate the activation value for all the neurons in the OUTPUT layer
    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
            AI.outputLayer[i].activation += AI.hiddenLayers[HIDDEN_LAYER_NUM - 2][j].activation * AI.outputLayer[i].weights[j];
        AI.outputLayer[i].activation = logistic(AI.outputLayer[i].activation + AI.outputLayer[i].bias);
    }
}

void backpropagation()
{ // Format the data set
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

double cost()
{
    double cost = 0;
    for (int i = 0; i < OUTPUT_NUM; i++)
        cost += pow((AI.outputLayer[i].activation - AI.anticipation[i]), 2);

    return cost;
}

void trainQueue(Sample sampleSet[], int length)
{
    BackpropagationData totalDataset;
    memset(&totalDataset, 0x0, sizeof(BackpropagationData));

    // Collect delta data from each sample and add them up into total which will be used to adjust the network
    for (int i = 0; i < length; i++)
    {
        feedForward(&sampleSet[i]);
        backpropagation();

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

void train(Sample* sample)
{
    BackpropagationData totalDataset;
    memset(&totalDataset, 0x0, sizeof(BackpropagationData));
    memset(AI.anticipation, 0x0, sizeof(double) * OUTPUT_NUM);

    // Collect delta data from each sample and add them up into total which will be used to adjust the network

    feedForward(sample);
    backpropagation();

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

// void trainQueue(int label)
// {
//     if (index == QUEUE_LENGTH)
//     {
//         trainFromQueue(queue, QUEUE_LENGTH);
//         index = 0;
//         memset(queue, 0x0, sizeof(Sample) * QUEUE_LENGTH);
//     }
//     memset(data.labels, 0x0, sizeof(int) * OUTPUT_NUM);
//     data.labels[label] = 1.0;
//     queue[index++]     = data;
// }

int indexOfDesiredOutput(const Sample* data)
{
    for (int i = 0; i < OUTPUT_NUM; i++)
        if (data->labels[i] == 1)
            return i;
    return 0;
}

int output(const Sample* const input)
{
    feedForward(input);
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
    feedForward(sample);

    for (int i = 0; i < INPUT_NUM; i++)
    {
        printf("%1.1f ", sample->inputs[i]);
        if ((i + 1) % 28 == 0) putchar('\n');
    }

    // printf("Inputs: ");
    // for (int i = 0; i < INPUT_NUM; i++)
    //     printf("%.3f\t", sample->inputs[i]);

    printf("\nActual/Label: ");
    for (int i = 0; i < OUTPUT_NUM; i++)
        printf("%.3d\t", sample->labels[i]);

    printf("\nDesired Index: %d\n\033[%dmCost: %f\n\033[0m", indexOfDesiredOutput(sample), cost() > 0.5 ? 31 : 0, cost());
    for (int i = 0; i < OUTPUT_NUM; i++)
    {
        if (AI.outputLayer[i].activation > max)
        {
            index = i;
            max   = AI.outputLayer[i].activation;
        }
        printf("\033[%dm%d.Output: %.10f <> Desired: %.1f\n\033[0m", AI.anticipation[i] - AI.outputLayer[i].activation >= 0 ? 34 : 0, i, AI.outputLayer[i].activation, AI.anticipation[i]);
    }
    printf("\033[%dmNeural Network's output: %d <> Actual: %d\n\033[0m", index != indexOfDesiredOutput(sample) ? 31 : 32, index, indexOfDesiredOutput(sample));
}

void DebugDisplayNetwork()
{
    // for (int i = 0; i < HIDDEN_LAYER_NUM - 1; i++)
    //     for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
    //         AI.hiddenLayers[i][j].bias -= totalDataset.deltaHiddenBias[i + 1][j] * LEARNING_RATE;

    // for (int i = 0; i < OUTPUT_NUM; i++)
    // {
    //     AI.outputLayer[i].bias -= totalDataset.deltaOutputBias[i] * LEARNING_RATE;
    //     for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
    //         AI.outputLayer[i].weights[j] -= totalDataset.deltaOutputWeights[i][j] * LEARNING_RATE;
    // }
    // for (int i = 0; i < HIDDEN_LAYER_NEURONS_NUM; i++)
    // {
    //     AI.firstHiddenLayer[i].bias -= totalDataset.deltaHiddenBias[0][i] * LEARNING_RATE;
    //     for (int j = 0; j < INPUT_NUM; j++)
    //         AI.firstHiddenLayer[i].weights[j] -= totalDataset.deltaFirstHiddenWeights[i][j] * LEARNING_RATE;
    // }
    // for (int i = 0; i < HIDDEN_LAYER_NUM - 1; i++)
    //     for (int j = 0; j < HIDDEN_LAYER_NEURONS_NUM; j++)
    //         for (int k = 0; k < HIDDEN_LAYER_NEURONS_NUM; k++)
    //             AI.hiddenLayers[i][j].weights[k] -= totalDataset.deltaHiddenWeights[i][j][k] * LEARNING_RATE;
}

/*int main()
{
    Sample set1 =
        {
            {0, 0.2, 0.3},            //Data
            {1, 0, 0, 0, 0, 0, 0, 0}, //Label
        };
    Sample set2 =
        {
            {0.3, 0, 1},              //Data
            {0, 1, 0, 0, 0, 0, 0, 0}, //Label
        };
    Sample set3 =
        {
            {0, 0.88, 0.15},          //Data
            {0, 0, 1, 0, 0, 0, 0, 0}, //Label
        };
    Sample set4 =
        {
            {0, 0.65, 0.85},          //Data
            {0, 0, 0, 1, 0, 0, 0, 0}, //Label
        };
    Sample set5 =
        {
            {1, 0, 0.2},              //Data
            {0, 0, 0, 0, 1, 0, 0, 0}, //Label
        };
    Sample set6 =
        {
            {0.95, 0.1, 0.65},        //Data
            {0, 0, 0, 0, 0, 1, 0, 0}, //Label
        };
    Sample set7 =
        {
            {1, 0.7, 0.1},            //Data
            {0, 0, 0, 0, 0, 0, 1, 0}, //Label
        };
    Sample set8 =
        {
            {0.67, 0.9, 1},           //Data
            {0, 0, 0, 0, 0, 0, 0, 1}, //Label
        };
    Sample test1 =
        {
            {0.1, 0, 0},              //Data
            {1, 0, 0, 0, 0, 0, 0, 0}, //Label
        };

    Sample test2 =
        {
            {0, 0.85, 0.95},          //Data
            {0, 0, 0, 1, 0, 0, 0, 0}, //Label
        };

    Sample array[8];

    array[0] = set1;
    array[1] = set2;
    array[2] = set3;
    array[3] = set4;
    array[4] = set5;
    array[5] = set6;
    array[6] = set7;
    array[7] = set8;

    srand((unsigned)time(NULL));
    Initialize();

    for (int i = 0; i < 8; i++)
    {
        printf("\nSet %d test: \n", i + 1);
        consoleTest(&array[i]);
    }

    printf("\n");
    system("pause");

    int train_times = 45000;
    for (int i = 0; i < train_times; i++)
    {
        RandomArray(array, 8);
        train(array, 8);
        if (i % Random(650, 1200) == 0 || i == train_times - 1)
        {
            system("cls");
            printf("train : %d/%d\t\033[32m%.2f%%\033[0m\nNetwork Error: %f\33[?25l\n", i + 1, train_times, (i / (double)train_times) * 100.0, Cost());
        }
    }

    for (int i = 0; i < Random(8, 8); i++)
    {
        printf("\nSet %d test: \n", i + 1);
        consoleTest(&array[i]);
    }

    printf("\nTest sample 1 test: \n");
    consoleTest(&test1);

    printf("\nTest sample 2 test: \n");
    consoleTest(&test2);
}*/