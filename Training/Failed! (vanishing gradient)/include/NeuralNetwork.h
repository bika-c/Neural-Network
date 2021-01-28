#pragma once

#include "mnist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_ROW    20
#define INPUT_COLUMN 20
#define INPUT_NUM    10
#define OUTPUT_NUM   10

#define HIDDEN_LAYER_NUM         2
#define HIDDEN_LAYER_NEURONS_NUM 8
#define LEARNING_RATE            0.2

// #define RANDOM_ALGORITHM (rand() % 1000) * 1.0 / RAND_MAX
#define RANDOM_ALGORITHM 0.1
#define QUEUE_LENGTH     10

typedef struct InputSample
{
    double inputs[INPUT_NUM];
    int    labels[OUTPUT_NUM];
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

// NeuralNetwork AI;

void randomArray(Sample sampleArray[], int length);

void swop(Sample* sample1, Sample* sample2);

int Random(int min, int max);

double logistic(double input);

double dSigmoid(double input);

// Initialize all the weights and Bias with random numbers
void initializeNetwork();

void feedForward(const Sample* sample);

void backpropagation();

double cost();

void trainQueue(Sample sample[], int length);

void train(Sample* sample);

int indexOfDesiredOutput(const Sample* data);

int output(const Sample* const input);

void consoleTest(const Sample* sample);

void DebugDisplayNetwork();