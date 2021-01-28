// Simple Neural Network
// Created by William Chen
// Failed to learn on a large scale.
// Suspicious of vanishing gradient

#pragma once

#include "pixel.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_ROW    20
#define INPUT_COLUMN 20
#define INPUT_NUM    NUM_PIXEL
#define OUTPUT_NUM   10

#define HIDDEN_LAYER_NUM         2
#define HIDDEN_LAYER_NEURONS_NUM 10
#define LEARNING_RATE            0.35

#define RANDOM_ALGORITHM (rand() % 1000) * 1.0 / RAND_MAX
#define QUEUE_LENGTH     10

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

// NeuralNetwork AI;

// Initialize all the weights and Bias with random numbers
void initializeNetwork();

// Train a batch of samples
void trainInBatches(int labels[]);

// Train a single sample
void train(int label);

// Output/predict
int output(const Sample* const input);

// Print result to console
void consoleTest(const Sample* sample);

// Debug purpose (Unfinished)
void DebugDisplayNetwork();