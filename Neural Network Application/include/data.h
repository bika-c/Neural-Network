#pragma once

// #include "NeuralNetwork.h"
#include "graphics.h"
#include "lenet.h"

#define FILE_NAME "./asset/data/model.dat"

void FileNotFound();

bool SearchForDataFile();

void generalizeData();

void readFromFile();

void writeToFile();