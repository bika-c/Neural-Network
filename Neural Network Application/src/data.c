#include "../include/data.h"
#include <stdio.h>
#include <string.h>

extern Pixel pixelSquares[NUM_PIXEL];
LeNet5       AI;
Image        image;

bool SearchForDataFile()
{
    FILE* temp  = fopen(FILE_NAME, "rb");
    bool  check = ! (temp == NULL);
    fclose(temp);
    return check;
}

void FileNotFound()
{
    errorMessage("The file 'model.dat' is missing! Please download from Github and place it under asset/data");
    quit();
    exit(0);
}

void readFromFile()
{
    if (! SearchForDataFile())
        return;
    FILE* file = fopen(FILE_NAME, "rb");
    fread(&AI, sizeof(LeNet5), 1, file);
    fclose(file);
}

void writeToFile()
{
    if (! SearchForDataFile())
        return;
    FILE* file = fopen(FILE_NAME, "wb");
    fwrite(&AI, sizeof(LeNet5), 1, file);
    fclose(file);
}


//Read data from from scaled pixel
void generalizeData()
{
    memset(image, 0x0, sizeof(unsigned char) * NUM_PIXEL);
    // memset(&data, 0x0, sizeof(Sample) );
    for (int i = 0; i < PIXELS_HEIGHT; i++)
    {
        for (int j = 0; j < PIXELS_WIDTH; j++)
        {
            image[i][j] = (unsigned char)((pixelSquares[i * PIXELS_WIDTH + j].activation) * 255) > 255 ? 255 : (unsigned char)((pixelSquares[i * PIXELS_WIDTH + j].activation) * 255);
        }
        // if (data.inputs[i] != 0)
        //     printf("[%d]: %f\n", i, data.inputs[i]);
    }
}