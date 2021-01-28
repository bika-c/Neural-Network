#include "../include/NeuralNetwork.h"
#include <stdlib.h>

NeuralNetwork AI;

extern double train_image[NUM_TRAIN][SIZE];
extern double test_image[NUM_TEST][SIZE];
extern int    train_label[NUM_TRAIN];
extern int    test_label[NUM_TEST];
extern Sample data[NUM_TRAIN];

Sample test[NUM_TEST];

void readFromFile()
{
    // if (! SearchForDataFile())
    //     return;
    FILE* file = fopen("data.dat", "rb");
    fread(&AI, sizeof(NeuralNetwork), 1, file);
    fclose(file);
}

void writeToFile()
{
    // if (! SearchForDataFile())
    //     return;
    FILE* file = fopen("data.dat", "wb");
    fwrite(&AI, sizeof(NeuralNetwork), 1, file);
    fclose(file);
}

void SHOW(Sample* testData)
{
    consoleTest(testData);
    system("pause");
    system("cls");
}
int main()
{
    load_mnist();
    initializeNetwork();

    for (int i = 0; i < NUM_TRAIN; i++)
    {
        for (int j = 0; j < INPUT_NUM; j++)
        {
            data[i].inputs[j] = train_image[i][j];
        }
        data[i].labels[train_label[i]] = 1;
    }

    for (int i = 0; i < NUM_TEST; i++)
    {
        for (int j = 0; j < INPUT_NUM; j++)
        {
            test[i].inputs[j] = test_image[i][j];
        }
        test[i].labels[test_label[i]] = 1;
    }

    consoleTest(&test[0]);

    system("pause");
    system("cls");
    SHOW(&data[0]);

    int training_times = NUM_TRAIN/10;
    for (int i = 0; i < training_times; i += 10)
    {
        randomArray(&data[i], 10);
        trainQueue(&data[i], 10);
        // if (i % Random(650, 1200) == 0 || i == training_times - 1)
        // {
        system("cls");
        printf("Training : %d/%d\t\033[32m%.2f%%\033[0m\nNetwork Error: %f\33[?25l\n", i + 1, training_times, (i / (double)training_times) * 100.0, cost());
        // }
    }

    SHOW(&data[0]);
    SHOW(&test[0]);
}