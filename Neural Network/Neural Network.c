#include "NeuralNetwork.h"

void swop(Sample* sample1, Sample* sample2)
{
    Sample temp;
    temp     = *sample1;
    *sample1 = *sample2;
    *sample2 = temp;
}

void RandomArray(Sample sampleArray[], int length)
{
    int j = 0;
    for (int i = 0; i < length; i++)
    {
        j = rand() % length;
        swop(&sampleArray[i], &sampleArray[j]);
    }
}

int main()
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

    int training_times = 50000;
    for (int i = 0; i < training_times; i++)
    {
        RandomArray(array, 8);
        Training(array, 8);
        if (i % Random(650, 1200) == 0 || i == training_times - 1)
        {
            system("cls");
            printf("Training : %d/%d\t\033[32m%.2f%%\033[0m\nNetwork Error: %f\33[?25l\n", i + 1, training_times, (i / (double)training_times) * 100.0, Cost());
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
}