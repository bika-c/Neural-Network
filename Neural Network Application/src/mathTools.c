#include "../include/specialCircleMathTools.h"
#include <stdio.h>

float centralAngle(float chordLength, float radius)
{
    return 2 * asin((chordLength / 2) / radius);
}

float archLength(float chordLength, float radius)
{
    return radius * centralAngle(chordLength, radius);
}

float sectorArea(float chordLength, float radius)
{
    return archLength(chordLength, radius) * radius / 2;
}

float segmentArea(float chordLength, float radius)
{
    // printf("chordL: %f, radius: %f, area: %f\n", chordLength, radius, pow(radius, 2) * (centralAngle(chordLength, radius) - sin(centralAngle(chordLength, radius))) / 2);
    return pow(radius, 2) * (centralAngle(chordLength, radius) - sin(centralAngle(chordLength, radius))) / 2;
}

float chordLength(float radius, float distance)
{
    // printf("radius： %f, distance: %f, chord: %f\n", radius, distance, 2 * sqrt(fabs(pow(radius, 2) - pow(distance, 2))));
    return 2 * sqrt(fabs(pow(radius, 2) - pow(distance, 2)));
}

float cornerArea(float chordLength, float radius, float x, float h)
{
    return segmentArea(chordLength, radius) + (x * h / 2);
}