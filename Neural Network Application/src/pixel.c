#include "../include/pixel.h"

Pixel  pixelSquares[NUM_PIXEL];

// Map all sacled pixels to screen
void initializePixelSquares()
{
    int x = 0;
    int y = 0;
    for (int i = 0; i < NUM_PIXEL; i++)
    {
        pixelSquares[i].rect.w = PIXEL;
        pixelSquares[i].rect.h = PIXEL;
        pixelSquares[i].rect.x = x;
        pixelSquares[i].rect.y = y;
        if (pixelSquares[i].rect.x == CANVAS_WIDTH - PIXEL)
        {
            y += PIXEL;
            x = 0;
            continue;
        }
        x += PIXEL;
    }
}