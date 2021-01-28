#pragma once

#include "screen.h"
#include <SDL2/SDL_rect.h>

#define PIXEL         20
#define PIXELS_WIDTH  (int)(CANVAS_WIDTH / PIXEL)
#define PIXELS_HEIGHT (int)(CANVAS_HEIGHT / PIXEL)
#define NUM_PIXEL     (int)(CANVAS_AREA / (PIXEL * PIXEL))

typedef struct PixelSquares
{
    SDL_Rect rect;
    float    activation;
} Pixel; // Scaled pixel

void initializePixelSquares();