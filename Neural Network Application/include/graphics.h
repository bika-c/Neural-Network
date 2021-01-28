#pragma once

#include "lenet.h"
#include "brush.h"
#include "specialCircleMathTools.h"
#include <SDL2/SDL2_gfxPrimitives.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2\SDL.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#define FONT_NAME  "./asset/arial.ttf"
#define FONT_SIZE  150
#define TEXT_X_POS CANVAS_WIDTH + 120
#define TEXT_Y_POS 220

#define SCREEN_PIXEL_FORMAT SDL_PIXELFORMAT_RGBA8888

typedef struct UI
{
    SDL_Window*      window;
    SDL_Renderer*    renderer;
    SDL_Texture*     texture;
    TTF_Font*        font;
    Uint32           pixels[CANVAS_AREA];
    SDL_PixelFormat* format;
} AppContent;

extern AppContent cont;


// Refresh the screen
void refresh(SDL_Event* event);

// Coresponding key event
void keyBoardEvent(SDL_Keysym key);


// Initialize the program
void initialize();

// Call AI to recognize the screen
void recognize();

// Prompt errors
void errorMessage(const char* const message);

// Display text in the defined area
void displayText(const char* const message);

// Safe quit
void quit();