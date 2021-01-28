#include "../include/graphics.h"
#include "../include/data.h"
#include "SDL2/SDL2_gfxPrimitives.h"
#include "SDL2/SDL_ttf.h"
#include <stdio.h>
#include <stdlib.h>

extern Pixel  pixelSquares[NUM_PIXEL];
extern Image  image;
extern LeNet5 AI;

// static char outputMessage[29] = "AI thinks this number is:\n 0";
static SDL_Color color      = {255, 255, 255, 255};
static SDL_Rect  background = {CANVAS_WIDTH, 0, 300, HEIGHT};

int        affectedPixels[4];
AppContent cont;

void TextAreaClear()
{
    SDL_SetRenderDrawColor(cont.renderer, 75, 75, 75, 255);
    SDL_RenderFillRect(cont.renderer, &background);
}

float areaPercentage(int coveredArea)
{
    return (float)coveredArea / (PIXEL * PIXEL);
}

// Map/scale the painted area to the big scaled pixels
void scaleScreenPixToPixels()
{
    for (int i = 0; i < 4; i++)
    {
        if (affectedPixels[i] < 0) continue;
        int initialP = pixelSquares[affectedPixels[i]].rect.y * CANVAS_WIDTH + pixelSquares[affectedPixels[i]].rect.x, area = 0;

        for (int row = initialP; row < initialP + CANVAS_WIDTH * PIXEL + PIXEL; row += CANVAS_WIDTH)
        {
            // printf("row: %d, row + %d\n", row, CANVAS_WIDTH);
            for (int column = row; column < row + PIXEL; column++)
            {
                // if (row < record)
                // printf("row: %d, column: %d\n", row, column);
                if (cont.pixels[column] != 0)
                {
                    area++;
                    // printf("area: %d\n", area);
                }
            }
            // printf("row: %d\n", row);
        }
        // printf("area: %d\n", area);
        pixelSquares[affectedPixels[i]].activation = areaPercentage(area);
        // printf("activation: %f\n", pixelSquares[affectedPixels[i]].activation);
    }
}

// Calculate which scaled pixels are affected
void CalculateAffectedPixels(int x, int y)
{
    SDL_memset(affectedPixels, 0, 4);

    double temp = 0;
    double dx   = modf(x / (float)PIXEL, &temp);
    double dy   = modf(y / (float)PIXEL, &temp);

    int relativeIndex = (int)(y / PIXEL) * (int)(PIXELS_WIDTH) + (int)(x / PIXEL);

    affectedPixels[0] = relativeIndex;

    if (dx < 0.5 && dy < 0.5)
    {
        affectedPixels[1] = relativeIndex % PIXELS_WIDTH == 0 ? -1 : relativeIndex - 1;       //Left
        affectedPixels[2] = relativeIndex < PIXELS_WIDTH ? -1 : relativeIndex - PIXELS_WIDTH; //Up
        affectedPixels[3] = affectedPixels[1] == -1 ? -1 : relativeIndex - PIXELS_WIDTH - 1;  //Up-Left
    }
    else if (dx > 0.5 && dy < 0.5)
    {
        affectedPixels[1] = (relativeIndex + 1) % PIXELS_WIDTH == 0 ? -1 : relativeIndex + 1; //Right
        affectedPixels[2] = relativeIndex < PIXELS_WIDTH ? -1 : relativeIndex - PIXELS_WIDTH; //Up
        affectedPixels[3] = affectedPixels[1] == -1 ? -1 : relativeIndex - PIXELS_WIDTH + 1;  //Up-Right
    }
    else if (dx < 0.5 && dy > 0.5)
    {
        affectedPixels[1] = relativeIndex % PIXELS_WIDTH == 0 ? -1 : relativeIndex - 1;                              //Left
        affectedPixels[2] = relativeIndex >= (PIXELS_HEIGHT - 1) * PIXELS_WIDTH ? -1 : relativeIndex + PIXELS_WIDTH; //Down
        affectedPixels[3] = affectedPixels[1] == -1 ? -1 : relativeIndex + PIXELS_WIDTH - 1;                         //Down-Left
    }
    else if (dx > 0.5 && dy > 0.5)
    {
        affectedPixels[1] = (relativeIndex + 1) % PIXELS_WIDTH == 0 ? -1 : relativeIndex + 1;                        //Right
        affectedPixels[2] = relativeIndex >= (PIXELS_HEIGHT - 1) * PIXELS_WIDTH ? -1 : relativeIndex + PIXELS_WIDTH; //Down
        affectedPixels[3] = affectedPixels[1] == -1 ? -1 : relativeIndex + PIXELS_WIDTH + 1;                         //Down-Right
    }
    else if (dx == 0.5 && dy < 0.5)
    {
        affectedPixels[1] = relativeIndex < PIXELS_WIDTH ? -1 : relativeIndex - PIXELS_WIDTH; //Up
        affectedPixels[2] = affectedPixels[3] = -1;
    }
    else if (dx == 0.5 && dy > 0.5)
    {
        affectedPixels[1] = relativeIndex >= (PIXELS_HEIGHT - 1) * PIXELS_WIDTH ? -1 : relativeIndex + PIXELS_WIDTH; //Down
        affectedPixels[2] = affectedPixels[3] = -1;
    }
    else if (dx < 0.5 && dy == 0.5)
    {
        affectedPixels[1] = relativeIndex % PIXELS_WIDTH == 0 ? -1 : relativeIndex - 1; //Left
        affectedPixels[2] = affectedPixels[3] = -1;
    }
    else if (dx > 0.5 && dy == 0.5)
    {
        affectedPixels[1] = (relativeIndex + 1) % PIXELS_WIDTH == 0 ? -1 : relativeIndex + 1; //Right
        affectedPixels[2] = affectedPixels[3] = -1;
    }
    else if (dx == 0.5 && dy == 0.5)
    {
        affectedPixels[0] = relativeIndex;
        affectedPixels[1] = affectedPixels[2] = affectedPixels[3] = -1;
    }
}

// Simulate drawing. Store screen pixels to an array
void drawSimulation(int x, int y)
{
    SDL_SetRenderTarget(cont.renderer, cont.texture);
    filledCircleColor(cont.renderer, x, y, BRUSH_RADIUS, BRUSH_COLOR);

    SDL_RenderReadPixels(cont.renderer, NULL, SCREEN_PIXEL_FORMAT, cont.pixels, sizeof(Uint32) * CANVAS_WIDTH);

    SDL_SetRenderTarget(cont.renderer, NULL);
    // SDL_UpdateTexture(cont.texture, NULL, cont.pixels, sizeof(Uint32) * CANVAS_WIDTH);
    // SDL_RenderCopy(cont.renderer, cont.texture, NULL, &(SDL_Rect) {0, 0, CANVAS_WIDTH, CANVAS_HEIGHT});
}

// Color the big pixels
void UpdatePixelSquares(int x, int y)
{
    drawSimulation(x, y);
    CalculateAffectedPixels(x, y);
    scaleScreenPixToPixels();
    SDL_SetRenderDrawBlendMode(cont.renderer, SDL_BLENDMODE_BLEND);
    for (int i = 0; i < 4; i++)
    {
        SDL_SetRenderDrawColor(cont.renderer, 255, 255, 255, pixelSquares[affectedPixels[i]].activation * 255);
        // printf("activation: %f\n", pixelSquares[affectedPixels[i]].activation);
        SDL_RenderFillRect(cont.renderer, &pixelSquares[affectedPixels[i]].rect);
    }
}

// Call AI
void recognize()
{
    generalizeData();
    int ans = Predict(&AI, image, 10) + 48;
    displayText((char*)&ans);
}

// Refresh the screen
void refresh(SDL_Event* event)
{
    if (event->motion.x <= CANVAS_WIDTH && event->motion.y <= CANVAS_HEIGHT)
        UpdatePixelSquares(event->motion.x, event->motion.y);

    // recognize();
}

void clearScreen()
{
    // SDL_memset(cont.pixels, 45, sizeof(Uint32) * CANVAS_AREA);
    SDL_memset4(cont.pixels, 0, CANVAS_AREA);
    SDL_UpdateTexture(cont.texture, NULL, cont.pixels, sizeof(Uint32) * CANVAS_WIDTH);
    SDL_RenderCopy(cont.renderer, cont.texture, NULL, NULL);
    // SDL_SetRenderDrawColor(cont.renderer, 255, 0, 0, 255);
    for (int i = 0; i < NUM_PIXEL; i++)
    {
        // SDL_RenderDrawRect(cont.renderer, &pixelSquares[i].rect);
        pixelSquares[i].activation = 0;
    }
    displayText("N");
}

void keyBoardEvent(SDL_Keysym key)
{
    // int ans = Predict(&AI, data, 10) + 48;
    switch (key.sym)
    {
        case SDLK_c:
            clearScreen();
            break;
        // case SDLK_0:
        // case SDLK_1:
        // case SDLK_2:
        // case SDLK_3:
        // case SDLK_4:
        // case SDLK_5:
        // case SDLK_6:
        // case SDLK_7:
        // case SDLK_8:
        // case SDLK_9:
        //     generalizeData();
        //     train(key.sym - 48);
        //     recognize();
        //     // clearScreen();
        //     break;
        case SDLK_SPACE:
            recognize();
            break;
    }
}

void initialize()
{
    if (! SearchForDataFile())
        FileNotFound();
    else
        readFromFile();
    TTF_Init();
    cont.window   = SDL_CreateWindow("PIXEL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, SDL_WINDOW_ALLOW_HIGHDPI);
    cont.renderer = SDL_CreateRenderer(cont.window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
    cont.texture  = SDL_CreateTexture(cont.renderer, SCREEN_PIXEL_FORMAT, SDL_TEXTUREACCESS_TARGET, CANVAS_HEIGHT, CANVAS_HEIGHT);
    cont.font     = TTF_OpenFont(FONT_NAME, FONT_SIZE);
    cont.format   = SDL_AllocFormat(SCREEN_PIXEL_FORMAT);
    initializePixelSquares();
    // initializeNetwork();
    SDL_SetRenderDrawBlendMode(cont.renderer, SDL_BLENDMODE_BLEND);
    clearScreen();
}

void errorMessage(const char* const message)
{
    SDL_MessageBoxButtonData  button      = {SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, 1, "OK"};
    SDL_MessageBoxColorScheme colorScheme = {
        {/* .colors (.r, .g, .b) */
         /* [SDL_MESSAGEBOX_COLOR_BACKGROUND] */
         {255, 0, 0},
         /* [SDL_MESSAGEBOX_COLOR_TEXT] */
         {0, 255, 0},
         /* [SDL_MESSAGEBOX_COLOR_BUTTON_BORDER] */
         {255, 255, 0},
         /* [SDL_MESSAGEBOX_COLOR_BUTTON_BACKGROUND] */
         {0, 0, 255},
         /* [SDL_MESSAGEBOX_COLOR_BUTTON_SELECTED] */
         {255, 0, 255}}};
    const SDL_MessageBoxData messageboxdata = {
        SDL_MESSAGEBOX_ERROR, /* .flags */
        cont.window,          /* .window */
        "Error",              /* .title */
        message,              /* .message */
        1,                    /* .numbuttons */
        &button,              /* .buttons */
        &colorScheme          /* .colorScheme */
    };
    int buttonid;
    SDL_ShowMessageBox(&messageboxdata, 0);
}

void displayText(const char* const message)
{
    TextAreaClear();
    SDL_Surface* textSurface = TTF_RenderText_Blended(cont.font, message, color);
    SDL_Texture* text        = SDL_CreateTextureFromSurface(cont.renderer, textSurface);

    int texW = 0;
    int texH = 0;
    SDL_QueryTexture(text, NULL, NULL, &texW, &texH);
    SDL_Rect pos = {TEXT_X_POS, TEXT_Y_POS, texW, texH};
    SDL_RenderCopy(cont.renderer, text, NULL, &pos);

    SDL_DestroyTexture(text);
    SDL_FreeSurface(textSurface);
}

void quit()
{
    SDL_DestroyTexture(cont.texture);
    SDL_DestroyRenderer(cont.renderer);
    SDL_DestroyWindow(cont.window);
    TTF_CloseFont(cont.font);
    // writeToFile();
    TTF_Quit();
    SDL_Quit();
    exit(0);
}