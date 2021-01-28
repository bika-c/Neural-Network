#include "../include/data.h"
#include "../include/graphics.h"
#include <SDL2/SDL.h>

int main(int argc, char* argv[])
{
    SDL_Event event;

    initialize();

    while (true)
    {
        while (SDL_PollEvent(&event))
        {

            switch (event.type)
            {
                case SDL_QUIT:
                    quit();
                    break;
                case SDL_MOUSEBUTTONUP:
                    if (event.button.button == SDL_BUTTON_LEFT)
                        recognize();
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT)
                        refresh(&event);
                case SDL_MOUSEMOTION:
                    if (event.button.button == SDL_BUTTON_LEFT)
                        refresh(&event);
                    break;
                case SDL_KEYDOWN:
                    keyBoardEvent(event.key.keysym);
                    break;
            }
            SDL_RenderPresent(cont.renderer);
        }
    }
    quit();
    return 0;
}