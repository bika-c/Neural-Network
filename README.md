# Hand Written Digit Recognition

- Download the entire folder.
- Left-click to draw.
- Auto-feedback.

![](https://image.alkaid.cloud/Github/Neural-Network/demo.gif)

Source code is in the Neural Network Application folder. Requires third-party library to compile:

- SDL2
- SDL2-ttf
- SDL2-gfx

Compile command:  
`gcc main.c lenet.c mnist.c data.c graphics.c mathTools.c pixel.c -W -O2 -std=c11 -I{PATH TO YOUR include FOLDER} -L{PATH TO YOUR lib FOLDER} -lSDL2main -lSDL2 -lSDL2_ttf -lSDL2_gfx -mwindows -o 'main.exe'`
