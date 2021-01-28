#pragma once

#include "pixel.h"
#include <corecrt_math_defines.h>

#define PI           M_PI
#define BRUSH_RADIUS PIXEL / 2.0
#define BRUSH_AREA   PI* BRUSH_RADIUS* BRUSH_RADIUS
#define BRUSH_COLOR  4294967295 //WHITE