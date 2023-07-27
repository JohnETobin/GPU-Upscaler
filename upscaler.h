#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

Image* upscale_image(Image* image, int target_resolution, ExceptionInfo* exception);

__global__ void gpu_upscale(PixelPacket* originalPixels, PixelPacket* updatedPixels, int new_width, int new_height);
