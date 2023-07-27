#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <magick/MagickCore.h>
#include <cuda.h>
#include "upscaler.h"

int main(int argc, char** argv){
    //ensure the proper number of arguments are given
    if(argc < 2 || argc > 4){
        printf("Usage: %s <path to image> <target resolution> [path to destination]\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    char* outputPath;
    if(argc == 4){
        outputPath = argv[3];
    }
    else{
        //make the default output path the current directory
        outputPath = (char*) malloc(sizeof(char)* 2);
        outputPath[0] = '.';
        outputPath[1] = '\0';
    }
    int targetResolution;
    targetResolution = atoi(argv[2]);
    
    
    //Initialize the image info structure and read an image.
    
    ExceptionInfo* exception;
    Image* image;
    Image* images;
    ImageInfo* image_info;

    //read in image from file
    MagickCoreGenesis(*argv,MagickTrue);
    exception=AcquireExceptionInfo();
    image_info=CloneImageInfo((ImageInfo *) NULL);
    (void) strcpy(image_info->filename,argv[1]);
    images=ReadImage(image_info,exception);
    if (exception->severity != UndefinedException){
        CatchException(exception);
    }
    if (images == (Image *) NULL){
        exit(1);
    }

    image = RemoveFirstImageFromList(&images);
    if(image == NULL){
        perror("No image");
        exit(EXIT_FAILURE);
    }

    //upscales the image and returns the upscaled version
    Image* upscaled_image = upscale_image(image, targetResolution, exception);

    Image* saving_list = NewImageList();
    if (upscaled_image == (Image *) NULL){
        MagickError(exception->severity,exception->reason,exception->description);
    }
    (void) AppendImageToList(&saving_list, upscaled_image);

    //Free the memory holding the input image
    DestroyImage(image);

    //write the upscaled image to disk
    (void) strcpy(saving_list->filename, outputPath);
    WriteImage(image_info, upscaled_image);

    //clean up memory
    saving_list=DestroyImageList(saving_list);
    image_info=DestroyImageInfo(image_info);
    exception=DestroyExceptionInfo(exception);
    MagickCoreTerminus();
}

/*
Procedure:           upscale_image
Parameters:          Image* image - pointer to the image to be upscaled
                     int target_resolution - the vertical resolution to which the image should be upscaled
                     ExceptionInfo* exception - ImageCore's way of tracking what might have caused an error
Purpose:             To upscale image to the target vertical resolution while maintaining the aspect ratio.
Produces:            A pointer to an image that has been upscaled.
Preconditions:       image must be loaded into memory via MagickCoreGenesis
                     target resolution must be greater than the vertical resolution of image
Postconditions:      the vertical resolution of the returned image is the smallest value greater than the
                     original resolution that is a power of two times the original resolution.
*/
Image* upscale_image(Image* image, int target_resolution, ExceptionInfo* exception) {
    Image* new_image = image;

    Image* upscaled_image;
    //make a background that can be passed to the NewMagickImage function
    MagickPixelPacket background;
    background.red = 0;
    background.green = 0;
    background.blue = 0;
    int new_height;
    int new_width;

    int currentResolution = image->rows;  
    int currentCols = image->columns;

    //loop to do the upscaling until the current resolution equals or exceeds the target resolution
    while (currentResolution < target_resolution) {

        new_height = 2 * currentResolution;
        new_width = 2 * currentCols;

        //make a new image of the right size for the upscaled image
        ImageInfo* image_info;
        image_info=CloneImageInfo((ImageInfo *) NULL);
        upscaled_image = NewMagickImage(image_info, new_width, new_height, &background); 

        PixelPacket* originalPixels = GetImagePixels(new_image, 0, 0, new_width / 2, new_height / 2);
        PixelPacket* updatedPixels = SetImagePixels(upscaled_image, 0, 0, new_width, new_height);

        PixelPacket* originalPixelsGPU;
        PixelPacket* updatedPixelsGPU;

        //allocate space on the GPU for the not-fully-upscaled image
        if(cudaMalloc(&originalPixelsGPU, sizeof(PixelPacket)*(new_width / 2 * new_height / 2))!= cudaSuccess){
            fprintf(stderr, "Failed to allocate original image on GPU\n");
            exit(2);
        }
        //copy the not-fully-upscaled image to the GPU
        if(cudaMemcpy(originalPixelsGPU, originalPixels, sizeof(PixelPacket)*(new_width / 2 * new_height / 2), cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Failed to copy original image to the GPU\n");
            exit(2);
        }
        //allocate space on the GPU for the upscaled image
        if(cudaMalloc(&updatedPixelsGPU, sizeof(PixelPacket)*(new_width * new_height)) != cudaSuccess){
            fprintf(stderr, "Failed to allocate updated image on GPU\n");
            exit(2);
        }
        //copy the upscaled image to the GPU so that the fields are filled in
        if(cudaMemcpy(updatedPixelsGPU, updatedPixels, sizeof(PixelPacket)*(new_width * new_height), cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Failed to copy updated image to the GPU\n");
            exit(2);
        }
        
        //invoke the kernel
        gpu_upscale<<<new_width, new_height>>>(originalPixelsGPU, updatedPixelsGPU, new_width, new_height);  // THINK ABOUT THIS

        //allows all threads on the GPU to finish before continuing
        if (cudaDeviceSynchronize() != cudaSuccess) {
        fprintf(stderr, "Cuda Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
        }

        //copy the upscaled image from GPU memory to the CPU memory
        if(cudaMemcpy(updatedPixels, updatedPixelsGPU, sizeof(PixelPacket)*(new_width * new_height), cudaMemcpyDeviceToHost) != cudaSuccess) {
            fprintf(stderr, "Failed to copy updated image from the GPU\n");
            exit(2);
        }

        //Save image changes
        if(SyncAuthenticPixels(upscaled_image, exception) == MagickFalse){
            fprintf(stderr, "Failed to save image changes\n");
            exit(2);
        }
        //prepare for the next iteration of the While loop
        new_image = upscaled_image;
        currentResolution *= 2;
        currentCols *= 2;
    }
    return new_image;
}

/*
Procedure:           gpu_upscale
Parameters:          PixelPacket* originalPixels - the array of pixels of the non-upscaled image
                     PixelPacket* updatedPixelsGPU - the array of pixels of the image to be upscaled
                     int new_width - the horizontal resolution of the image whose pixels are being filled in
                     int new_height - the vertical resolution of the image whose pixels are being filled in
Purpose:             To change the colors of pixels in the output image in a manner consistent with the
                     bilinear interpolation algorithm for image upscaling
Produces:            Nothing (void) - the updatedPixelsGPU is modified to be an upscaled image
Preconditions:       originalPixels and updatedPixelsGPU must point to GPU memory.
                     new_width and new_height must be the correct resolutions for the updatedPixelsGPU array
                     (and thus, two times the resolution of the originalPixels array)
Postconditions:      updatedPixelsGPU is updated to be the upscaled image based on the bilinear interpolation
                     algorithm.
*/
__global__ void gpu_upscale(PixelPacket* originalPixels, PixelPacket* updatedPixelsGPU, int new_width, int new_height) {

    int row = threadIdx.x;
    int col = blockIdx.x;
    int scaledOldRow = 2 * row + 1;
    int scaledOldCol = 2 * col + 1;

    int oldRow = scaledOldRow / 4;
    int oldCol = scaledOldCol / 4;

    //if a thread is at an edge or corner where there don't exist enough pixels to calculate what the new
    //pixel's value is, then simply return.
    if(row==0 || row==new_height-1 || col==0 || col==new_width-1){
        return;
    }

    //below is the actual implementation of the bilinear interpolation algorithm.
    if(scaledOldRow % 4 == 1){       // upper
        if(scaledOldCol % 4 == 1){
            // upper left
            float upperRed = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].red + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol - 1)].red;
            float upperGreen = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].green + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol - 1)].green;
            float upperBlue = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].blue + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol - 1)].blue;

            float lowerRed = 0.75 * originalPixels[new_width / 2 * (oldRow - 1) + oldCol].red + 0.25 * originalPixels[new_width / 2 * (oldRow - 1) + (oldCol - 1)].red;
            float lowerGreen = 0.75 * originalPixels[new_width / 2 * (oldRow - 1) + oldCol].green + 0.25 * originalPixels[new_width / 2 * (oldRow - 1) + (oldCol - 1)].green;
            float lowerBlue = 0.75 * originalPixels[new_width / 2 * (oldRow - 1) + oldCol].blue + 0.25 * originalPixels[new_width / 2 * (oldRow - 1) + (oldCol - 1)].blue;

            int actualRed = (int) (0.75 * lowerRed + 0.25 * upperRed);
            int actualGreen = (int) (0.75 * lowerGreen + 0.25 * upperGreen);
            int actualBlue = (int) (0.75 * lowerBlue + 0.25 * upperBlue);

            updatedPixelsGPU[new_width * row + col].red = actualRed;
            updatedPixelsGPU[new_width * row + col].green = actualGreen;
            updatedPixelsGPU[new_width * row + col].blue = actualBlue;
        }else{
            // upper right
            float upperRed = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].red + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol + 1)].red;
            float upperGreen = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].green + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol + 1)].green;
            float upperBlue = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].blue + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol + 1)].blue;

            float lowerRed = 0.75 * originalPixels[new_width / 2 * (oldRow - 1) + oldCol].red + 0.25 * originalPixels[new_width / 2 * (oldRow - 1) + (oldCol + 1)].red;
            float lowerGreen = 0.75 * originalPixels[new_width / 2 * (oldRow - 1) + oldCol].green + 0.25 * originalPixels[new_width / 2 * (oldRow - 1) + (oldCol + 1)].green;
            float lowerBlue = 0.75 * originalPixels[new_width / 2 * (oldRow - 1) + oldCol].blue + 0.25 * originalPixels[new_width / 2 * (oldRow - 1) + (oldCol + 1)].blue;

            int actualRed = (int) (0.75 * lowerRed + 0.25 * upperRed);
            int actualGreen = (int) (0.75 * lowerGreen + 0.25 * upperGreen);
            int actualBlue = (int) (0.75 * lowerBlue + 0.25 * upperBlue);

            updatedPixelsGPU[new_width * row + col].red = actualRed;
            updatedPixelsGPU[new_width * row + col].green = actualGreen;
            updatedPixelsGPU[new_width * row + col].blue = actualBlue;
        }
    }else{                           // lower
        if(scaledOldCol % 4 == 1){
            // lower left
            float upperRed = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].red + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol - 1)].red;
            float upperGreen = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].green + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol - 1)].green;
            float upperBlue = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].blue + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol - 1)].blue;

            float lowerRed = 0.75 * originalPixels[new_width / 2 * (oldRow + 1) + oldCol].red + 0.25 * originalPixels[new_width / 2 * (oldRow + 1) + (oldCol - 1)].red;
            float lowerGreen = 0.75 * originalPixels[new_width / 2 * (oldRow + 1) + oldCol].green + 0.25 * originalPixels[new_width / 2 * (oldRow + 1) + (oldCol - 1)].green;
            float lowerBlue = 0.75 * originalPixels[new_width / 2 * (oldRow + 1) + oldCol].blue + 0.25 * originalPixels[new_width / 2 * (oldRow + 1) + (oldCol - 1)].blue;

            int actualRed = (int) (0.75 * lowerRed + 0.25 * upperRed);
            int actualGreen = (int) (0.75 * lowerGreen + 0.25 * upperGreen);
            int actualBlue = (int) (0.75 * lowerBlue + 0.25 * upperBlue);

            updatedPixelsGPU[new_width * row + col].red = actualRed;
            updatedPixelsGPU[new_width * row + col].green = actualGreen;
            updatedPixelsGPU[new_width * row + col].blue = actualBlue;
        }else{
            // lower right
           float upperRed = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].red + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol + 1)].red;
            float upperGreen = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].green + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol + 1)].green;
            float upperBlue = 0.75 * originalPixels[new_width / 2 * oldRow + oldCol].blue + 0.25 * originalPixels[new_width / 2 * oldRow + (oldCol + 1)].blue;

            float lowerRed = 0.75 * originalPixels[new_width / 2 * (oldRow + 1) + oldCol].red + 0.25 * originalPixels[new_width / 2 * (oldRow + 1) + (oldCol + 1)].red;
            float lowerGreen = 0.75 * originalPixels[new_width / 2 * (oldRow + 1) + oldCol].green + 0.25 * originalPixels[new_width / 2 * (oldRow + 1) + (oldCol + 1)].green;
            float lowerBlue = 0.75 * originalPixels[new_width / 2 * (oldRow + 1) + oldCol].blue + 0.25 * originalPixels[new_width / 2 * (oldRow + 1) + (oldCol + 1)].blue;

            int actualRed = (int) (0.75 * lowerRed + 0.25 * upperRed);
            int actualGreen = (int) (0.75 * lowerGreen + 0.25 * upperGreen);
            int actualBlue = (int) (0.75 * lowerBlue + 0.25 * upperBlue);

            updatedPixelsGPU[new_width * row + col].red = actualRed;
            updatedPixelsGPU[new_width * row + col].green = actualGreen;
            updatedPixelsGPU[new_width * row + col].blue = actualBlue;
        }
    }
}
