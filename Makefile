CC := nvcc
CFLAGS := -g -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16\
 -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -I/usr/include/x86_64-linux-gnu//ImageMagick-6\
  -I/usr/include/ImageMagick-6 -I/usr/include/x86_64-linux-gnu//ImageMagick-6 -I/usr/include/ImageMagick-6\
   -lMagickWand-6.Q16 -lMagickCore-6.Q16

all: upscaler

clean:
	rm -f upscaler

upscaler: upscaler.cu upscaler.h Makefile
	$(CC) $(CFLAGS) -o upscaler upscaler.cu

format:
	@echo "Reformatting source code."
	@clang-format -i --style=file $(wildcard *.c) $(wildcard *.h) $(wildcard *.cu)
	@echo "Done."

.PHONY: all clean zip format
