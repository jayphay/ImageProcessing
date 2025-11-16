#include <stdio.h>
#include <stdlib.h>

void writePGM(const char* filename, unsigned char* data, int width, int height)
{
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Error writing file %s\n", filename);
        return;
    }

    // Write header: P5 = raw grayscale
    fprintf(f, "P5\n%d %d\n255\n", width, height);

    // Write pixel data
    fwrite(data, 1, width * height, f);
    fclose(f);
}
