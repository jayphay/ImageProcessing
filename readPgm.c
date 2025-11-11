unsigned char* readPGM(const char* filename, int* width, int* height) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    char format[3];
    fscanf(f, "%2s", format);

    if (strcmp(format, "P5") != 0) {
        fclose(f);
        return NULL;
    }

    fscanf(f, "%d %d", width, height);
    int maxVal;
    fscanf(f, "%d", &maxVal);
    fgetc(f); // skip newline

    int size = (*width) * (*height);
    unsigned char* data = (unsigned char*)malloc(size);
    fread(data, 1, size, f);
    fclose(f);
    return data;
}
