/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define USE_MMAP 0 // only for SEQUENTIAL_TIME_STEPS!=0

#define SEQUENTIAL_TIME_STEPS 0 // if zero, time steps are treated in parallel

#if USE_MMAP
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#endif

#if 0 // this is the stand-alone executable version

int main(int argc, char *argv[])
{
  if (argc != 8) {
    printf("usage: %s <timestep file> <path to data> <elements per data file> <bytes per element (12 for 3-vect)> <first time step (0)> <number of time steps (-1 for all)> <outfile>\n NOTE: if generating multiple files for same component (support for 32bit systems), last timestep of preceding file must also be contained as first step of the next file", argv[0]);
    return 1;
  }

  FILE *fpInfo;
  fpInfo = fopen(argv[1], "r");
  if (!fpInfo) {
    printf("could not open input file %s\n", argv[1]);
    return 1;
  }

#if !USE_MMAP
  FILE *fpOut;
  fpOut = fopen(argv[7], "wb");
  if (!fpOut) {
    printf("could not open output file %s\n", argv[7]);
    return 1;
  }
#else
  int fdOut = open(argv[7], O_RDWR);
  if (fdOut < 1) {
    printf("could not open output file %s\n", argv[7]);
    return 1;
  }
#endif

  // get number of time steps
  int timeStepNb = 0;
  if (atoi(argv[6]) < 1) {
    char buf[1024] = "";
    while (fscanf(fpInfo, "%s", buf)) {
      if (strlen(buf) < 1) break;
      buf[0] = '\0';
      timeStepNb++;
    }
    timeStepNb - atoi(argv[5]);
    printf("found %d time steps\n", timeStepNb);
  }
  else {
    timeStepNb = atoi(argv[6]);
  }

  int firstTimeStep = atoi(argv[5]);

#if USE_MMAP
  float *map = (float *) mmap(0, timeStepNb * atoi(argv[3]) * atoi(argv[4]),
                              PROT_WRITE, MAP_SHARED, fdOut, 0);
  if (map == MAP_FAILED) {
    printf("error mapping %s\n", argv[7]);
    return 1;
  }
#endif

  // go
  int elementNb = atoi(argv[3]);
  int elementSize = atoi(argv[4]);
  fseek(fpInfo, 0, SEEK_SET);
#if SEQUENTIAL_TIME_STEPS
  char buf[1024] = "\n";
  int timeStep = 0;
  
  // skip time steps before first time step
  for (int t=0; t<firstTimeStep; t++) {
    fscanf(fpInfo, "%*s");
  }

  // go
  while (fscanf(fpInfo, "%s", buf) && (timeStep<timeStepNb)) {
    if (strlen(buf) < 1) break;

    // get path to current data file
    char path[1024];
    sprintf(path, "%s/%s", argv[2], buf);
    printf("converting %s\n", path);

    // open current data file
    FILE *fpCurr;
    fpCurr = fopen(path, "rb");
    if (!fpCurr) {
      printf("could not open file %s\n", path);
      return 1;
    }

    // write contents to output file
    for (int e=0; e<elementNb; e++) {

      // read element
      unsigned char el[100];
      fread(&el, elementSize, 1, fpCurr);

      //printf("idx=%d\n", (e * timeStepNb + timeStep) * elementSize);
#if !USE_MMAP
      fseek(fpOut, (e * timeStepNb + timeStep) * elementSize, SEEK_SET);
      fwrite(&el, elementSize, 1, fpOut);
#else
      memcpy(&map[(e * timeStepNb + timeStep) * elementSize], &el, elementSize);
#endif
    }

    fclose(fpCurr);

    // proceed to next entry
    buf[0] = '\0';
    
    timeStep++;
  }
#else
  // open all time step files
  FILE *fpTime[timeStepNb];
  int timeStep = 0;
  char buf[1024] = "\n";

  // skip time steps before first time step
  for (int t=0; t<firstTimeStep; t++) {
    fscanf(fpInfo, "%*s");
  }

  while (fscanf(fpInfo, "%s", buf) && (timeStep<timeStepNb)) {
    if (strlen(buf) < 1) break;
    
    // get path to current data file
    char path[1024];
    sprintf(path, "%s/%s", argv[2], buf);
    printf("opening %s\n", path);

    // open current data file
    fpTime[timeStep] = fopen(path, "rb");
    if (!fpTime[timeStep]) {
      printf("could not open file %s\n", path);
      return 1;
    }

    // proceed to next entry
    buf[0] = '\0';
    
    timeStep++;
  }

  // go over elements
  for (int e=0; e<elementNb; e++) {

    if (!(e % 1000))
    printf("%d%% done (el=%d)   \r", (int) ((((double) e) / elementNb) * 100), e);

    // go over time steps
    for (int t=0; t<timeStepNb; t++) {

      // read element
      unsigned char el[100];
      fread(&el, elementSize, 1, fpTime[t]);

      // write element
      fwrite(&el, elementSize, 1, fpOut);
    }
  }

  // close all time step files
  for (int t=0; t<timeStepNb; t++) {
    fclose(fpTime[t]);
  }
#endif

  fclose(fpInfo);

#if !USE_MMAP
  fclose(fpOut);
#else
  msync(map, timeStepNb * atoi(argv[3]) * atoi(argv[4]), MS_SYNC);
  munmap(map, timeStepNb * atoi(argv[3]) * atoi(argv[4]));
  close(fdOut);
#endif


  return 0;
}

#else

int genMMap(int timeStepNb, int firstTimeStep,
            int elementsPerDataFile,
            int bytesPerElement,
            const char dumpFileNames[][256],
            const char *pathToData,
            const char *outFile)
{ // returns 0 if success

#if !USE_MMAP
    FILE *fpOut;
    fpOut = fopen(outFile, "wb");
    if (!fpOut)
    {
        printf("could not open output file %s\n", outFile);
        return 1;
    }
#else
    int fdOut = open(outFile, O_RDWR);
    if (fdOut < 1)
    {
        printf("could not open output file %s\n", outFile);
        return 1;
    }
#endif

#if 0 // DELETEME
  // get number of time steps
  int timeStepNb = 0;
  if (atoi(argv[6]) < 1) {
    char buf[1024] = "";
    while (fscanf(fpInfo, "%s", buf)) {
      if (strlen(buf) < 1) break;
      buf[0] = '\0';
      timeStepNb++;
    }
    timeStepNb - atoi(argv[5]);
    printf("found %d time steps\n", timeStepNb);
  }
  else {
    timeStepNb = atoi(argv[6]);
  }
#endif

#if 0 // DELETEME
  int firstTimeStep = atoi(argv[5]);
#endif

#if USE_MMAP
    float *map = (float *)mmap(0, timeStepNb * elementsPerDataFile * bytesPerElement,
                               PROT_WRITE, MAP_SHARED, fdOut, 0);
    if (map == MAP_FAILED)
    {
        printf("error mapping %s\n", outFile);
        return 1;
    }
#endif

    // go
    int elementNb = elementsPerDataFile;
    int elementSize = bytesPerElement;
//fseek(fpInfo, 0, SEEK_SET);
#if SEQUENTIAL_TIME_STEPS
    char buf[1024] = "\n";
    int timeStep = 0;

    // skip time steps before first time step
    //for (int t=0; t<firstTimeStep; t++) {
    //  fscanf(fpInfo, "%*s");
    //}

    // go
    while ( //fscanf(fpInfo, "%s", buf) &&
        (timeStep < timeStepNb))
    {
        strcpy(buf, dumpFileNames[firstTimeStep + timeStep]);
        if (strlen(buf) < 1)
            break;

        // get path to current data file
        char path[1024];
        sprintf(path, "%s/%s", pathToData, buf);
        printf("converting %s\n", path);

        // open current data file
        FILE *fpCurr;
        fpCurr = fopen(path, "rb");
        if (!fpCurr)
        {
            printf("could not open file %s\n", path);
            return 1;
        }

        // write contents to output file
        for (int e = 0; e < elementNb; e++)
        {

            // read element
            unsigned char el[100];
            fread(&el, elementSize, 1, fpCurr);

//printf("idx=%d\n", (e * timeStepNb + timeStep) * elementSize);
#if !USE_MMAP
            fseek(fpOut, (e * timeStepNb + timeStep) * elementSize, SEEK_SET);
            fwrite(&el, elementSize, 1, fpOut);
#else
            memcpy(&map[(e * timeStepNb + timeStep) * elementSize], &el, elementSize);
#endif
        }

        fclose(fpCurr);

        // proceed to next entry
        buf[0] = '\0';

        timeStep++;
    }
#else
    // open all time step files
    FILE *fpTime[timeStepNb];
    int timeStep = 0;
    char buf[1024] = "\n";

    // skip time steps before first time step
    //for (int t=0; t<firstTimeStep; t++) {
    //  fscanf(fpInfo, "%*s");
    //}

    while ( //fscanf(fpInfo, "%s", buf) &&
        (timeStep < timeStepNb))
    {
        strcpy(buf, dumpFileNames[firstTimeStep + timeStep]);
        if (strlen(buf) < 1)
            break;

        // get path to current data file
        char path[1024];
        sprintf(path, "%s/%s", pathToData, buf);
        printf("opening %s\n", path);

        // open current data file
        fpTime[timeStep] = fopen(path, "rb");
        if (!fpTime[timeStep])
        {
            printf("could not open file %s\n", path);
            return 1;
        }

        // proceed to next entry
        buf[0] = '\0';

        timeStep++;
    }

    // go over elements
    for (int e = 0; e < elementNb; e++)
    {

        if (!(e % 1000))
            printf("%d%% done (el=%d)   \r", (int)((((double)e) / elementNb) * 100), e);

        // go over time steps
        for (int t = 0; t < timeStepNb; t++)
        {

            // read element
            unsigned char el[100];
            fread(&el, elementSize, 1, fpTime[t]);

            // write element
            fwrite(&el, elementSize, 1, fpOut);
        }
    }

    // close all time step files
    for (int t = 0; t < timeStepNb; t++)
    {
        fclose(fpTime[t]);
    }
#endif

//fclose(fpInfo);

#if !USE_MMAP
    fclose(fpOut);
#else
    msync(map, timeStepNb * elementsPerDataFile * bytesPerElement, MS_SYNC);
    munmap(map, timeStepNb * elementsPerDataFile * bytesPerElement);
    close(fdOut);
#endif

    printf("done                                \n");

    return 0;
}
#endif

int generateMMapFile(int timeStepNb, char dumpFileNames[][256],
                     int nnodes,
                     int maxMMapFileSize, const char *outputPath)
{
    // future: make these params
    char COMPONENT_NAME[256] = "Velocity";
    char COMPONENT_DESC[256] = "velo";

    int bytesPerElement = 12; // hard-coded 3-float vector
    int dumpFilesPerMMFile;
    if (maxMMapFileSize > 0)
    {
        dumpFilesPerMMFile = maxMMapFileSize / (nnodes * bytesPerElement);
    }
    else
    {
        dumpFilesPerMMFile = timeStepNb;
    }
    printf("time steps per mmap file: %d\n", dumpFilesPerMMFile);
    int mmFileCnt;
    if (dumpFilesPerMMFile > 1)
    {
        mmFileCnt = timeStepNb / (dumpFilesPerMMFile - 1);
        if (timeStepNb % (dumpFilesPerMMFile - 1) > 0)
            mmFileCnt++;
    }
    else
    {
        mmFileCnt = timeStepNb;
    }

    FILE *fp;
    char descName[256];
    sprintf(descName, "%s/data.info", outputPath);
    fp = fopen(descName, "w");
    if (!fp)
    {
        return 0;
    }

    fprintf(fp, "1\n%s\n%d\n", COMPONENT_NAME, mmFileCnt);

    for (int i = 0; i < timeStepNb; i += (dumpFilesPerMMFile - 1 > 0 ? dumpFilesPerMMFile - 1 : 1))
    {

        int start = i;
        int end = i + dumpFilesPerMMFile - 1;
        if (end > timeStepNb - 1)
        {
            end = timeStepNb - 1;
        }

        fprintf(fp, "%s_%d-%d.bin\n", COMPONENT_DESC, start, end);

        fprintf(fp, "%d\n", end - start + 1);

        for (int n = start; n <= end; n++)
        {
            fprintf(fp, "%s\n", dumpFileNames[n]);
        }

        // generate mmap file
        char outFile[256];
        sprintf(outFile, "%s/%s_%d-%d.bin", outputPath, COMPONENT_DESC, start, end);
        if (genMMap(end - start + 1,
                    start,
                    nnodes, // elementsPerDataFile,
                    bytesPerElement, // bytesPerElement,
                    dumpFileNames,
                    outputPath, // pathToData,
                    outFile))
        {
            return 0;
        }
    }

    fclose(fp);

    return 1; // success
}
