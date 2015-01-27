
#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#endif

#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda.h>
#if( __CUDA_API_VERSION >= 5000)
#include <gpu/helper_cuda.h>
#else
#include <gpu/cutil.h>
#endif

#include <cudpp.h>

#include <map>
#include <string>

#include "cudaEngine.h"

#include <config/CoviseConfig.h>
#ifdef COVISE
#include <do/coDistributedObject.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#else
enum ELEM_TYPE {
   TYPE_NONE = 0,
   TYPE_BAR = 1,
   TYPE_TRIANGLE = 2,
   TYPE_QUAD = 3,
   TYPE_TETRAHEDER = 4,
   TYPE_PYRAMID = 5,
   TYPE_PRISM = 6,
   TYPE_HEXAGON = 7,
   TYPE_HEXAEDER = 7,
   TYPE_POINT = 10,
   TYPE_POLYHEDRON = 11
};
#endif

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <GL/glew.h>
#include <sysdep/opengl.h>
#include <cuda_gl_interop.h>

#include "cudaState.h"
#include "tables.h"
#include "cudaCommon.h"

//#include "cuPrintf.cu"

#include "cudaCommon_kernels.cu"
#include "cudaEngine_kernels.cu"

#include <gpu/CUDAResourceManager.h>

CUDPPHandle cudpp;

void getMinMax(const float *data, int numElem, float *min, 
               float *max, float minV = -FLT_MAX, float maxV = FLT_MAX)
{
   int i;

   for (i = 0; i < numElem; i++)
   {
      register float actVal = data[i];
      if (actVal >= minV && actVal < *min)
         *min = actVal;
      if (actVal <= maxV && actVal > *max)
         *max = actVal;
   }
}

void countBins(const float *data, int numElem, float min, 
               float max, int numBins, int *bins)
{
   int i;

   float delta = max - min;
   
   for (i = 0; i< numElem; i++)
   {
      register float actData = data[i];
      
      // do not care about FLT_MAX elements in min/max calc.
      if (actData < FLT_MAX)
      {
         int binNo = (int) ((actData  -min) / delta * (numBins - 0.00000001));
         
         if (binNo >= 0 && binNo < numBins)
            ++bins[binNo];
      }
   }
}

void removeSpikesAdaptive(const float *data, int numElem,
                          float *min, float *max)
{
   int i;
   int numBins = 50;//numBinsAdaptive;

   int *bin = new int[numBins];

   // recursively do algoritm until either concergent of cutoff-Limits (top/bot)
   // are reaches
   int numValuesOverall = -1;
   int cutoffLeftTop=1;
   int cutoffLeftBot=1;
   bool foundSpikes;

   do
   {
      foundSpikes=false;
      // count in bins
      for (i=0; i<numBins;i++) bin[i]=0;
      countBins(data, numElem, *min, *max, numBins, bin);
      int numValues = 0;
      for ( i=0; i<numBins;i++)
         numValues+= bin[i];

      // 1st run : set global values
      if (numValuesOverall<0)
      {
         numValuesOverall = numValues;
         cutoffLeftBot = (int) (numValuesOverall * 0.05);
         cutoffLeftTop = (int) (numValuesOverall * 0.05);

         if (cutoffLeftTop<1)
            cutoffLeftTop=1;
         if (cutoffLeftBot<1)
            cutoffLeftBot=1;
      }

      // -------------- TOP CLIP RUN --------------
      // Start at top, find 1st empty bin, count members in bin
      int topBin = numBins-1;
      int numTopClip=0;
      while (    topBin>0
         && bin[topBin] != 0
         && numTopClip+bin[topBin] <= cutoffLeftTop )
      {
         numTopClip += bin[topBin];
         --topBin;
      }

      // if we found an empty box before the # of spike values ran out, this was a spike
      if ( bin[topBin] == 0 )
      {
         // continue all empty bins
         while ( topBin>0 && bin[topBin] == 0 )
         {
            --topBin;
         }
      }
      else
         topBin=numBins-1;
      // ==> topBin now points to non-empty bin at top

      // -------------- BOT CLIP RUN --------------
      // Start at bot, find 1st empty bin, count members in bin
      int botBin = 0;
      int numBotClip=0;
      while (    botBin<numBins
         && bin[botBin] != 0
         && numBotClip+bin[botBin] <= cutoffLeftBot )
      {
         numBotClip += bin[botBin];
         ++botBin;
      }

      // if we found an empty box before the # of spike values ran out, this was a spike
      if ( bin[botBin] == 0 )
      {
         // continue all empty bins
         while ( botBin<numBins && bin[botBin] == 0 )
            ++botBin;
      }
      else
         botBin=0;
      // ==> botBin now points to non-empty bin at top

      float newMin = *min;
      float newMax = *max;

      // -------------- BOT CLIP --------------
      if (    ( botBin!=0)                        // we found any top clip
         && ( botBin < numBins )                  // we cannot exeed top
         && ( topBin >= botBin )                  // at least one box is left
         && ( cutoffLeftBot >= numBotClip) )      // don't clip away more
      {
                                                  // set new min
         newMin = *min + (*max-*min)/numBins * botBin;
         cutoffLeftBot -= numBotClip;             // book off spikes
         foundSpikes=true;
      }

      // -------------- TOP CLIP --------------
      if (    ( topBin<numBins-1)                 // we found any top clip
         && ( topBin >= 0 )                       // we cannot hit ground
         && ( topBin >= botBin )                  // at least one box is left
         && ( cutoffLeftTop >= numTopClip) )      // don't clip away more
      {
                                                  // set new max
         newMax = *min + (*max-*min)/numBins * (topBin+1);
         cutoffLeftTop -= numTopClip;             // book off spikes
         foundSpikes=true;
      }

      *max = newMax;
      *min = newMin;
   }
   while (foundSpikes);

   float newMin = FLT_MAX;
   float newMax = -FLT_MAX;

   // at last: use higest/lowest real data values instead of bin boundaries
   getMinMax(data, numElem, &newMin, &newMax, *min, *max);

   *min = newMin;
   *max = newMax;
}

unsigned int upPow2(unsigned int x)
{
   x --;
   x |= x >> 1;
   x |= x >> 2;
   x |= x >> 4;
   x |= x >> 8;
   x |= x >> 16;
   return ++x;
}

struct TypeInfo tm[11] = { 
   {-1, 0, 0 },
   {-1, 0, 0 },
   {-1, 0, 0 },
   {-1, 0, 0 },
   { 0, 6, 4 },
   { 1, 8, 5 },
   {-1, 0, 0 },
   { 2, 12, 6 },
   {-1, 0, 0 },
   {-1, 0, 0 },
   {-1, 0, 0 } 
};

void CheckErr(const char *where)
{
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
      fprintf(stderr, "CUDA error: %s [%s]\n", cudaGetErrorString(err), where);
      //exit(-1);
   }
}


void InitCUDA(CUDAState* state, int device)
{
   GLenum glewerr = glewInit();
   if (glewerr != GLEW_OK) {

      fprintf(stderr, "cudaEngine.cu:InitCuda(): GLEW initialization failed: %s\n", glewGetErrorString(glewerr));
      return;
   }
   cudaSetDevice(device);
   cudaGLSetGLDevice(device);
   struct cudaDeviceProp cudaDevicesProperty;
   cudaGetDeviceProperties(&cudaDevicesProperty, device);
   fprintf(stderr," maxNumThreads %d\n",cudaDevicesProperty.maxThreadsPerBlock);
   fprintf(stderr," maxThreadsDim[0] %d\n", cudaDevicesProperty.maxThreadsDim[0]);
   fprintf(stderr," maxThreadsDim[1] %d\n", cudaDevicesProperty.maxThreadsDim[1]);
   fprintf(stderr," maxThreadsDim[2] %d\n", cudaDevicesProperty.maxThreadsDim[2]);
   fprintf(stderr," maxGridSize[0] %d\n", cudaDevicesProperty.maxGridSize[0]);
   fprintf(stderr," maxGridSize[1] %d\n", cudaDevicesProperty.maxGridSize[1]);
   fprintf(stderr," maxGridSize[2] %d\n", cudaDevicesProperty.maxGridSize[2]);
   cudaError_t err = cudaGetLastError(); // ignore cudaErrorSetOnActiveProcess
   
#ifdef __APPLE__
   // don't use coCoviseConfig::getFloat as it depends on std::string
   // - won't link when .cu is compiled with GCC and .cpp with Clang
   state->THREAD_N = 192;
   state->THREAD_N_FAT = 32;
#else
   state->THREAD_N = covise::coCoviseConfig::getInt("COVER.CudaNumThreads", 192);
   state->THREAD_N_FAT = covise::coCoviseConfig::getInt("COVER.CudaNumThreadsFat", 32);
#endif
   fprintf(stderr,"numThreads: %d\n",state->THREAD_N);
   
   // allocate textures for Marching Cubes tables
   cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

   CUDA_SAFE_CALL(cudaMalloc((void **) &(state->d_hexaTriTable), 256*16*sizeof(uint)));
   CUDA_SAFE_CALL(cudaMemcpy( state->d_hexaTriTable, (void *)hexaTriTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaBindTexture(0, hexaTriTex, state->d_hexaTriTable, channelDesc) );

   CUDA_SAFE_CALL(cudaMalloc((void **) &state->d_hexaNumVertsTable, 256*sizeof(uint)));
   CUDA_SAFE_CALL(cudaMemcpy(state->d_hexaNumVertsTable, hexaNumVertsTable, 256*sizeof(uint), cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaBindTexture(0, hexaNumVertsTex, state->d_hexaNumVertsTable, channelDesc) );

   CUDA_SAFE_CALL(cudaMalloc((void **) &state->d_tetraTriTable, 16*6*sizeof(uint)));
   CUDA_SAFE_CALL(cudaMemcpy(state->d_tetraTriTable, tetraTriTable, 16*6*sizeof(uint), cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaBindTexture(0, tetraTriTex, state->d_tetraTriTable, channelDesc) );

   CUDA_SAFE_CALL(cudaMalloc((void **) &state->d_tetraNumVertsTable, 16*sizeof(uint)));
   CUDA_SAFE_CALL(cudaMemcpy(state->d_tetraNumVertsTable, tetraNumVertsTable, 16*sizeof(uint), cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaBindTexture(0, tetraNumVertsTex, state->d_tetraNumVertsTable, channelDesc) );

   CUDA_SAFE_CALL(cudaMalloc((void **) &state->d_pyrTriTable, 32*12*sizeof(uint)));
   CUDA_SAFE_CALL(cudaMemcpy(state->d_pyrTriTable, pyrTriTable, 32*12*sizeof(uint), cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaBindTexture(0, pyrTriTex, state->d_pyrTriTable, channelDesc) );

   CUDA_SAFE_CALL(cudaMalloc((void **) &state->d_pyrNumVertsTable, 32*sizeof(uint)));
   CUDA_SAFE_CALL(cudaMemcpy(state->d_pyrNumVertsTable, pyrNumVertsTable, 32*sizeof(uint), cudaMemcpyHostToDevice) );
   CUDA_SAFE_CALL(cudaBindTexture(0, pyrNumVertsTex, state->d_pyrNumVertsTable, channelDesc) );
   CheckErr("allocate");

   //cudaPrintfInit();

  /* CUDPPResult result = CUDPP_SUCCESS;
   result = cudppCreate(&cudpp);
   if (result != CUDPP_SUCCESS)
      fprintf(stderr, "cudaEngine: Error initializing CUDPP Library.\n");*/
}

void CleanupCUDA(CUDAState* state)
{
   // TODO
  // cudppDestroy(cudpp);
}

#ifdef RENDER_STATE
void RenderCUDAState(State* state)
{
   glPushAttrib(GL_ALL_ATTRIB_BITS);
   glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

   glEnableClientState(GL_VERTEX_ARRAY);
   glBindBuffer(GL_ARRAY_BUFFER, state->vertexBuffer);
   glVertexPointer(4, GL_FLOAT, 0, 0);

   glEnableClientState(GL_NORMAL_ARRAY);
   glBindBuffer(GL_ARRAY_BUFFER, state->normalBuffer);
   glNormalPointer(GL_FLOAT, sizeof(float) * 4, 0);

   glEnableClientState(GL_TEXTURE_COORD_ARRAY);
   glClientActiveTextureARB(GL_TEXTURE0_ARB);
   glEnable(GL_TEXTURE_1D);
   glDisable(GL_TEXTURE_2D);
   glDisable(GL_TEXTURE_3D);
   glBindBuffer(GL_ARRAY_BUFFER, state->texcoordBuffer);
   glTexCoordPointer(1, GL_FLOAT, 0, 0);

#ifdef LIC
   glClientActiveTextureARB(GL_TEXTURE1_ARB);
   glEnable(GL_TEXTURE_2D);
   glBindBuffer(GL_ARRAY_BUFFER, state->licTexcoordBuffer);
   glTexCoordPointer(2, GL_FLOAT, 0, 0);
   glEnableClientState(GL_TEXTURE_COORD_ARRAY);
   glBindBuffer(GL_ARRAY_BUFFER, state->licResultBuffer);
   glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 0, 0);
   glEnableVertexAttribArray(6);
#endif

   if (state->activeVerts > 0) {
      //printf("render %d vertices\n", state->activeVerts); 
      glDrawArrays(GL_TRIANGLES, 0, state->activeVerts);
   }

   glPopClientAttrib();
   glPopAttrib();
}
#endif

void CleanupState(State *state)
{
   if (state->usg)
      CUDAResourceManager::getInstance()->deleteObject(state->usg->getName());
   if (state->data)
      CUDAResourceManager::getInstance()->deleteObject(state->data->getName());
   if (state->mapping)
      CUDAResourceManager::getInstance()->deleteObject(state->mapping->getName());
#ifdef LIC
   if (state->licMapping)
      CUDAResourceManager::getInstance()->deleteObject(state->licMapping->getName());
#endif

   cudppDestroyPlan(state->scanplan);
   CUDA_SAFE_CALL(cudaFree(state->d_elemClassification));
   CUDA_SAFE_CALL(cudaFree(state->d_elemVerts));
   
   CUDA_SAFE_CALL(cudaFree(state->d_scan));
   CUDA_SAFE_CALL(cudaFree(state->d_vertsScan));    
   CUDA_SAFE_CALL(cudaFree(state->d_compactedArray));
   
   deleteVBO(&state->vertexBuffer);
   deleteVBO(&state->normalBuffer);
   deleteVBO(&state->texcoordBuffer);
#ifdef LIC
   deleteVBO(&state->licResultBuffer);
   deleteVBO(&state->licTexcoordBuffer);
#endif

   free(state);
}

void updateState(State *state, int *typeList, int *elemList, int *connList,
                 float *x, float *y, float *z, int numElem, int numConn,
                 int numCoord, float *data)
{
   int *tl, *el, *cl;
   float *gv, *gdata;
   
   tl = state->usg->getTypeList();
   el = state->usg->getElemList();
   cl = state->usg->getConnList();
   
   gv = state->usg->getVertices();
   gdata = state->data->getData();

   /*
   CUDA_SAFE_CALL(cudaMemcpy(tl, typeList, numElem * sizeof(int),
                             cudaMemcpyHostToDevice));


   CUDA_SAFE_CALL(cudaMemcpy(cl, connList, numConn * sizeof(int),
                             cudaMemcpyHostToDevice));

   CUDA_SAFE_CALL(cudaMemcpy(el, elemList, numElem * sizeof(int),
                             cudaMemcpyHostToDevice));


   CUDA_SAFE_CALL(cudaMemcpy(gv, x, numCoord * sizeof(float),
                             cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(gv + numCoord, y, numCoord * sizeof(float),
                             cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(gv + 2 * numCoord, z, numCoord * sizeof(float),
                             cudaMemcpyHostToDevice));
   */
   CUDA_SAFE_CALL(cudaMemcpy(gdata, data, numCoord * sizeof(float),
                             cudaMemcpyHostToDevice));

}

State* InitStateCUDA(CUDAState *cudaState, const char *gridName,
                     const char *dataName, const char *mapName,
                     const int* typeList, const int* elemList, const int* connList, 
                     const float* x, const float* y, const float* z,
                     int numElem, int numConn, int numCoord,
                     const float* data, const float *map_x, const float *map_y, const float *map_z,
                     float min = 0.0, float max = 0.0, int numElemM = 0,
                     int numConnM = 0, int numCoordM = 0)
{
    State *state = (State *) malloc(sizeof(struct State));
    memset(state, 0, (sizeof(struct State)));

    unsigned int elemMemSize;
    if (numElemM)
       elemMemSize = sizeof(uint) * numElemM;
    else
       elemMemSize = sizeof(uint) * numElem;

    state->usg = CUDAResourceManager::getInstance()->addUSG(gridName, numElem, numConn, numCoord, typeList, elemList, connList, x, y, z, numElemM, numConnM, numCoordM);

    state->data = CUDAResourceManager::getInstance()->addScalar(dataName, numCoord, data, numCoordM);

#ifdef STREAMS
    CUDA_SAFE_CALL(cudaStreamCreate(&state->stream));
    CUDA_SAFE_CALL(cudaStreamCreate(&state->cstream));
    state->buffer = CUDAResourceManager::getInstance()->addScalar("buffer", numCoord, data, numCoordM);
#endif

    state->texMin = FLT_MAX;
    state->texMax = -FLT_MAX;
    
    if (map_x && map_y && map_z) {

       float *smem = (float *) malloc(numCoord * sizeof(float));
#ifdef LIC
       float *vmem = (float *) malloc(numCoord * sizeof(float) * 3);
#endif
       for (int index = 0; index < numCoord; index ++) {

          float v = sqrt((map_x[index] * map_x[index] +
                          map_y[index] * map_y[index] +
                          map_z[index] * map_z[index]));
          smem[index] = v;
#ifdef LIC
          vmem[index * 3] = map_x[index];
          vmem[index * 3 + 1] = map_y[index];
          vmem[index * 3 + 2] = map_z[index];
#endif
       }

       if (min == max) {
          getMinMax(smem, numCoord, &state->texMin, &state->texMax);
          removeSpikesAdaptive(smem, numCoord, &state->texMin, &state->texMax);
       } else {
          state->texMin = min;
          state->texMax = max;
       }

       state->mapping = CUDAResourceManager::getInstance()->addScalar(mapName, numCoord, smem, numCoordM);
#ifdef LIC
       char *licName = new char[128];
       #ifdef WIN32
       _snprintf(licName, 128, "v%s", mapName);
       #else
       snprintf(licName, 128, "v%s", mapName);
       #endif
       state->licMapping = CUDAResourceManager::getInstance()->addScalar(licName, numCoord * 3, vmem, numCoordM);
       free(vmem);
#endif
       free(smem);
       
    } else {
       state->mapping = 0;
       state->licMapping = 0;
    }

    // Allocate additional data structures for scans and intermediate steps

    // scan kernel data
    CUDA_SAFE_CALL(cudaMalloc((void**) &state->d_scan, elemMemSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &state->d_vertsScan, elemMemSize));
/*
    CUDA_SAFE_CALL(cudaMallocPitch(&state->d_scan, &state->d_scanPitch, elemMemSize, 2));
    state->d_vertsScan = state->d_scan + state->d_scanPitch;
*/
    // classification kernel data
    CUDA_SAFE_CALL(cudaMalloc((void**) &state->d_elemClassification, elemMemSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &state->d_elemVerts, elemMemSize));
    CUDA_SAFE_CALL(cudaMemset((void*) state->d_elemClassification, 0, elemMemSize));
    CUDA_SAFE_CALL(cudaMemset((void*) state->d_elemVerts, 0, elemMemSize));

    CUDA_SAFE_CALL(cudaMalloc((void **) &state->d_compactedArray, elemMemSize));

    //CUDPP
    CUDPPConfiguration config;
    config.op = CUDPP_ADD;
    config.datatype = CUDPP_UINT;
    config.algorithm = CUDPP_SCAN;
    config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

    if (numElemM)
       cudppPlan(cudpp, &state->scanplan, config, numElemM, 2,
                 state->d_scanPitch);
    else
       cudppPlan(cudpp, &state->scanplan, config, numElem, 2,
                 state->d_scanPitch);

    CheckErr("InitState");
    return state;
}

/*

#ifdef COVISE
State * InitState(covise::coDoUnstructuredGrid *grid,
                  covise::coDoFloat *data,
                  covise::coDoVec3 *mapping,
                  float min = 0.0, float max = 0.0)
{
   int numElem, numConn, numCoord;
   float *x = NULL, *y = NULL, *z = NULL;
   int *elemList = NULL, *connList = NULL, *typeList = NULL;
   float *values = NULL, *map_x = NULL, *map_y = NULL, *map_z = NULL;
   
   if (!grid || !values)
      return NULL;
   grid->getAddresses(&elemList, &connList, &x, &y, &z);
   grid->getGridSize(&numElem, &numConn, &numCoord);

   return InitState(grid->getName(),
                    typeList, elemList, connList, x, y, z,
                    numElem, numConn, numCoord, values,
                    map_x, map_y, map_z, min, max);
}
#endif
*/

/////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
/////////////////////////////////////////////////////////////////////////////

/*
 * 1. each thread classifies one volume element
 *     - will there be vertices generated for this element (1/0)
 *           -> d_elemClassification
 *     - how many vertices -> d_elemVerts
 * 2. d_elemClassification & d_elemVerts are prefix-summed
 *     -> d_scan & d_VertsScan
 * 3. #sum(d_scan) threads are launched, each one generates vertices for
 *      iso / cutting surface. d_VertsScan[threadId] tells the thread where
 *      to write its vertices in vertexBuffer / normalBuffer / texcoordBuffer
 */
#ifdef CUDA_ISO
void computeIsoMeshCUDA(CUDAState* state, State *iso, float isoValue,
                        int* numVertices, float** vertices, float** normals,
                        void *streamBuffer)
#else
   void computeCuttingMeshCUDA(CUDAState* state, State *iso, float *rot,
                        float nx, float ny, float nz, float dist,
                        int* numVertices, float** vertices, float** normals,
                        void *streamBuffer)
#endif
{
#ifdef STREAMS
   if (streamBuffer) {
      cudaMemcpyAsync(iso->buffer->getData(), streamBuffer, iso->buffer->getNumElem() * sizeof(float), cudaMemcpyHostToDevice, iso->cstream);
   }
#endif

   int numElem = iso->usg->getNumElem();
   int blockN;
   {
      dim3 block(512);
      dim3 grid((iso->usg->getNumElem() + block.x - 1) / block.x / 1, 1);
#ifdef CUDA_ISO
      printf("classifyIsoElements block (%d), grid (%d, %d)\n", block.x,
             grid.x, grid.y);
      classifyIsoElements<<<grid, block, 0, iso->stream>>>(
         iso->d_elemVerts, iso->d_elemClassification,
         iso->usg->getTypeList(), iso->usg->getElemList(),
         iso->usg->getConnList(), numElem,
         iso->data->getData(), isoValue);
#else
      classifyCuttingElements<<<grid, block, 0, iso->stream>>>(
         iso->d_elemVerts, iso->d_elemClassification,
         iso->usg->getTypeList(), iso->usg->getElemList(),
         iso->usg->getConnList(), numElem,
         iso->usg->getVertices(), iso->usg->getNumCoord(),
         iso->data->getData(), nx, ny, nz, dist);
#endif
      CheckErr("cudaEngine classify");
   }

   CUDA_SAFE_CALL(cudaThreadSynchronize());
   
   //CUDPP
   //cudppMultiScan(iso->scanplan, (void*) iso->d_scan, (void*) iso->d_elemClassification, iso->numElem, 2);
   cudppScan(iso->scanplan, (void*) iso->d_scan, (void*) iso->d_elemClassification, numElem);
   cudppScan(iso->scanplan, (void*) iso->d_vertsScan, (void*) iso->d_elemVerts, numElem);
   CheckErr("cudaEngine scan");
   
   // read back values to calculate total number of elements
   // since we are using an exclusive scan, the total is the last value of
   // the scan result plus the last value in the input array
   {
      uint lastElement, lastScanElement;
      CUDA_SAFE_CALL(cudaMemcpy((void*) &lastElement,
                                (void*) (iso->d_elemClassification + numElem - 1),
                                sizeof(uint), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy((void*) &lastScanElement,
                                (void*) (iso->d_scan + numElem - 1),
                                sizeof(uint), cudaMemcpyDeviceToHost));
      iso->activeElements = lastElement + lastScanElement;
      
      CUDA_SAFE_CALL(cudaMemcpy((void*) &lastElement,
                                (void*) (iso->d_elemVerts + numElem - 1),
                                sizeof(uint), cudaMemcpyDeviceToHost));
      CUDA_SAFE_CALL(cudaMemcpy((void*) &lastScanElement,
                                (void*) (iso->d_vertsScan + numElem - 1),
                                sizeof(uint), cudaMemcpyDeviceToHost));
      iso->activeVerts = lastElement + lastScanElement;
   }
   CheckErr("cudaEngine memcpy");

   uint offset = 0;
   *numVertices = iso->activeVerts;
   
#ifdef USE_VBO
   // map OpenGL buffer object for writing from CUDA
   float4 *d_vertexBuffer;
   float4 *d_normalBuffer;
   float *d_texcoordBuffer;
#ifdef LIC
   float3 *d_licResultBuffer;
   float2 *d_licTexcoordBuffer;
#endif
  
   if (iso->bufferSize < iso->activeVerts * 4 * sizeof(float)) {

      if (iso->bufferSize) {
         CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(iso->vertexBufferResource));
         CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(iso->normalBufferResource));
         CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(iso->texcoordBufferResource));
#ifdef LIC
         CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(iso->licResultBufferResource));
         CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(iso->licTexcoordBufferResource));
         glDeleteBuffers(1, &iso->licResultBuffer);
         glDeleteBuffers(1, &iso->licTexcoordBuffer);
#endif
         glDeleteBuffers(1, &iso->vertexBuffer);
         glDeleteBuffers(1, &iso->normalBuffer);
         glDeleteBuffers(1, &iso->texcoordBuffer);
      }

      uint memSize = (uint) (iso->activeVerts * 4 * sizeof(float) * 4.5);
      iso->bufferSize = memSize;
      printf("#create VBO: %d\n", memSize);

      /*
      glGenBuffers(1, &iso->vertexBuffer);
      glBindBuffer(GL_ARRAY_BUFFER, iso->vertexBuffer);
      glBufferData(GL_ARRAY_BUFFER, memSize, 0, GL_DYNAMIC_DRAW);
      */
      createVBO(&iso->vertexBuffer, memSize);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&iso->vertexBufferResource,
                                                  iso->vertexBuffer,
                                                  cudaGraphicsMapFlagsWriteDiscard));
                                                  
      //CUDA_SAFE_CALL(cudaGLRegisterBufferObject(iso->vertexBuffer));
      createVBO(&iso->normalBuffer, memSize);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&iso->normalBufferResource,
                                                  iso->normalBuffer,
                                                  cudaGraphicsMapFlagsWriteDiscard));
      //CUDA_SAFE_CALL(cudaGLRegisterBufferObject(iso->normalBuffer));
      createVBO(&iso->texcoordBuffer, memSize);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&iso->texcoordBufferResource,
                                                  iso->texcoordBuffer,
                                                  cudaGraphicsMapFlagsWriteDiscard));
      //CUDA_SAFE_CALL(cudaGLRegisterBufferObject(iso->texcoordBuffer));

#ifdef LIC
      createVBO(&iso->licResultBuffer, memSize);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&iso->licResultBufferResource,
                                                  iso->licResultBuffer,
                                                  cudaGraphicsMapFlagsNone));
      createVBO(&iso->licTexcoordBuffer, memSize);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&iso->licTexcoordBufferResource,
                                                  iso->licTexcoordBuffer,
                                                  cudaGraphicsMapFlagsNone));
#endif

      CheckErr("cudaEngine cudaGLRegisterBufferObject");
   }
#else
   uint memSize = (uint) (iso->activeVerts * 3 * sizeof(float));
   printf("memSize: %d\n", memSize);
   if (iso->bufferSize < memSize) {
      if (iso->bufferSize)
         CUDA_SAFE_CALL(cudaFree(iso->d_vertexBuffer));

      iso->bufferSize = memSize;
      
      CUDA_SAFE_CALL(cudaMalloc(&iso->d_vertexBuffer, memSize));
      CheckErr("cudaEngine malloc vertexBuffer");
   }
   if (vertices && (memSize && ! *vertices)) {
      *vertices = (float *) malloc(memSize);
      *normals = (float *) malloc(memSize);
   } 

#endif // USE_VBO

   //printf("iso->activeElements: %d\n", iso->activeElements);
   if (iso->activeElements > 0)
   {
      dim3 block(512);
      dim3 grid((iso->usg->getNumElem() + block.x - 1) / block.x / 1, 1);
      /*
      dim3 block(512);
      dim3 grid((numElem + block.x - 1) / block.x, 1);
      */
      compactVoxels<<<grid, block, 0, iso->stream>>>(iso->d_compactedArray, iso->d_elemClassification, iso->d_scan, numElem);

      CUDA_SAFE_CALL(cudaThreadSynchronize());
      CheckErr("cudaEngine compactVoxels");

#ifdef USE_VBO
      //map VBOs to cuda arrays
      size_t s;
      CUDA_SAFE_CALL(cudaGraphicsMapResources(1,
                                              &iso->vertexBufferResource, 0));
      CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
                                              (void**) &d_vertexBuffer,
                                              &s,
                                              iso->vertexBufferResource));

      CUDA_SAFE_CALL(cudaGraphicsMapResources(1,
                                              &iso->normalBufferResource, 0));
      CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
                                              (void**) &d_normalBuffer,
                                              &s,
                                              iso->normalBufferResource));

      CUDA_SAFE_CALL(cudaGraphicsMapResources(1,
                                              &iso->texcoordBufferResource, 0));
      CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
                                              (void**) &d_texcoordBuffer,
                                              &s,
                                              iso->texcoordBufferResource));

#ifdef LIC
      CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &iso->licResultBufferResource, 0));
      CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**) &d_licResultBuffer, &s, iso->licResultBufferResource));

      CUDA_SAFE_CALL(cudaGraphicsMapResources(1,
                                              &iso->licTexcoordBufferResource, 0));
      CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer(
                                              (void**) &d_licTexcoordBuffer,
                                              &s,
                                              iso->licTexcoordBufferResource));
#endif

      blockN = iDivUp(iso->activeElements, state->THREAD_N_FAT);
      
#ifdef CUDA_ISO
      float *mapping = NULL;
      if (iso->mapping)
         mapping = iso->mapping->getData();
      printf("activeVerts %d, activeElements %d\n", iso->activeVerts, iso->activeElements);
      printf("generateIsoTriangles(%d, %d)\n", blockN, state->THREAD_N_FAT);
      generateIsoTriangles<<<blockN, state->THREAD_N_FAT, 0, iso->stream>>>(
         d_vertexBuffer, d_normalBuffer, d_texcoordBuffer,
         iso->usg->getVertices(), iso->usg->getNumCoord(),
         iso->d_compactedArray, iso->d_vertsScan, iso->activeVerts,
         iso->usg->getTypeList(), iso->usg->getElemList(),
         iso->usg->getConnList(), iso->activeElements, 
         iso->data->getData(), mapping, iso->texMin, iso->texMax, isoValue);
#else
      float *mapping = NULL;
      if (iso->mapping)
         mapping = iso->mapping->getData();
#ifdef LIC
      float3 *licMapping = NULL;
      if (iso->licMapping)
         licMapping = (float3 *) iso->licMapping->getData();
#endif
      CUDA_SAFE_CALL(cudaMemset((void*) d_vertexBuffer, 0, iso->activeVerts * 4 * sizeof(float)));

#ifdef LIC
      generateCuttingTriangles<<<blockN, state->THREAD_N, 0, iso->stream>>>(
         d_vertexBuffer, d_normalBuffer, d_texcoordBuffer, d_licTexcoordBuffer,
         iso->usg->getVertices(), iso->usg->getNumCoord(),
         iso->d_compactedArray, iso->d_vertsScan, iso->activeVerts,
         iso->usg->getTypeList(), iso->usg->getElemList(),
         iso->usg->getConnList(), iso->activeElements, iso->data->getData(),
         mapping, licMapping, d_licResultBuffer, iso->texMin, iso->texMax, nx, ny, nz, dist);
#else
      generateCuttingTriangles<<<blockN, state->THREAD_N, 0, iso->stream>>>(
         d_vertexBuffer, d_normalBuffer, d_texcoordBuffer, 0,
         iso->usg->getVertices(), iso->usg->getNumCoord(),
         iso->d_compactedArray, iso->d_vertsScan, iso->activeVerts,
         iso->usg->getTypeList(), iso->usg->getElemList(),
         iso->usg->getConnList(), iso->activeElements, iso->data->getData(),
         mapping, 0, 0, iso->texMin, iso->texMax, nx, ny, nz, dist);
#endif

#endif
      CheckErr("cudaEngine generateTriangles");
      CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &iso->normalBufferResource, 0));
      CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &iso->vertexBufferResource, 0));
      CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &iso->texcoordBufferResource, 0));
#ifdef LIC
      CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &iso->licResultBufferResource, 0));

      CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &iso->licTexcoordBufferResource, 0));
#endif
#else
      block = dim3(128);
      grid = dim3((iso->activeVerts + block.x - 1) / block.x, 1);
#ifdef CUDA_ISO
      generateIsoTrianglesC<<<grid, block, 0, iso->stream>>>(
         iso->activeVerts, iso->d_vertexBuffer,
         iso->d_normalBuffer, iso->d_vertices, iso->numCoord,
         iso->d_compactedArray, iso->d_vertsScan, iso->activeVerts,
         iso->d_typeList, iso->d_elemList, iso->d_connList, iso->activeElements,
         iso->d_values, isoValue);
#else
      generateCuttingTrianglesC<<<grid, block, 0, iso->stream>>>(
         iso->activeVerts, iso->d_vertexBuffer,
         iso->d_normalBuffer, iso->d_vertices, iso->numCoord,
         iso->d_compactedArray, iso->d_vertsScan, iso->activeVerts,
         iso->d_typeList, iso->d_elemList, iso->d_connList, iso->activeElements,
         iso->d_values, nx, ny, nz, dist);
#endif
      CheckErr("cudaEngine generateTrianglesC");
#endif // USE_VBO

      offset += iso->activeVerts;
   }
   /*
#ifndef USE_VBO
   if (iso->activeElements > 0 && *vertices) {
      CUDA_SAFE_CALL(cudaMemcpy((void*) *vertices, (void*) iso->d_vertexBuffer, iso->activeVerts * 3 * sizeof(float), cudaMemcpyDeviceToHost));
      CheckErr("cudaEngine memcpy vertexBuffer result");
   }
#endif
   */

#ifdef READBACK
   if (iso->activeElements > 0 && vertices) {

      *vertices = (float *) malloc(iso->activeVerts * 3 * sizeof(float));
      *normals = (float *) malloc(iso->activeVerts * 3 * sizeof(float));

      CUDA_SAFE_CALL(cudaMemcpy((void*) *vertices, (void*) d_vertexBuffer, iso->activeVerts * 3 * sizeof(float), cudaMemcpyDeviceToHost));
      CheckErr("cudaEngine memcpy vertexBuffer result");
   }
#endif

   //cudaPrintfDisplay(stdout, true);
   
   CUDA_SAFE_CALL(cudaThreadSynchronize());

#ifdef STREAMS
   //printf("%d (success: %d)\n", cudaStreamQuery(iso->cstream), cudaSuccess);
#endif

}

/*
#ifdef CUDA_ISO
void computeIsoMeshCUDAs(CUDAState* state, State *iso, float isoValue,
                         int* typeList, int* elemList, int* connList, 
                         float* x, float* y, float* z,
                         int numElem, int numConn, int numCoord,
                         float* data, float *map_x, float *map_y, float *map_z,
                         int* numVertices, float** vertices, float** normals)
{
   GPUUsg *usg = CUDAResourceManager::getInstance()->addUSG(gridName, numElem, numConn, numCoord, typeList, elemList, connList, x, y, z, true);
   
}

#endif
*/
