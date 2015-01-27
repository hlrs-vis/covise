/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUDA_ENGINE_H
#define CUDA_ENGINE_H

#include "cudaState.h"
// forward declaration
//extern "C" {
extern void computeIsoMeshCUDA(CUDAState *state, State *iso, float isoValue,
                               int *numVertices, float **vertices, float **normals,
                               void *streamBuffer);

extern void computeCuttingMeshCUDA(CUDAState *state, State *iso, float *rot,
                                   float nx, float ny, float nz, float dist,
                                   int *numVertices, float **vertices, float **normals,
                                   void *streamBuffer);

extern void CleanupState(State *state);

extern void updateState(State *state, int *typeList, int *elemList,
                        int *connList, float *x, float *y, float *z,
                        int numElem, int numConn, int numCoord, float *data);

extern State *InitStateCUDA(CUDAState *state, const char *gridName,
                            const char *valueName, const char *mapName,
                            const int *typeList, const int *elemList, const int *connList,
                            const float *x, const float *y, const float *z,
                            int numElem, int numConn, int numCoord,
                            const float *values, const float *map_x, const float *map_y,
                            const float *map_z, float min, float max,
                            int numElemM, int numConnM, int numCoordM);

extern void InitCUDA(CUDAState *state, int device);
extern void CleanupCUDA(CUDAState *state);
//}
void RenderIsoCUDA(CUDAState *state);

class CUDAEngine
{
    //state, passed to all CUDA calls (CUDA does not allow C++)
private:
    CUDAState *state;

    // do we want to keep the generated geometry in a VBO?
    // this avoids to copy data back and forth from the device
    bool vboGeometry;

public:
    void Init(int device)
    {
        state = new CUDAState();
        InitCUDA(state, device);
    }

    void Cleanup()
    {
        if (vboGeometry)
            CleanupCUDA(state);
    }

    CUDAEngine()
        : state(NULL)
        , vboGeometry(false)
    {
    }
    ~CUDAEngine()
    {
        if (state) // init was called
            Cleanup();
    }

    void RenderIsosurface()
    {
        RenderIsoCUDA(state);
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Run the Cuda part of the computation
    ////////////////////////////////////////////////////////////////////////////////

    void computeIsoMesh(State *iso, float isoValue,
                        int *numVertices, float **vertices, float **normals,
                        void *streamBuffer = NULL)
    {
        computeIsoMeshCUDA(state, iso, isoValue,
                           numVertices, vertices, normals, streamBuffer);
    }
    /*
   void computeIsoMeshs(State *iso, float isoValue,
                       int* numVertices, float** vertices, float** normals)
   {    
      computeIsoMeshCUDAs(state, iso, isoValue,
                          numVertices, vertices, normals);
   }
*/
    void computeCuttingMesh(State *iso, float *rot, float nx, float ny, float nz, float dist,
                            int *numVertices, float **vertices, float **normals,
                            float *streamBuffer = NULL)
    {
        computeCuttingMeshCUDA(state, iso, rot, nx, ny, nz, dist,
                               numVertices, vertices, normals, streamBuffer);
    }

    State *InitState(const char *gridName,
                     const char *valueName, const char *mapName,
                     const int *typeList, const int *elemList, const int *connList,
                     const float *x, const float *y, const float *z,
                     int numElem, int numConn, int numCoord,
                     const float *values, const float *map_x, const float *map_y, const float *map_z, float min, float max, int numElemM = 0, int numConnM = 0, int numVertM = 0)
    {
        return InitStateCUDA(state, gridName, valueName, mapName, typeList,
                             elemList, connList, x, y, z, numElem, numConn,
                             numCoord, values, map_x, map_y, map_z, min, max, numElemM, numConnM, numVertM);
    }
};

#endif //CUDA_ENGINE_H
