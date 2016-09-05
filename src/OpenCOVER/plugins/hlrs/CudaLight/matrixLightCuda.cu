#include <gpu/helper_cuda.h>
#include <math.h>

#include <GL/glew.h>

// CUDA standard includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 16
#define GRID_SIZE 1
#define NUMLIGHTS 2

/**
* CUDA Kernel Device code
**/

__global__ void
    summUpTextures(unsigned char sourceLights[NUMLIGHTS][BLOCK_SIZEY][BLOCK_SIZEX], unsigned char destinationLight[BLOCK_SIZEY][BLOCK_SIZEX])
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= BLOCK_SIZEY || col >= BLOCK_SIZEX)return;
    int s = 0;
    for(int n=0;n<NUMLIGHTS;n++)
    {
        s += sourceLights[n][row][col]; 
    }
    destinationLight[row][col] = s;

}
