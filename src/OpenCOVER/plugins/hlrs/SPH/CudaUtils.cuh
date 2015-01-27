#ifndef __CudaUtils_cuh__
#define __CudaUtils_cuh__

typedef unsigned int uint;

uint iDivUp(uint a, uint b);

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads);

#endif