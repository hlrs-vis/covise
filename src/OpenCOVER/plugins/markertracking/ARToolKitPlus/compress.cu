
#ifndef _COMPRESS_CU_
#define _COMPRESS_CU_


//! DISTANCE //
__device__ void distance(const float3 color1, const float3 color2, uint & distance) {
  // calculate euclidean distance of two colors in RGB-SPACE
    float3 diff = color1-color2;
    distance = (uint)dot(diff, diff);
}

//! SORT //
__device__ void sort(const float * values, int * ranks) {
  // sort values and save ranks
    const int tid = threadIdx.x;
    const int tidy = threadIdx.y;
    const int blockOffset = tidy*16;
    int rank = 0;

    #pragma unroll
    for (int i = 0; i < 16; i++){
        rank += (values[blockOffset + i] < values[blockOffset + tid]);
    }

    ranks[blockOffset + tid] = rank;
      //__syncthreads(); nicht noetig, da immer alle 16 gleichzeitig

    // same index
    #pragma unroll
    for (int i = 0; i < 15; i++){
        if (tid > i && ranks[blockOffset + tid] == ranks[blockOffset + i]) ++ranks[blockOffset + tid];
    }

}

#endif // #ifndef _COMPRESS_CU_
