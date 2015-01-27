
#ifndef CUDA_ISO_COMMON_CUH
#define CUDA_ISO_COMMON_CUH

// compute interpolated vertex along an edge
static __device__ float3 vertexInterp(float isolevel, float3 p0, float3 p1, float f0, float f1)
{
   float diff = (f1 - f0);
   if (fabs(isolevel - f0) < EPSILON) {
      return(p0);
   }
   if (fabs(isolevel - f1) < EPSILON) {
      return(p1);
   }
   if (fabs(diff) < EPSILON) {
      return(p0);
   }

   float t = (isolevel - f0) / diff;
   return lerp(p0, p1, t);
} 

// compute interpolated vertex along an edge
static __device__ float3 interp(float isolevel, float3 p0, float3 p1, float f0, float f1, float v0, float v1, float *v)
{
   float diff = (f1 - f0);
   if (fabs(diff) < EPSILON) {
      *v = v0;
      return p0;
   }
      
   if (fabs(isolevel - f0) < EPSILON) {
      *v = v0;
      return(p0);
   }
   if (fabs(isolevel - f1) < EPSILON) {
      *v = v1;
      return(p1);
   }
   if (fabs(diff) < EPSILON) {
      *v = v0;
      return(p0);
   }

   float t = (isolevel - f0) / diff;

   *v = v0 + (t * (v1 - v0));
   return lerp(p0, p1, t);
} 

// compute interpolated vertex along an edge
static __device__ float3 interp(float isolevel, float3 p0, float3 p1, float f0, float f1, float v0, float v1, float *v, float3 l0, float3 l1, float3 *l)
{
   float diff = (f1 - f0);
   if (fabs(diff) < EPSILON) {
      *v = v0;
      *l = make_float3(l0.x, l0.y, l0.z);
      return p0;
   }
      
   if (fabs(isolevel - f0) < EPSILON) {
      *v = v0;
      *l = make_float3(l0.x, l0.y, l0.z);
      return p0;
   }
   if (fabs(isolevel - f1) < EPSILON) {
      *v = v1;
      *l = make_float3(l1.x, l1.y, l1.z);
      return p1;
   }
   if (fabs(diff) < EPSILON) {
      *v = v0;
      *l = make_float3(l0.x, l0.y, l0.z);
      return p0;
   }

   float t = (isolevel - f0) / diff;

   *v = v0 + (t * (v1 - v0));
   float3 a = l0 + (t * (l1 - l0));
   *l = make_float3(a.x, a.y, a.z);
   return lerp(p0, p1, t);
} 


// compact voxel array
__global__ void compactVoxels(uint *compactedVoxelArray, uint *voxelOccupied, uint *voxelOccupiedScan, uint numVoxels)
{
   uint i =
      (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;
   /*
    uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
    uint i = __mul24(blockId, blockDim.x) + threadIdx.x;
   */
    if (i < numVoxels)
       if (voxelOccupied[i])
          compactedVoxelArray[ voxelOccupiedScan[i] ] = i;
    __syncthreads();
}

#endif //CUDA_ISO_COMMON_CUH
