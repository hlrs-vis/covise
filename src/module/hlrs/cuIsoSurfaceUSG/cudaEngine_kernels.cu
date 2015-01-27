#ifndef _CUBES_KERNEL_H_
#define _CUBES_KERNEL_H_

//#include "cuPrintf.cu"
#include "cudaCommon.h"

// calculate triangle normal
__device__ float3 calcNormal(float3 *v0, float3 *v1, float3 *v2)
{
    float3 edge0 = *v1 - *v0;
    float3 edge1 = *v2 - *v0;
    // note - it's faster to perform normalization in vertex shader rather than here
    return cross(edge0, edge1);
}

// classify elements based on A) their type B) the number of vertices they will generate
// one thread per element
// output:  -  elemClassification: array 0/1 of elements of that type that are occupied.
//             lenght: numElems * ELEM_COUNT (or the number of element we want to require)  
//             pre: elemClassification is zeroed out
//          -  elemVerts: array containing the number of vertexes required at that element / voxel.
//
// one thread per element
__global__ void classifyIsoElements(uint* elemVerts, uint* elemClassification, //OUT
                                 int* typeList, int* elemList, int* connList, uint numElems, //geometry IN
                                 float* values, float isoValue) //scalar field IN
{
   //uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   uint elemIdx = 
      (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;

   uint elemType = typeList[elemIdx];


   if (elemIdx < numElems) 
   {
      //classify 
      if (elemType == TYPE_HEXAEDER) //hexa, do marching cubes
      {
         // read field values
         float field[8];
         int vertice_indexes[8];

         int p = elemList[elemIdx];

         vertice_indexes[0] = connList[p + 5];
         vertice_indexes[1] = connList[p + 6];
         vertice_indexes[2] = connList[p + 2];
         vertice_indexes[3] = connList[p + 1];
         vertice_indexes[4] = connList[p + 4];
         vertice_indexes[5] = connList[p + 7];
         vertice_indexes[6] = connList[p + 3];
         vertice_indexes[7] = connList[p];

         field[0] = values[vertice_indexes[0]];
         field[1] = values[vertice_indexes[1]];
         field[2] = values[vertice_indexes[2]];
         field[3] = values[vertice_indexes[3]];
         field[4] = values[vertice_indexes[4]];
         field[5] = values[vertice_indexes[5]];
         field[6] = values[vertice_indexes[6]];
         field[7] = values[vertice_indexes[7]];

         // calculate flag indicating if each vertex is inside or outside isosurface
         uint tableIndex;
         tableIndex =  uint(field[0] < isoValue); 
         tableIndex += uint(field[1] < isoValue)*2; 
         tableIndex += uint(field[2] < isoValue)*4; 
         tableIndex += uint(field[3] < isoValue)*8; 
         tableIndex += uint(field[4] < isoValue)*16; 
         tableIndex += uint(field[5] < isoValue)*32; 
         tableIndex += uint(field[6] < isoValue)*64; 
         tableIndex += uint(field[7] < isoValue)*128;

         //tableIndex += cubesOffset;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(hexaNumVertsTex, tableIndex);
         elemVerts[elemIdx] = numVerts;
         elemClassification[elemIdx] = (numVerts > 0);
      }
      else if (elemType == TYPE_TETRAHEDER) // tetra, do marching tetraheders
      {
         // read field values
         float field[4];
         int p = elemList[elemIdx];

         field[0] = values[connList[p + 3]];
         field[1] = values[connList[p + 2]];
         field[2] = values[connList[p + 1]];
         field[3] = values[connList[p]];
        
         // calculate flag indicating if each vertex is inside or outside isosurface
         uint tableIndex;
         tableIndex =  uint(field[0] < isoValue); 
         tableIndex += uint(field[1] < isoValue)*2; 
         tableIndex += uint(field[2] < isoValue)*4; 
         tableIndex += uint(field[3] < isoValue)*8; 
         //tableIndex += tetraOffset;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(tetraNumVertsTex, tableIndex);         
         elemVerts[elemIdx] = numVerts;      
         elemClassification[elemIdx] = (numVerts > 0);
      }
      else if (elemType == TYPE_PYRAMID)
      {
         // read field values
         float field[5];
         int p = elemList[elemIdx];

         field[0] = values[connList[p + 3]];
         field[1] = values[connList[p + 2]];
         field[2] = values[connList[p + 0]];
         field[3] = values[connList[p + 1]];
         field[4] = values[connList[p + 4]];
        
         // calculate flag indicating if each vertex is inside or outside isosurface
         uint tableIndex;
         tableIndex =  uint(field[0] < isoValue); 
         tableIndex += uint(field[1] < isoValue)*2; 
         tableIndex += uint(field[2] < isoValue)*4; 
         tableIndex += uint(field[3] < isoValue)*8; 
         tableIndex += uint(field[3] < isoValue)*16; 
         //tableIndex += tetraOffset;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(pyrNumVertsTex, tableIndex);         
         elemVerts[elemIdx] = numVerts;
         elemClassification[elemIdx] = (numVerts > 0);
      }
      else
      {
         elemVerts[elemIdx] = 0;
         elemClassification[elemIdx] = 0;
      }
   }
   __syncthreads();
}

// classify elements based on A) their type B) the number of vertices they will generate
// one thread per element
// output:  -  elemClassification: array 0/1 of elements of that type that are occupied.
//             lenght: numElems * ELEM_COUNT (or the number of element we want to require)  
//             pre: elemClassification is zeroed out
//          -  elemVerts: array containing the number of vertexes required at that element / voxel.
//
// one thread per element
__global__ void classifyCuttingElements(uint* elemVerts, uint* elemClassification, //OUT
        int* typeList, int* elemList, int* connList, uint numElems,
        float *vertices, uint numCoords, //geometry IN
        float* values, float nx, float ny, float nz, float dist) //scalar IN
{
   uint elemIdx = 
      (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;

   //uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   uint elemType = typeList[elemIdx];

   if (elemIdx < numElems) 
   {
      //classify 
      if (elemType == TYPE_HEXAEDER) //hexa, do marching cubes
      {
         // read field values
         float field[8];
         int vertice_indexes[8];

         int p = elemList[elemIdx];

         vertice_indexes[0] = connList[p + 5];
         vertice_indexes[1] = connList[p + 6];
         vertice_indexes[2] = connList[p + 2];
         vertice_indexes[3] = connList[p + 1];
         vertice_indexes[4] = connList[p + 4];
         vertice_indexes[5] = connList[p + 7];
         vertice_indexes[6] = connList[p + 3];
         vertice_indexes[7] = connList[p];

         uint tableIndex;
         for (int index = 0; index < 8; index ++) {
            float x0 = vertices[vertice_indexes[index]];
            float y0 = vertices[vertice_indexes[index] + numCoords];
            float z0 = vertices[vertice_indexes[index] + numCoords * 2];
            field[index] = (nx * x0 + ny * y0 + nz * z0 - dist);
         }

         // calculate flag indicating if each vertex is inside or outside cuttingsurface
         tableIndex =  uint(field[0] < 0.0); 
         tableIndex += uint(field[1] < 0.0)*2; 
         tableIndex += uint(field[2] < 0.0)*4; 
         tableIndex += uint(field[3] < 0.0)*8; 
         tableIndex += uint(field[4] < 0.0)*16; 
         tableIndex += uint(field[5] < 0.0)*32; 
         tableIndex += uint(field[6] < 0.0)*64; 
         tableIndex += uint(field[7] < 0.0)*128;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(hexaNumVertsTex, tableIndex);
         elemVerts[elemIdx] = numVerts;
         elemClassification[elemIdx] = (numVerts > 0);
      }
      else if (elemType == TYPE_TETRAHEDER) // tetra, do marching tetraheders
      {
         // read field values
         float field[4];
         int vertice_indexes[4];

         int p = elemList[elemIdx];
         vertice_indexes[0] = connList[p + 3];
         vertice_indexes[1] = connList[p + 2];
         vertice_indexes[2] = connList[p + 1];
         vertice_indexes[3] = connList[p];
        
         for (int index = 0; index < 4; index ++) {
            float x0 = vertices[vertice_indexes[index]];
            float y0 = vertices[vertice_indexes[index] + numCoords];
            float z0 = vertices[vertice_indexes[index] + numCoords * 2];
            field[index] = (nx * x0 + ny * y0 + nz * z0 - dist);
         }
        
         // calculate flag indicating if each vertex is inside or outside cuttingsurface
         uint tableIndex;
         tableIndex =  uint(field[0] < 0.0); 
         tableIndex += uint(field[1] < 0.0)*2; 
         tableIndex += uint(field[2] < 0.0)*4; 
         tableIndex += uint(field[3] < 0.0)*8; 
         //tableIndex += tetraOffset;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(tetraNumVertsTex, tableIndex);

         elemVerts[elemIdx] = numVerts;
         elemClassification[elemIdx] = (numVerts > 0);
      }
      else if (elemType == TYPE_PYRAMID)
      {
         // read field values
         float field[5];
         int vertice_indexes[5];

         int p = elemList[elemIdx];

         vertice_indexes[0] = connList[p + 3];
         vertice_indexes[1] = connList[p + 2];
         vertice_indexes[2] = connList[p + 0];
         vertice_indexes[3] = connList[p + 1];
         vertice_indexes[4] = connList[p + 4];

         for (int index = 0; index < 5; index ++) {
            float x0 = vertices[vertice_indexes[index]];
            float y0 = vertices[vertice_indexes[index] + numCoords];
            float z0 = vertices[vertice_indexes[index] + numCoords * 2];
            field[index] = (nx * x0 + ny * y0 + nz * z0 - dist);
         }
         // calculate flag indicating if each vertex is inside or outside cuttingsurface
         uint tableIndex;
         tableIndex =  uint(field[0] < 0.0); 
         tableIndex += uint(field[1] < 0.0)*2; 
         tableIndex += uint(field[2] < 0.0)*4; 
         tableIndex += uint(field[3] < 0.0)*8; 
         tableIndex += uint(field[3] < 0.0)*16; 
         //tableIndex += tetraOffset;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(pyrNumVertsTex, tableIndex);

         elemVerts[elemIdx] = numVerts;
         elemClassification[elemIdx] = (numVerts > 0);
      }
   }
   __syncthreads();
}

// classify elements based on A) their type B) the number of vertices they will generate
// one thread per element
// output:  -  elemClassification: array 0/1 of elements of that type that are occupied.
//             lenght: numElems * ELEM_COUNT (or the number of element we want to require)  
//             pre: elemClassification is zeroed out
//          -  elemVerts: array containing the number of vertexes required at that element / voxel.
//
// one thread per element
__global__ void classifyCuttingElementsSphere(uint* elemVerts, uint* elemClassification, //OUT
                                              int* typeList, int* elemList, int* connList, uint numElems,
                                              float *vertices, uint numCoords, //geometry IN
                                              float* values, float nx, float ny, float nz, float radius) //scalar field IN
{
   //uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   uint elemIdx = 
      (((blockIdx.y * gridDim.x) + blockIdx.x) * blockDim.x) + threadIdx.x;

   uint elemType = typeList[elemIdx];

   if (elemIdx < numElems) 
   {
      //classify 
      if (elemType == TYPE_HEXAEDER) //hexa, do marching cubes
      {
         // read field values
         float field[8];
         int vertice_indexes[8];

         int p = elemList[elemIdx];

         vertice_indexes[0] = connList[p + 5];
         vertice_indexes[1] = connList[p + 6];
         vertice_indexes[2] = connList[p + 2];
         vertice_indexes[3] = connList[p + 1];
         vertice_indexes[4] = connList[p + 4];
         vertice_indexes[5] = connList[p + 7];
         vertice_indexes[6] = connList[p + 3];
         vertice_indexes[7] = connList[p];

         uint tableIndex;
         for (int index = 0; index < 8; index ++) {
            float x0 = vertices[vertice_indexes[index]];
            float y0 = vertices[vertice_indexes[index] + numCoords];
            float z0 = vertices[vertice_indexes[index] + numCoords * 2];
            field[index] = (nx - x0) * (nx - x0) + (ny - y0) * (ny - y0) +
               (nz - z0) * (nz -z0) - radius * radius;
         }

         // calculate flag indicating if each vertex is inside or outside cuttingsurface
         tableIndex =  uint(field[0] < 0.0); 
         tableIndex += uint(field[1] < 0.0)*2; 
         tableIndex += uint(field[2] < 0.0)*4; 
         tableIndex += uint(field[3] < 0.0)*8; 
         tableIndex += uint(field[4] < 0.0)*16; 
         tableIndex += uint(field[5] < 0.0)*32; 
         tableIndex += uint(field[6] < 0.0)*64; 
         tableIndex += uint(field[7] < 0.0)*128;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(hexaNumVertsTex, tableIndex);
         elemVerts[elemIdx] = numVerts;
         elemClassification[elemIdx] = (numVerts > 0);
      }
      else if (elemType == TYPE_TETRAHEDER) // tetra, do marching tetraheders
      {
         // read field values
         float field[4];
         int vertice_indexes[4];

         int p = elemList[elemIdx];
         vertice_indexes[0] = connList[p + 3];
         vertice_indexes[1] = connList[p + 2];
         vertice_indexes[2] = connList[p + 1];
         vertice_indexes[3] = connList[p];
        
         for (int index = 0; index < 4; index ++) {
            float x0 = vertices[vertice_indexes[index]];
            float y0 = vertices[vertice_indexes[index] + numCoords];
            float z0 = vertices[vertice_indexes[index] + numCoords * 2];
            field[index] = (nx - x0) * (nx - x0) + (ny - y0) * (ny - y0) +
               (nz - z0) * (nz -z0) - radius * radius;
         }

         // calculate flag indicating if each vertex is inside or outside cuttingsurface
         uint tableIndex;
         tableIndex =  uint(field[0] < 0.0); 
         tableIndex += uint(field[1] < 0.0)*2; 
         tableIndex += uint(field[2] < 0.0)*4; 
         tableIndex += uint(field[3] < 0.0)*8; 
         //tableIndex += tetraOffset;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(tetraNumVertsTex, tableIndex);
         elemVerts[elemIdx] = numVerts;
         elemClassification[elemIdx] = (numVerts > 0);
      }
      else if (elemType == TYPE_PYRAMID)
      {
         // read field values
         float field[5];
         int vertice_indexes[5];

         int p = elemList[elemIdx];

         vertice_indexes[0] = connList[p + 3];
         vertice_indexes[1] = connList[p + 2];
         vertice_indexes[2] = connList[p + 0];
         vertice_indexes[3] = connList[p + 1];
         vertice_indexes[4] = connList[p + 4];

         for (int index = 0; index < 5; index ++) {
            float x0 = vertices[vertice_indexes[index]];
            float y0 = vertices[vertice_indexes[index] + numCoords];
            float z0 = vertices[vertice_indexes[index] + numCoords * 2];
            field[index] = (nx - x0) * (nx - x0) + (ny - y0) * (ny - y0) +
               (nz - z0) * (nz -z0) - radius * radius;
         }
         // calculate flag indicating if each vertex is inside or outside cuttingsurface
         uint tableIndex;
         tableIndex =  uint(field[0] < 0.0); 
         tableIndex += uint(field[1] < 0.0)*2; 
         tableIndex += uint(field[2] < 0.0)*4; 
         tableIndex += uint(field[3] < 0.0)*8; 
         tableIndex += uint(field[3] < 0.0)*16; 
         //tableIndex += tetraOffset;

         // read number of vertices from texture
         uint numVerts = tex1Dfetch(pyrNumVertsTex, tableIndex);
         
         elemVerts[elemIdx] = numVerts;
         elemClassification[elemIdx] = (numVerts > 0);
      }
   }
   __syncthreads();
}

#ifndef COVISE
__device__ void generateIsoTrianglesPyr(float4 *pos, float4 *norm, float *tex,
                                     float* vertexArrayIn, int numCoords,
                                     uint* compactedArray,
                                     uint* vertsScan, uint maxVerts,
                                     int* typeList, int* elemList,
                                     int* connList, uint activeElems,
                                     float* values, float *map, float texMin,
                                     float texMax, float isoValue)
#else
__device__ void generateIsoTrianglesPyrC(int totalVertices, float *vertices,
				      float *normals, float* vertexArrayIn,
				      int numCoords, uint* compactedArray,
				      uint* vertsScan, uint maxVerts,
				      int* typeList, int* elemList,
				      int* connList, uint activeElems,
				      float* values, float isoValue)
#endif
{
   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elemIdx < activeElems)  
    {

    int idx = compactedArray[elemIdx];
    int i = elemList[idx];
    // our points are given directly in the unstructured grid
    // the connList array at position i has the positions inside x, y, z

    // calculate cell vertex positions
    //int xi = connList[i];
    //int yi = connList[i] + numCoords;
    //int zi = connList[i] + (numCoords << 1);
    
    float3 v[5];
    float field[5];
    int xi, yi, zi;
#ifndef COVISE
    float mapping[5];
#endif
    xi = connList[i + 3];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[0] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[0] = values[xi];
#ifndef COVISE
    if (map)
       mapping[0] = map[xi];
#endif

    xi = connList[i + 2];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[1] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[1] = values[xi];
#ifndef COVISE
    if (map)
       mapping[1] = map[xi];
#endif

    xi = connList[i + 0];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[2] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[2] = values[xi];
#ifndef COVISE
    if (map)
       mapping[2] = map[xi];
#endif

    xi = connList[i + 1];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[3] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[3] = values[xi];
#ifndef COVISE
    if (map)
       mapping[3] = map[xi];
#endif

    xi = connList[i + 4];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[4] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[4] = values[xi];
#ifndef COVISE
    if (map)
       mapping[4] = map[xi];
#endif

    // recalculate flag
    // (this is faster than storing it in global memory)    
    uint tableIndex;
    tableIndex =  uint(field[0] < isoValue); 
    tableIndex += uint(field[1] < isoValue)*2; 
    tableIndex += uint(field[2] < isoValue)*4; 
    tableIndex += uint(field[3] < isoValue)*8; 
    tableIndex += uint(field[4] < isoValue)*16; 

    // find the vertices where the surface intersects it
    float3 vertlist[8]; //one per edge

#ifndef COVISE
    float maplist[8];
    vertlist[0] = interp(isoValue, v[0], v[1], field[0], field[1], mapping[0], mapping[1], &maplist[0]);
    vertlist[1] = interp(isoValue, v[0], v[2], field[0], field[2], mapping[0], mapping[2], &maplist[1]);
    vertlist[2] = interp(isoValue, v[1], v[3], field[1], field[3], mapping[1], mapping[3], &maplist[2]);
    vertlist[3] = interp(isoValue, v[2], v[3], field[2], field[3], mapping[2], mapping[3], &maplist[3]);
    vertlist[4] = interp(isoValue, v[0], v[4], field[0], field[4], mapping[0], mapping[4], &maplist[4]);
    vertlist[5] = interp(isoValue, v[1], v[4], field[1], field[4], mapping[1], mapping[4], &maplist[5]);
    vertlist[6] = interp(isoValue, v[2], v[4], field[2], field[4], mapping[2], mapping[4], &maplist[6]);
    vertlist[7] = interp(isoValue, v[3], v[4], field[3], field[4], mapping[3], mapping[4], &maplist[7]);
#else
    vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
    vertlist[1] = vertexInterp(isoValue, v[0], v[2], field[0], field[2]);
    vertlist[2] = vertexInterp(isoValue, v[1], v[3], field[1], field[3]);
    vertlist[3] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
    vertlist[4] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
    vertlist[5] = vertexInterp(isoValue, v[1], v[4], field[1], field[4]);
    vertlist[6] = vertexInterp(isoValue, v[2], v[4], field[2], field[4]);
    vertlist[7] = vertexInterp(isoValue, v[3], v[4], field[3], field[4]);
#endif
    
   // output triangle vertices
   uint numVerts = tex1Dfetch(pyrNumVertsTex, tableIndex);
   for(int i=0; i<numVerts; i+=3) 
   {
      uint index = vertsScan[idx] + i;

      float3* v[3];
      uint edge[3];

      edge[0] = tex1Dfetch(pyrTriTex, (tableIndex*12) + i);
      v[0] = &vertlist[edge[0]];

      edge[1] = tex1Dfetch(pyrTriTex, (tableIndex*12) + i + 1);
      v[1] = &vertlist[edge[1]];

      edge[2] = tex1Dfetch(pyrTriTex, (tableIndex*12) + i + 2);
      v[2] = &vertlist[edge[2]];

      // calculate triangle surface normal
      float3 n = calcNormal(v[0], v[1], v[2]);

#ifndef COVISE
      if (index < (maxVerts - 2)) 
      {
         pos[index] = make_float4(*v[0], 1.0f);
         norm[index] = make_float4(n, 0.0f);

         pos[index+1] = make_float4(*v[1], 1.0f);
         norm[index+1] = make_float4(n, 0.0f);

         pos[index+2] = make_float4(*v[2], 1.0f);
         norm[index+2] = make_float4(n, 0.0f);

         if (map)
         {
            tex[index] = (maplist[edge[0]] - texMin) / (texMax - texMin);
            tex[index + 1] = (maplist[edge[1]] - texMin) / (texMax - texMin);
            tex[index + 2] = (maplist[edge[2]] - texMin) / (texMax - texMin);
         }
      }
#else
      int yIndex = index + totalVertices;
      int zIndex = index + totalVertices * 2;

      if (index < (maxVerts - 2)) 
      {
	  vertices[index] = v[0]->x;
	  vertices[index + 1] = v[1]->x;
	  vertices[index + 2] = v[2]->x;

	  vertices[yIndex] = v[0]->y;
	  vertices[yIndex + 1] = v[1]->y;
	  vertices[yIndex + 2] = v[2]->y;

	  vertices[zIndex] = v[0]->z;
	  vertices[zIndex + 1] = v[1]->z;
	  vertices[zIndex + 2] = v[2]->z;
      }
#endif
   }
   }
}

#ifndef COVISE
__device__ void generateIsoTrianglesTetra(float4 *pos, float4 *norm, float *tex,
                                       float* vertexArrayIn, int numCoords, 
                                       uint* compactedThetraArray,
				       uint* vertsScan, uint maxVerts,
                                       int* typeList, int* elemList,
                                       int* connList, uint activeThetra,
				       float* values, float *map, float texMin,
                                       float texMax, float isoValue)
#else
__device__ void generateIsoTrianglesTetraC(int totalVertices, float *vertices,
					float *normals, float* vertexArrayIn,
					int numCoords,
					uint* compactedThetraArray,
					uint* vertsScan, uint maxVerts,
					int* typeList, int* elemList,
					int* connList, uint activeThetra,
					float* values, float isoValue)
#endif
{
   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elemIdx < activeThetra)  
    {

    int idx = compactedThetraArray[elemIdx];
    int i = elemList[idx];
    // our points are given directly in the unstructured grid
    // the connList array at position i has the positions inside x, y, z

    // calculate cell vertex positions
    //int xi = connList[i];
    //int yi = connList[i] + numCoords;
    //int zi = connList[i] + (numCoords << 1);
    
    float3 v[4];
    float field[4];
#ifndef COVISE
    float mapping[4];
#endif
    int xi, yi, zi;

    xi = connList[i + 3];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[0] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[0] = values[xi];
#ifndef COVISE
    if (map)
       mapping[0] = map[xi];
#endif

    xi = connList[i + 2];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[1] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[1] = values[xi];
#ifndef COVISE
    if (map)
       mapping[1] = map[xi];
#endif

    xi = connList[i + 1];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[2] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[2] = values[xi];
#ifndef COVISE
    if (map)
       mapping[2] = map[xi];
#endif

    xi = connList[i + 0];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[3] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[3] = values[xi];
#ifndef COVISE
    if (map)
       mapping[3] = map[xi];
#endif

    // recalculate flag
    // (this is faster than storing it in global memory)    
    uint tableIndex;
    tableIndex =  uint(field[0] < isoValue); 
    tableIndex += uint(field[1] < isoValue)*2; 
    tableIndex += uint(field[2] < isoValue)*4; 
    tableIndex += uint(field[3] < isoValue)*8; 

    // find the vertices where the surface intersects it

    float3 vertlist[6]; //one per edge

    float maplist[6];
    
    vertlist[0] = interp(isoValue, v[0], v[1], field[0], field[1], mapping[0], mapping[1], &maplist[0]);
    vertlist[1] = interp(isoValue, v[0], v[2], field[0], field[2], mapping[0], mapping[2], &maplist[1]);
    vertlist[2] = interp(isoValue, v[0], v[3], field[0], field[3], mapping[0], mapping[3], &maplist[2]);
    vertlist[3] = interp(isoValue, v[1], v[2], field[1], field[2], mapping[1], mapping[2], &maplist[3]);
    vertlist[4] = interp(isoValue, v[1], v[3], field[1], field[3], mapping[1], mapping[3], &maplist[4]);
    vertlist[5] = interp(isoValue, v[2], v[3], field[2], field[3], mapping[2], mapping[3], &maplist[5]);

    // output triangle vertices
    uint numVerts = tex1Dfetch(tetraNumVertsTex, tableIndex);
    for(int i=0; i<numVerts; i+=3) 
    {
        uint index = vertsScan[idx] + i;

        float3 *v[3];
        uint edge[3];
        edge[0] = tex1Dfetch(tetraTriTex, (tableIndex*6) + i);
        v[0] = &vertlist[edge[0]];

        edge[1] = tex1Dfetch(tetraTriTex, (tableIndex*6) + i + 1);
        v[1] = &vertlist[edge[1]];

        edge[2] = tex1Dfetch(tetraTriTex, (tableIndex*6) + i + 2);
        v[2] = &vertlist[edge[2]];

        // calculate triangle surface normal
        float3 n = calcNormal(v[0], v[1], v[2]);

#ifndef COVISE
        if (index < (maxVerts - 2)) 
        {
            pos[index] = make_float4(*v[0], 1.0f);
            norm[index] = make_float4(n, 0.0f);

            pos[index+1] = make_float4(*v[1], 1.0f);
            norm[index+1] = make_float4(n, 0.0f);

            pos[index+2] = make_float4(*v[2], 1.0f);
            norm[index+2] = make_float4(n, 0.0f);

            if (map) {
               tex[index] = (maplist[edge[0]] - texMin) / (texMax - texMin);
               tex[index + 1] = (maplist[edge[1]] - texMin) / (texMax - texMin);
               tex[index + 2] = (maplist[edge[2]] - texMin) / (texMax - texMin);
            }
        }
#else
	int yIndex = index + totalVertices;
	int zIndex = index + totalVertices * 2;

        if (index < (maxVerts - 2)) 
        {
            vertices[index] = v[0]->x;
            vertices[index + 1] = v[1]->x;
            vertices[index + 2] = v[2]->x;
            
            vertices[yIndex] = v[0]->y;
            vertices[yIndex + 1] = v[1]->y;
            vertices[yIndex + 2] = v[2]->y;
            
            vertices[zIndex] = v[0]->z;
            vertices[zIndex + 1] = v[1]->z;
            vertices[zIndex + 2] = v[2]->z;
        }
#endif
    }
    }    
}


#ifndef COVISE
__device__ void generateIsoTrianglesHexa(float4 *pos, float4 *norm, float *tex,
                                      float* vertexArrayIn, int numCoords,
                                      uint* compactedHexaArray, 
                                      uint* vertsScan, uint maxVerts,
                                      int* typeList, int* elemList,
                                      int* connList, uint activeHexa,
				      float* values, float *map, float texMin,
                                      float texMax, float isoValue)
#else
__device__ void generateIsoTrianglesHexaC(int totalVertices, float *vertices,
				       float *normals, float* vertexArrayIn,
				       int numCoords, uint* compactedHexaArray,
				       uint* vertsScan, uint maxVerts,
				       int* typeList,
				       int* elemList, int* connList,
				       uint activeHexa,
				       float* values, float isoValue)
#endif
{
    uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elemIdx < activeHexa)
    {

    int idx = compactedHexaArray[elemIdx];
    int i = elemList[idx];

    // our points are given directly in the unstructured grid
    // the connList array at position i has the positions inside x, y, z

    // calculate cell vertex positions
    //int xi = connList[i];
    //int yi = connList[i] + numCoords;
    //int zi = connList[i] + (numCoords << 1);
    float3 v[8];
    float field[8];
#ifndef COVISE
    float mapping[8];
#endif
    int xi, yi, zi;

    xi = connList[i + 5];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[0] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[0] = values[xi];
#ifndef COVISE
    if (map)
       mapping[0] = map[xi];
#endif

    xi = connList[i + 6];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[1] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[1] = values[xi];
#ifndef COVISE
    if (map)
       mapping[1] = map[xi];
#endif

    xi = connList[i + 2];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[2] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[2] = values[xi];
#ifndef COVISE
    if (map)
       mapping[2] = map[xi];
#endif

    xi = connList[i + 1];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[3] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[3] = values[xi];
#ifndef COVISE
    if (map)
       mapping[3] = map[xi];
#endif

    xi = connList[i + 4];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[4] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[4] = values[xi];
#ifndef COVISE
    if (map)
       mapping[4] = map[xi];
#endif

    xi = connList[i + 7];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[5] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[5] = values[xi];
#ifndef COVISE
    if (map)
       mapping[5] = map[xi];
#endif

    xi = connList[i + 3];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[6] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[6] = values[xi];
#ifndef COVISE
    if (map)
       mapping[6] = map[xi];
#endif

    xi = connList[i];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[7] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[7] = values[xi];
#ifndef COVISE
    if (map)
       mapping[7] = map[xi];
#endif

    // recalculate flag
    // (this is faster than storing it in global memory)
    uint tableIndex;
    tableIndex =  uint(field[0] < isoValue); 
    tableIndex += uint(field[1] < isoValue)*2; 
    tableIndex += uint(field[2] < isoValue)*4; 
    tableIndex += uint(field[3] < isoValue)*8; 
    tableIndex += uint(field[4] < isoValue)*16; 
    tableIndex += uint(field[5] < isoValue)*32; 
    tableIndex += uint(field[6] < isoValue)*64; 
    tableIndex += uint(field[7] < isoValue)*128;

    // find the vertices where the surface intersects the cube 

   float3 vertlist[12]; //one per edge
#ifndef COVISE
   float maplist[12];
   vertlist[0] = interp(isoValue, v[0], v[1], field[0], field[1], mapping[0], mapping[1], &maplist[0]);

   vertlist[1] = interp(isoValue, v[1], v[2], field[1], field[2], mapping[1], mapping[2], &maplist[1]);
   vertlist[2] = interp(isoValue, v[2], v[3], field[2], field[3], mapping[2], mapping[3], &maplist[2]);

   vertlist[3] = interp(isoValue, v[3], v[0], field[3], field[0], mapping[3], mapping[0], &maplist[3]);

   vertlist[4] = interp(isoValue, v[4], v[5], field[4], field[5], mapping[4], mapping[5], &maplist[4]);
   vertlist[5] = interp(isoValue, v[5], v[6], field[5], field[6], mapping[5], mapping[6], &maplist[5]);
   vertlist[6] = interp(isoValue, v[6], v[7], field[6], field[7], mapping[6], mapping[7], &maplist[6]);
   vertlist[7] = interp(isoValue, v[7], v[4], field[7], field[4], mapping[7], mapping[4], &maplist[7]);

   vertlist[8] = interp(isoValue, v[0], v[4], field[0], field[4], mapping[0], mapping[4], &maplist[8]);
   vertlist[9] = interp(isoValue, v[1], v[5], field[1], field[5], mapping[1], mapping[5], &maplist[9]);
   vertlist[10] = interp(isoValue, v[2], v[6], field[2], field[6], mapping[2], mapping[6], &maplist[10]);
   vertlist[11] = interp(isoValue, v[3], v[7], field[3], field[7], mapping[3], mapping[7], &maplist[11]);
#else
   vertlist[0] = vertexInterp(isoValue, v[0], v[1], field[0], field[1]);
   vertlist[1] = vertexInterp(isoValue, v[1], v[2], field[1], field[2]);
   vertlist[2] = vertexInterp(isoValue, v[2], v[3], field[2], field[3]);
   vertlist[3] = vertexInterp(isoValue, v[3], v[0], field[3], field[0]);

   vertlist[4] = vertexInterp(isoValue, v[4], v[5], field[4], field[5]);
   vertlist[5] = vertexInterp(isoValue, v[5], v[6], field[5], field[6]);
   vertlist[6] = vertexInterp(isoValue, v[6], v[7], field[6], field[7]);
   vertlist[7] = vertexInterp(isoValue, v[7], v[4], field[7], field[4]);

   vertlist[8] = vertexInterp(isoValue, v[0], v[4], field[0], field[4]);
   vertlist[9] = vertexInterp(isoValue, v[1], v[5], field[1], field[5]);
   vertlist[10] = vertexInterp(isoValue, v[2], v[6], field[2], field[6]);
   vertlist[11] = vertexInterp(isoValue, v[3], v[7], field[3], field[7]);
#endif
    // output triangle vertices

    uint numVerts = tex1Dfetch(hexaNumVertsTex, tableIndex);

    for(int i=0; i<numVerts; i+=3) 
    {
       uint index = vertsScan[idx] + i;
       float3 *v[3];
       uint edge[3];
       edge[0] = tex1Dfetch(hexaTriTex, (tableIndex*16) + i);
       v[0] = &vertlist[edge[0]];
       
       edge[1] = tex1Dfetch(hexaTriTex, (tableIndex*16) + i + 1);
       v[1] = &vertlist[edge[1]];
       
       edge[2] = tex1Dfetch(hexaTriTex, (tableIndex*16) + i + 2);
       v[2] = &vertlist[edge[2]];
       
       // calculate triangle surface normal
       float3 n = calcNormal(v[0], v[1], v[2]);

#ifndef COVISE
       
       if (index < (maxVerts - 2))
       {
           pos[index] = make_float4(*v[0], 1.0f);
           norm[index] = make_float4(n, 0.0f);
           pos[index+1] = make_float4(*v[1], 1.0f);
           norm[index+1] = make_float4(n, 0.0f);
           pos[index+2] = make_float4(*v[2], 1.0f);
           norm[index+2] = make_float4(n, 0.0f);

           if (map) {
              tex[index] = (maplist[edge[0]] - texMin) / (texMax - texMin);
              tex[index + 1] = (maplist[edge[1]] - texMin) / (texMax - texMin);
              tex[index + 2] = (maplist[edge[2]] - texMin) / (texMax - texMin);
           }
       }
#else
	int yIndex = index + totalVertices;
	int zIndex = index + totalVertices * 2;

        if (index < (maxVerts - 2))
        {
	  vertices[index] = v[0]->x;
	  vertices[index + 1] = v[1]->x;
	  vertices[index + 2] = v[2]->x;
	  vertices[yIndex] = v[0]->y;
	  vertices[yIndex + 1] = v[1]->y;
	  vertices[yIndex + 2] = v[2]->y;
	  vertices[zIndex] = v[0]->z;
	  vertices[zIndex + 1] = v[1]->z;
	  vertices[zIndex + 2] = v[2]->z;
        }
#endif
    }
    }
   __syncthreads();
}

#ifndef COVISE
__device__ void generateCuttingTrianglesHexa(float4 *pos, float4 *norm,
                                             float *tex, float2 *licTex,
                                     float* vertexArrayIn, int numCoords,
                                     uint* compactedHexaArray,
                                     uint* vertsScan, uint maxVerts,
                                     int* typeList, int* elemList,
                                     int* connList, uint activeHexa,
                                     float* values, float *map, float3 *licMap,
                                     float3 *licResult,
                                     float texMin, float texMax, float nx,
                                     float ny, float nz, float dist)
#else
__device__ void generateCuttingTrianglesHexaC(int totalVertices, float *vertices,
				       float *normals, float* vertexArrayIn,
				       int numCoords, uint* compactedHexaArray,
				       uint* vertsScan, uint maxVerts,
				       int* typeList,
				       int* elemList, int* connList,
				       uint activeHexa,
                                       float* values, float nx, float ny, float nz, float dist)
#endif
{
    uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
    uint numVerts;

    if (elemIdx < activeHexa)
    {

    int idx = compactedHexaArray[elemIdx];
    int i = elemList[idx];
    // our points are given directly in the unstructured grid
    // the connList array at position i has the positions inside x, y, z

    // calculate cell vertex positions
    //int xi = connList[i];
    //int yi = connList[i] + numCoords;
    //int zi = connList[i] + (numCoords << 1);
    
    float3 v[8];
    float field[8];
#ifndef COVISE
    float mapping[8];
#ifdef LIC
    float3 licMapping[8];
#endif
#endif
    int xi, yi, zi;
    xi = connList[i + 5];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[0] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[0] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);
#ifndef COVISE
    if (map)
       mapping[0] = map[xi];
#ifdef LIC
       licMapping[0] = licMap[xi];
#endif
#endif
    xi = connList[i + 6];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[1] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[1] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);

#ifndef COVISE
    if (map)
       mapping[1] = map[xi];
#ifdef LIC
       licMapping[1] = licMap[xi];
#endif
#endif
    xi = connList[i + 2];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[2] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[2] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);
#ifndef COVISE
    if (map)
       mapping[2] = map[xi];
#ifdef LIC
       licMapping[2] = licMap[xi];
#endif
#endif
    xi = connList[i + 1];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[3] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[3] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);

#ifndef COVISE
    if (map)
       mapping[3] = map[xi];
#ifdef LIC
       licMapping[3] = licMap[xi];
#endif
#endif
    xi = connList[i + 4];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[4] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[4] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);
#ifndef COVISE
    if (map)
       mapping[4] = map[xi];
#ifdef LIC
       licMapping[4] = licMap[xi];
#endif
#endif
    xi = connList[i + 7];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[5] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[5] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);
#ifndef COVISE
    if (map)
       mapping[5] = map[xi];
#ifdef LIC
       licMapping[5] = licMap[xi];
#endif
#endif
    xi = connList[i + 3];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[6] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[6] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);
#ifndef COVISE
    if (map)
       mapping[6] = map[xi];
#ifdef LIC
       licMapping[6] = licMap[xi];
#endif
#endif
    xi = connList[i];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[7] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[7] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);
#ifndef COVISE
    if (map)
       mapping[7] = map[xi];
#ifdef LIC
       licMapping[7] = licMap[xi];
#endif
#endif

    // recalculate flag
    // (this is faster than storing it in global memory)
    uint tableIndex;
    tableIndex =  uint(field[0] < 0.0); 
    tableIndex += uint(field[1] < 0.0)*2; 
    tableIndex += uint(field[2] < 0.0)*4; 
    tableIndex += uint(field[3] < 0.0)*8; 
    tableIndex += uint(field[4] < 0.0)*16; 
    tableIndex += uint(field[5] < 0.0)*32; 
    tableIndex += uint(field[6] < 0.0)*64; 
    tableIndex += uint(field[7] < 0.0)*128;

    // find the vertices where the surface intersects the cube 

   float3 vertlist[12]; //one per edge

#ifndef COVISE
   float maplist[12];
#ifdef LIC
   float3 licMaplist[12];
   vertlist[0] = interp(0.0, v[0], v[1], field[0], field[1], mapping[0], mapping[1], &maplist[0], licMapping[0], licMapping[1], &licMaplist[0]);
   vertlist[1] = interp(0.0, v[1], v[2], field[1], field[2], mapping[1], mapping[2], &maplist[1], licMapping[1], licMapping[2], &licMaplist[1]);
   vertlist[2] = interp(0.0, v[2], v[3], field[2], field[3], mapping[2], mapping[3], &maplist[2], licMapping[2], licMapping[3], &licMaplist[2]);
   vertlist[3] = interp(0.0, v[3], v[0], field[3], field[0], mapping[3], mapping[0], &maplist[3], licMapping[3], licMapping[0], &licMaplist[3]);

   vertlist[4] = interp(0.0, v[4], v[5], field[4], field[5], mapping[4], mapping[5], &maplist[4], licMapping[4], licMapping[5], &licMaplist[4]);
   vertlist[5] = interp(0.0, v[5], v[6], field[5], field[6], mapping[5], mapping[6], &maplist[5], licMapping[5], licMapping[6], &licMaplist[5]);
   vertlist[6] = interp(0.0, v[6], v[7], field[6], field[7], mapping[6], mapping[7], &maplist[6], licMapping[6], licMapping[7], &licMaplist[6]);
   vertlist[7] = interp(0.0, v[7], v[4], field[7], field[4], mapping[7], mapping[4], &maplist[7], licMapping[7], licMapping[4], &licMaplist[7]);

   vertlist[8] = interp(0.0, v[0], v[4], field[0], field[4], mapping[0], mapping[4], &maplist[8], licMapping[0], licMapping[4], &licMaplist[8]);
   vertlist[9] = interp(0.0, v[1], v[5], field[1], field[5], mapping[1], mapping[5], &maplist[9], licMapping[1], licMapping[5], &licMaplist[9]);
   vertlist[10] = interp(0.0, v[2], v[6], field[2], field[6], mapping[2], mapping[6], &maplist[10], licMapping[2], licMapping[6], &licMaplist[10]);
   vertlist[11] = interp(0.0, v[3], v[7], field[3], field[7], mapping[3], mapping[7], &maplist[11], licMapping[3], licMapping[7], &licMaplist[11]);
#else
   vertlist[0] = interp(0.0, v[0], v[1], field[0], field[1], mapping[0], mapping[1], &maplist[0]);
   vertlist[1] = interp(0.0, v[1], v[2], field[1], field[2], mapping[1], mapping[2], &maplist[1]);
   vertlist[2] = interp(0.0, v[2], v[3], field[2], field[3], mapping[2], mapping[3], &maplist[2]);
   vertlist[3] = interp(0.0, v[3], v[0], field[3], field[0], mapping[3], mapping[0], &maplist[3]);

   vertlist[4] = interp(0.0, v[4], v[5], field[4], field[5], mapping[4], mapping[5], &maplist[4]);
   vertlist[5] = interp(0.0, v[5], v[6], field[5], field[6], mapping[5], mapping[6], &maplist[5]);
   vertlist[6] = interp(0.0, v[6], v[7], field[6], field[7], mapping[6], mapping[7], &maplist[6]);
   vertlist[7] = interp(0.0, v[7], v[4], field[7], field[4], mapping[7], mapping[4], &maplist[7]);

   vertlist[8] = interp(0.0, v[0], v[4], field[0], field[4], mapping[0], mapping[4], &maplist[8]);
   vertlist[9] = interp(0.0, v[1], v[5], field[1], field[5], mapping[1], mapping[5], &maplist[9]);
   vertlist[10] = interp(0.0, v[2], v[6], field[2], field[6], mapping[2], mapping[6], &maplist[10]);
   vertlist[11] = interp(0.0, v[3], v[7], field[3], field[7], mapping[3], mapping[7], &maplist[11]);
#endif
#else
   vertlist[0] = vertexInterp(0.0, v[0], v[1], field[0], field[1]);
   vertlist[1] = vertexInterp(0.0, v[1], v[2], field[1], field[2]);
   vertlist[2] = vertexInterp(0.0, v[2], v[3], field[2], field[3]);
   vertlist[3] = vertexInterp(0.0, v[3], v[0], field[3], field[0]);

   vertlist[4] = vertexInterp(0.0, v[4], v[5], field[4], field[5]);
   vertlist[5] = vertexInterp(0.0, v[5], v[6], field[5], field[6]);
   vertlist[6] = vertexInterp(0.0, v[6], v[7], field[6], field[7]);
   vertlist[7] = vertexInterp(0.0, v[7], v[4], field[7], field[4]);

   vertlist[8] = vertexInterp(0.0, v[0], v[4], field[0], field[4]);
   vertlist[9] = vertexInterp(0.0, v[1], v[5], field[1], field[5]);
   vertlist[10] = vertexInterp(0.0, v[2], v[6], field[2], field[6]);
   vertlist[11] = vertexInterp(0.0, v[3], v[7], field[3], field[7]);
#endif

    // output triangle vertices
    numVerts = tex1Dfetch(hexaNumVertsTex, tableIndex);
#ifdef DEBUG
    printf("%d numVerts: %d\n", elemIdx, numVerts);
#endif
    for(int i=0; i<numVerts; i+=3) 
    {
        uint index = vertsScan[idx] + i;

        float3 *v[3];
        uint edge[3];
        edge[0] = tex1Dfetch(hexaTriTex, (tableIndex*16) + i);
        v[0] = &vertlist[edge[0]];

        edge[1] = tex1Dfetch(hexaTriTex, (tableIndex*16) + i + 1);
        v[1] = &vertlist[edge[1]];

        edge[2] = tex1Dfetch(hexaTriTex, (tableIndex*16) + i + 2);
        v[2] = &vertlist[edge[2]];

        // calculate triangle surface normal
        float3 n = calcNormal(v[0], v[1], v[2]);

#ifndef COVISE
        if (index < (maxVerts - 2))
        {
           pos[index] = make_float4(*v[0], 1.0f);
           norm[index] = make_float4(n, 0.0f);

           pos[index+1] = make_float4(*v[1], 1.0f);
           norm[index+1] = make_float4(n, 0.0f);

           pos[index+2] = make_float4(*v[2], 1.0f);
           norm[index+2] = make_float4(n, 0.0f);

           if (map) {
              tex[index] = (maplist[edge[0]] - texMin) / (texMax - texMin);
              tex[index + 1] = (maplist[edge[1]] - texMin) / (texMax - texMin);
              tex[index + 2] = (maplist[edge[2]] - texMin) / (texMax - texMin);
           }
#ifdef LIC
           if (licResult) {
              licResult[index] = licMaplist[edge[0]];
              licResult[index + 1] = licMaplist[edge[1]];
              licResult[index + 2] = licMaplist[edge[2]];
           }
           // test bounding box
           // (0.000000 36.600006, 0.000000 22.650002, 1.999428 2.000020)
           // (0.000000 67.5, -8.455 8.455, -8.455 12.957)
           if (licTex) {
              /*
              licTex[index] = make_float2(v[0]->x / 67.5, (v[0]->y + 8.455) / 16.91);
              licTex[index + 1] = make_float2(v[1]->x / 67.5, (v[1]->y + 8.455) / 16.91);
              licTex[index + 2] = make_float2(v[2]->x / 67.5, (v[2]->y + 8.455) / 16.91);
              */
              licTex[index] = make_float2(v[0]->x / 36.6, v[0]->y / 22.65);
              licTex[index + 1] = make_float2(v[1]->x / 36.6, v[1]->y / 22.65);
              licTex[index + 2] = make_float2(v[2]->x / 36.6, v[2]->y / 22.65);

           }
#endif
        }        
#else
	int yIndex = index + totalVertices;
	int zIndex = index + totalVertices * 2;

        if (index < (maxVerts - 2))
        {
	  vertices[index] = v[0]->x;
	  vertices[index + 1] = v[1]->x;
	  vertices[index + 2] = v[2]->x;

	  vertices[yIndex] = v[0]->y;
	  vertices[yIndex + 1] = v[1]->y;
	  vertices[yIndex + 2] = v[2]->y;

	  vertices[zIndex] = v[0]->z;
	  vertices[zIndex + 1] = v[1]->z;
	  vertices[zIndex + 2] = v[2]->z;
        }
        //printf("  index %d done\n", index);
#endif
    }
    }
}

__device__ void generateCuttingTrianglesTetra(float4 *pos, float4 *norm,
                                              float *tex, float2 *licTex,
                                       float* vertexArrayIn, int numCoords, 
                                       uint* compactedThetraArray,
				       uint* vertsScan, uint maxVerts,
                                       int* typeList, int* elemList,
                                       int* connList, uint activeThetra,
				       float* values, float *map, float3 *licMap,
                                       float3 *licResult, 
                                       float texMin, float texMax, float nx,
                                       float ny, float nz, float dist)
{
    uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (elemIdx < activeThetra)  
    {

    int idx = compactedThetraArray[elemIdx];
    int i = elemList[idx];
    // our points are given directly in the unstructured grid
    // the connList array at position i has the positions inside x, y, z

    // calculate cell vertex positions
    //int xi = connList[i];
    //int yi = connList[i] + numCoords;
    //int zi = connList[i] + (numCoords << 1);
    
    float3 v[4];
    float field[4];
#ifndef COVISE
    float mapping[4];
#endif
    int xi, yi, zi;

    xi = connList[i + 3];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[0] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[0] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);

#ifndef COVISE
    if (map)
       mapping[0] = map[xi];
#endif

    xi = connList[i + 2];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[1] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[1] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);

#ifndef COVISE
    if (map)
       mapping[1] = map[xi];
#endif

    xi = connList[i + 1];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[2] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[2] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);

#ifndef COVISE
    if (map)
       mapping[2] = map[xi];
#endif

    xi = connList[i + 0];
    yi = xi + numCoords;
    zi = yi + numCoords;
    v[3] = make_float3(vertexArrayIn[xi], vertexArrayIn[yi], vertexArrayIn[zi]);
    field[3] = (nx * vertexArrayIn[xi] + ny * vertexArrayIn[yi] + nz * vertexArrayIn[zi] - dist);

#ifndef COVISE
    if (map)
       mapping[3] = map[xi];
#endif

    // recalculate flag
    // (this is faster than storing it in global memory)    
    uint tableIndex;
    tableIndex =  uint(field[0] < 0.0); 
    tableIndex += uint(field[1] < 0.0)*2; 
    tableIndex += uint(field[2] < 0.0)*4; 
    tableIndex += uint(field[3] < 0.0)*8; 

    // find the vertices where the surface intersects it

    float3 vertlist[6]; //one per edge
    float maplist[6];
    vertlist[0] = interp(0.0, v[0], v[1], field[0], field[1], mapping[0], mapping[1], &maplist[0]);
    vertlist[1] = interp(0.0, v[0], v[2], field[0], field[2], mapping[0], mapping[2], &maplist[1]);
    vertlist[2] = interp(0.0, v[0], v[3], field[0], field[3], mapping[0], mapping[3], &maplist[2]);
    vertlist[3] = interp(0.0, v[1], v[2], field[1], field[2], mapping[1], mapping[2], &maplist[3]);
    vertlist[4] = interp(0.0, v[1], v[3], field[1], field[3], mapping[1], mapping[3], &maplist[4]);
    vertlist[5] = interp(0.0, v[2], v[3], field[2], field[3], mapping[2], mapping[3], &maplist[5]);

    // output triangle vertices
    uint numVerts = tex1Dfetch(tetraNumVertsTex, tableIndex);

    for(int i=0; i<numVerts; i+=3) 
    {
       uint index = vertsScan[idx] + i;

        float3 *v[3];
        uint edge[3];
        edge[0] = tex1Dfetch(tetraTriTex, (tableIndex*6) + i);
        v[0] = &vertlist[edge[0]];

        edge[1] = tex1Dfetch(tetraTriTex, (tableIndex*6) + i + 1);
        v[1] = &vertlist[edge[1]];

        edge[2] = tex1Dfetch(tetraTriTex, (tableIndex*6) + i + 2);
        v[2] = &vertlist[edge[2]];

        // calculate triangle surface normal
        float3 n = calcNormal(v[0], v[1], v[2]);

        if (index < (maxVerts - 2)) 
        {
           pos[index] = make_float4(*v[0], 1.0f);
            norm[index] = make_float4(n, 0.0f);

            pos[index+1] = make_float4(*v[1], 1.0f);
            norm[index+1] = make_float4(n, 0.0f);

            pos[index+2] = make_float4(*v[2], 1.0f);
            norm[index+2] = make_float4(n, 0.0f);

            if (map) {
               tex[index] = (maplist[edge[0]] - texMin) / (texMax - texMin);
               tex[index + 1] = (maplist[edge[1]] - texMin) / (texMax - texMin);
               tex[index + 2] = (maplist[edge[2]] - texMin) / (texMax - texMin);
            }
        }
    }

    }
}


#ifndef COVISE
__global__ void generateIsoTriangles(float4 *pos, float4 *norm, float *tex,
                                     float* vertexArrayIn, int numCoords,
                                     uint* compactedArray,
                                     uint* vertsScan, uint maxVerts,
                                     int* typeList, int* elemList,
                                     int* connList, uint activeElems,
                                     float* values, float *map, float texMin,
                                     float texMax, float isoValue)
{
   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (elemIdx < activeElems)  
   {
      //uint elemType = typeList[elemIdx];
      uint elemType = typeList[compactedArray[elemIdx]];

      switch (elemType) {

          case TYPE_HEXAEDER:
             generateIsoTrianglesHexa(pos, norm, tex, vertexArrayIn, numCoords,
                                      compactedArray, vertsScan, maxVerts, typeList,
                                      elemList, connList, activeElems,
                                      values, map, texMin, texMax, isoValue);
             break;
          case TYPE_PYRAMID:
             generateIsoTrianglesPyr(pos, norm, tex, vertexArrayIn, numCoords,
                                      compactedArray, vertsScan, maxVerts, typeList,
                                      elemList, connList, activeElems,
                                      values, map, texMin, texMax, isoValue);
             break;
          case TYPE_TETRAHEDER:
             generateIsoTrianglesTetra(pos, norm, tex, vertexArrayIn, numCoords,
                                       compactedArray, vertsScan, maxVerts, typeList,
                                       elemList, connList, activeElems,
                                       values, map, texMin, texMax, isoValue);
             break;
      }
   }
}
#else
__global__ void generateIsoTrianglesC(int totalVertices, float *vertices,
                                      float *normals, float* vertexArrayIn,
                                      int numCoords, uint* compactedArray,
                                      uint* vertsScan, uint maxVerts,
                                      int* typeList,
                                      int* elemList, int* connList,
                                      uint activeElems,
                                      float* values, float isoValue)
{
   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (elemIdx < activeElems)  
   {
      uint elemType = typeList[compactedArray[elemIdx]];

      switch (elemType) {

          case TYPE_HEXAEDER:
             generateIsoTrianglesHexaC(totalVertices, vertices, normals, vertexArrayIn, numCoords,
                                       compactedArray, vertsScan, maxVerts, typeList, elemList,
                                       connList, activeElems, values, isoValue);
             break;
          case TYPE_PYRAMID:
             generateIsoTrianglesPyrC(totalVertices, vertices, normals, vertexArrayIn, numCoords,
                                      compactedArray, vertsScan, maxVerts, typeList, elemList,
                                      connList, activeElems, values, isoValue);
             break;
          case TYPE_TETRAHEDER:
             generateIsoTrianglesTetraC(totalVertices, vertices, normals, vertexArrayIn, numCoords,
                                        compactedArray, vertsScan, maxVerts, typeList, elemList,
                                        connList, activeElems, values, isoValue);
             break;
      }
   }
}
#endif

#ifndef COVISE
__global__ void generateCuttingTriangles(float4 *pos, float4 *norm, float *tex,
                                         float2 *licTex,
                                         float* vertexArrayIn, int numCoords,
                                         uint* compactedArray,
                                         uint* vertsScan, uint maxVerts,
                                         int* typeList, int* elemList,
                                         int* connList, uint activeElems,
                                         float* values, float *map,
                                         float3 *licMap, float3 *licResult,
                                         float texMin, float texMax, float nx,
                                         float ny, float nz, float dist)
{
   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (elemIdx < activeElems)  
   {
      uint elemType = typeList[compactedArray[elemIdx]];

      switch (elemType) {

          case TYPE_HEXAEDER:
             generateCuttingTrianglesHexa(pos, norm, tex, licTex,
                                          vertexArrayIn, numCoords,
                                          compactedArray, vertsScan, maxVerts,
                                          typeList, elemList, connList,
                                          activeElems, values, map, licMap,
                                          licResult,
                                          texMin, texMax, nx, ny, nz, dist);
             break;
/*
          case TYPE_PYRAMID:
             generateCuttingTrianglesPyr(pos, norm, tex, vertexArrayIn, numCoords,
                                      compactedArray, vertsScan, maxVerts, typeList,
                                      elemList, connList, activeElems,
                                      values, map, texMin, texMax, nx, ny, nz, dist);
             break;
*/
          case TYPE_TETRAHEDER:
             generateCuttingTrianglesTetra(pos, norm, tex, licTex,
                                           vertexArrayIn, numCoords,
                                           compactedArray, vertsScan, maxVerts,
                                           typeList, elemList, connList,
                                           activeElems, values, map, licMap,
                                           licResult,
                                           texMin, texMax, nx, ny, nz, dist);
             break;
      }
   }
}
#else
__global__ void generateCuttingTrianglesC(int totalVertices, float *vertices,
                                      float *normals, float* vertexArrayIn,
                                      int numCoords, uint* compactedArray,
                                      uint* vertsScan, uint maxVerts,
                                      int* typeList,
                                      int* elemList, int* connList,
                                      uint activeElems,
                                      float* values, float nx, float ny, float nz, float dist)
{
   uint elemIdx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if (elemIdx < activeElems)  
   {
      uint elemType = typeList[compactedArray[elemIdx]];

      switch (elemType) {

          case TYPE_HEXAEDER:
             generateCuttingTrianglesHexaC(totalVertices, vertices, normals, vertexArrayIn, numCoords,
                                       compactedArray, vertsScan, maxVerts, typeList, elemList,
                                       connList, activeElems, values, nx, ny, nz, dist);
             break;
/*
          case TYPE_PYRAMID:
             generateCuttingTrianglesPyrC(totalVertices, vertices, normals, vertexArrayIn, numCoords,
                                      compactedArray, vertsScan, maxVerts, typeList, elemList,
                                      connList, activeElems, values, nx, ny, nz, dist);
             break;
          case TYPE_TETRAHEDER:
             generateCuttingTrianglesTetraC(totalVertices, vertices, normals, vertexArrayIn, numCoords,
                                        compactedArray, vertsScan, maxVerts, typeList, elemList,
                                        connList, activeElems, values, nx, ny, nz, dist);
             break;
*/
      }
   }
}
#endif

#endif
