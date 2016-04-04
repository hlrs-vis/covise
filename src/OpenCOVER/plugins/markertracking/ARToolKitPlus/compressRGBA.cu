

//#ifndef _COMPRESSRGBA_CU_
//#define _COMPRESSRGBA_CU_

#if BLOCKSPERBLOCK == 1
  #define COMPRESSA compressRGBA1
  #define DECOMPRESSA decompressRGBA1
#elif BLOCKSPERBLOCK == 2
  #define COMPRESSA compressRGBA2
  #define DECOMPRESSA decompressRGBA2
#elif BLOCKSPERBLOCK == 4
  #define COMPRESSA compressRGBA4
  #define DECOMPRESSA decompressRGBA4
#endif  



//!***************//
//! GPU FUNCTIONS //
//!***************//

//! COMPRESS //
__global__ void COMPRESSA(const unsigned char * image, uint2 * result, const uint picWidth) {
    const int BPB = BLOCKSPERBLOCK;

    // IDs //
    const int id = threadIdx.x;         // 1 to 16 per block
    const int blx = blockIdx.x;
    const int bly = blockIdx.y;
    const int pixblockId = threadIdx.y; // 0, 1, ..., BPB: pixelbl

    const int line = (id>>2);           // rounded, <<2 later, 0, 4, 8, 12,...


    // COORDINATES in input texture //
    const int i = blx*4*BPB + pixblockId*4 + (id - 4*line);
    const int j = bly*4 + line;

    const int offset = pixblockId*16;   // 16 values per pixelblock
    const int offset2 = pixblockId*4;   //  4 values per pixelblock

    __shared__ float3 colors[BPB*16];
    __shared__ float distances[BPB*16];
    __shared__ int ranks[BPB*16];


    // COLORS //
    colors[offset + id].x = image[i*4 + j*picWidth*4 + 0];
    colors[offset + id].y = image[i*4 + j*picWidth*4 + 1];
    colors[offset + id].z = image[i*4 + j*picWidth*4 + 2];


    // DISTANCE //
    distances[offset + id] = 0.3f*colors[offset + id].x + 0.59f*colors[offset + id].y + 0.11f*colors[offset + id].z;
      //__syncthreads(); nicht noetig, da immer alle 16 gleichzeitig (halfwarp)
    sort(distances, ranks);
      //__syncthreads(); nicht noetig, da immer alle 16 gleichzeitig (halfwarp)


    // SAVED COLORS //
    __shared__ float3 colorSet[BPB*4];
    __shared__ uint bright[BPB];
    __shared__ uint dark[BPB];

    if(ranks[offset + id]==0){
      // dark
      colorSet[offset2 + 0] = colors[offset + id];
      dark[pixblockId] = (__float2uint_rn(colorSet[offset2+0].x)&0xF8)>>3 | (__float2uint_rn(colorSet[offset2+0].y)&0xFC)<<3 | (__float2uint_rn(colorSet[offset2+0].z)&0xF8)<<8;
    }
    if(ranks[offset + id]==15){
      // bright
      colorSet[offset2 + 3] = colors[offset + id];
      bright[pixblockId] = (__float2uint_rn(colorSet[offset2+3].x)&0xF8)<<13 | (__float2uint_rn(colorSet[offset2+3].y)&0xFC)<<19 | (__float2uint_rn(colorSet[offset2+3].z)&0xF8)<<24;
    }
    __syncthreads();


    // INTERPOLATED COLORS //
    colorSet[offset2 + 1] = (colorSet[offset2 + 3] + 2.0f*colorSet[offset2 + 0])*0.33333f;
    colorSet[offset2 + 2] = (colorSet[offset2 + 3]*2.0f + colorSet[offset2 + 0])*0.33333f;
      //__syncthreads(); alle threads versuchens, aber nur einer kommt durch


    // INDICES //
    __shared__ uint res[16*BPB]; // id
    uint d0=0, d1=0, d2=0, d3=0;
    distance(colors[offset + id], colorSet[offset2 + 0], d0);
    distance(colors[offset + id], colorSet[offset2 + 1], d1);
    distance(colors[offset + id], colorSet[offset2 + 2], d2);
    distance(colors[offset + id], colorSet[offset2 + 3], d3);
    uint minDist = d0;
    uint index = 0;
    if(d1<minDist){
      index = 1;     // alle anderen threads warten so lange
      minDist = d1;  // trotzdem schneller
    }
    if(d2<minDist){
      index = 2;
      minDist = d2;
    }
    if(d3<minDist){
      index = 3;
    }
    res[id + offset] = index;
    __syncthreads();


    // SAVE INDICES//
    uint indices=0;
    #pragma unroll
    for(int h=0; h<16; h++){
      indices |= res[h + offset]<<(2*h);
    }
    result[(blx*BPB + pixblockId) + __float2int_rn(bly*picWidth*0.25f)].x = dark[pixblockId] | bright[pixblockId];
    result[(blx*BPB + pixblockId) + __float2int_rn(bly*picWidth*0.25f)].y = indices;

}



//! DECOMPRESS //
__global__ void DECOMPRESSA(const uint2 * input, unsigned char * output, const uint picWidth) {
    const int BPB = BLOCKSPERBLOCK;

    const int id = threadIdx.x;
    const int blx = blockIdx.x;
    const int bly = blockIdx.y;
    const int pixblockId = threadIdx.y; // 0, 1, ..., BPB

    const int line = id>>2; // <<2 later, 0, 4, 8, 12,...
    const int i = blx*4*BPB + pixblockId*4 + (id - 4*line);
    const int j = bly*4 + line;

    const int offset2 = pixblockId*4; // 4 values per pixelblock

    uint c = input[(blx*BPB + pixblockId) + __float2int_rn(bly*picWidth*0.25f)].x;
    uint3 colors[4*BPB];

    // dark color //
    colors[offset2].x = (c & 0x0000001F)<<3;
    colors[offset2].y = (c & 0x000007E0)>>3;
    colors[offset2].z = (c & 0x0000F800)>>8;

    // bright color //
    colors[offset2 + 3].x = (c & 0x001F0000)>>13;
    colors[offset2 + 3].y = (c & 0x07E00000)>>19;
    colors[offset2 + 3].z = (c & 0xF8000000)>>24;

    // interpolated colors //
    colors[offset2 + 1] = (colors[offset2 + 3] + 2*(colors[offset2 + 0]))/3;
    colors[offset2 + 2] = (colors[offset2 + 3]*2 + (colors[offset2 + 0]))/3;

    // indices //
    uint mask = 0x00000003;
    uint inp = (input[(blx*BPB + pixblockId) + __float2int_rn(bly*picWidth*0.25f)].y>>id*2) & mask;

    // output //
    if(inp==0){
      output[i*4 + j*picWidth*4 + 0] = colors[offset2 + 0].x;
      output[i*4 + j*picWidth*4 + 1] = colors[offset2 + 0].y;
      output[i*4 + j*picWidth*4 + 2] = colors[offset2 + 0].z;
      output[i*4 + j*picWidth*4 + 3] = 0xFF;
    } else if(inp==1){
      output[i*4 + j*picWidth*4 + 0] = colors[offset2 + 1].x;
      output[i*4 + j*picWidth*4 + 1] = colors[offset2 + 1].y;
      output[i*4 + j*picWidth*4 + 2] = colors[offset2 + 1].z;
      output[i*4 + j*picWidth*4 + 3] = 0xFF;
    } else if(inp==2){
      output[i*4 + j*picWidth*4 + 0] = colors[offset2 + 2].x;
      output[i*4 + j*picWidth*4 + 1] = colors[offset2 + 2].y;
      output[i*4 + j*picWidth*4 + 2] = colors[offset2 + 2].z;
      output[i*4 + j*picWidth*4 + 3] = 0xFF;
    } else if(inp==3){
      output[i*4 + j*picWidth*4 + 0] = colors[offset2 + 3].x;
      output[i*4 + j*picWidth*4 + 1] = colors[offset2 + 3].y;
      output[i*4 + j*picWidth*4 + 2] = colors[offset2 + 3].z;
      output[i*4 + j*picWidth*4 + 3] = 0xFF;
    }

}


#undef COMPRESSA
#undef DECOMPRESSA

//#endif
