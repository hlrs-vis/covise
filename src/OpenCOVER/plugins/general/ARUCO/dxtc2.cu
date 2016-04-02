

//!#######################//
//! S3TC/DXT1 compression //
//!  on GPU, using CUDA   //
//!   by Frank Naegele    //
//!   mail@f-naegele.de   //
//!#######################//


#define GLUT_NO_LIB_PRAGMA
//! INCLUDES //
#include <stdio.h>

// CUTIL //
#include <cutil.h>
#include <cutil_math.h>

// OpenGL //
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glext.h>

inline __host__ __device__ uint3 operator/(uint3 a, uint b)
{
    return make_uint3(a.x / b, a.y / b, a.z / b);
}


//! INCLUDE KERNEL //
#include "compress.cu"

#define BLOCKSPERBLOCK 2
  #include "compressRGB.cu"
  #include "compressRGBA.cu"
#undef  BLOCKSPERBLOCK

#define BLOCKSPERBLOCK 4
  #include "compressRGB.cu"
  #include "compressRGBA.cu"
#undef BLOCKSPERBLOCK

#define BLOCKSPERBLOCK 1
  #include "compressRGB.cu"
  #include "compressRGBA.cu"



//! FUNCTIONS //

void cudaCompression(unsigned int width, unsigned int height, unsigned char *uncompressedData, unsigned char *compressedData, int videoMode) {
  //! RUN COMPRESSION //

    //std::cout << "compressing: " << width << " " << height << std::endl;

    if(height%4 != 0){
      fprintf(stderr, "CUDA COMPRESSION ERROR: image height not multiple of 4\n");
      exit(EXIT_FAILURE);
    }

    int BPB = 0;
    if(width%16 == 0){
      BPB = 4;
    } else if (width%8 == 0) {
      BPB = 2;
    } else if (width%4 == 0) {
      BPB = 1;
    } else {
      fprintf(stderr, "CUDA COMPRESSION ERROR: image width not multiple of 4\n");
      exit(EXIT_FAILURE);
    }

  // ALLOCATE MEMORY AND COPY TO DEVICE //
    uint numOfBytes = 0;
    uint compressedSize = (width / 2) * (height / 4) * 4;
    uint uncompressedSize = 0;

    videoMode = GL_RGB;

    if(videoMode == GL_RGB || videoMode == GL_RGB8) {
      uncompressedSize = width * height * 3;
      numOfBytes = 3;
    } else if (videoMode == GL_RGBA || videoMode == GL_RGBA8) {
      uncompressedSize = width * height * 4;
      numOfBytes = 4;
    } else {
      fprintf(stderr, "Unknown videoMode: %x\n", videoMode);
      fprintf(stderr, "RGB:  %x %x\n", GL_RGB, GL_RGB8);
      fprintf(stderr, "RGBA: %x %x\n", GL_RGBA, GL_RGBA8);
      exit(EXIT_FAILURE); // videoMode not supported!
    }
    CUT_CONDITION( 0 != compressedSize );
    CUT_CONDITION( 0 != uncompressedSize );

    unsigned char * d_data    = NULL;  // data on GPU
    CUDA_SAFE_CALL( cudaMalloc((void**) &d_data, uncompressedSize) );
    CUDA_SAFE_CALL( cudaMemcpy(d_data, uncompressedData, uncompressedSize, cudaMemcpyHostToDevice) );

    uint2 * d_result = NULL;  // result on GPU
    CUDA_SAFE_CALL( cudaMalloc((void**) &d_result, compressedSize) );


  // RUN ON GPU //
    int numHorBlocks = width/4 / BPB;  // total amount of horizontal cuda thread blocks
    int numVertBlocks = height/4;      // ...and vertical
    dim3 grid(numHorBlocks, numVertBlocks);
    dim3 block(16, BPB);                    // x: 16 pixel/pixelblock, y: pixelblocks/cudablock

    if(numOfBytes == 3) {
      if(BPB==4){
        compressRGB4<<<grid, block>>>(d_data, d_result, width);
      } else if(BPB==2){
        compressRGB2<<<grid, block>>>(d_data, d_result, width);
      } else {
        compressRGB1<<<grid, block>>>(d_data, d_result, width);
      }
      CUT_CHECK_ERROR("compressRGB");
    } else if (numOfBytes == 4) {
      if(BPB==4){
        compressRGBA4<<<grid, block>>>(d_data, d_result, width);
      } else if(BPB==2){
        compressRGBA2<<<grid, block>>>(d_data, d_result, width);
      } else {
        compressRGBA1<<<grid, block>>>(d_data, d_result, width);
      }
      CUT_CHECK_ERROR("compressRGBA");
    }


  // COPY BACK AND FREE MEMORY //
    CUDA_SAFE_CALL( cudaMemcpy((unsigned char *)compressedData, d_result, compressedSize, cudaMemcpyDeviceToHost) );

    CUDA_SAFE_CALL(cudaFree(d_data));
    CUDA_SAFE_CALL(cudaFree(d_result));

}



void cudaDecompression(unsigned int width, unsigned int height, unsigned char *compressedData, unsigned char *uncompressedData, int videoMode) {
  //! RUN DECOMPRESSION //

    if(height%4 != 0){
      fprintf(stderr, "CUDA DECOMPRESSION ERROR: image height not multiple of 4\n");
      exit(EXIT_FAILURE);
    }

    int BPB = 0;
    if(width%16 == 0){
      BPB = 4;
    } else if (width%8 == 0) {
      BPB = 2;
    } else if (width%4 == 0) {
      BPB = 1;
    } else {
      fprintf(stderr, "CUDA DECOMPRESSION ERROR: image width not multiple of 4\n");
      exit(EXIT_FAILURE);
    }

  // ALLOCATE MEMORY AND COPY TO DEVICE //
    uint numOfBytes = 0;
    uint uncompressedSize = 0;
    if(videoMode == GL_RGB || videoMode == GL_RGB8) {
      uncompressedSize = width * height * 3;                 // in Byte, 4 Bytes per Pixel (32bpp)
      numOfBytes = 3;
    } else if (videoMode == GL_RGBA || videoMode == GL_RGBA8) {
      uncompressedSize = width * height * 4;                 // in Byte, 4 Bytes per Pixel (32bpp)
      numOfBytes = 4;
    } else {
      exit(EXIT_FAILURE); // videoMode not supported!
    }

    const uint compressedSize = (width / 4) * (height / 4) * 8;  // w * h * 0.5 in Byte (4bpp)

    uint2 * d_inputDe = NULL; // data on GPU
    CUDA_SAFE_CALL( cudaMalloc((void**) &d_inputDe, compressedSize) );
    CUDA_SAFE_CALL( cudaMemcpy(d_inputDe, compressedData, compressedSize, cudaMemcpyHostToDevice) );

    unsigned char * d_outputDe = NULL; // result on GPU
    CUDA_SAFE_CALL( cudaMalloc((void**) &d_outputDe, uncompressedSize) );


  // RUN ON GPU //
    int numHorBlocks = width/4 / BPB;  // total amount of horizontal cuda thread blocks
    int numVertBlocks = height/4;      // ...and vertical
    dim3 gridDe(numHorBlocks, numVertBlocks);
    dim3 blockDe(16, BPB);

    if(numOfBytes == 3) {
      if(BPB==4){
        decompressRGB4<<<gridDe, blockDe>>>(d_inputDe, d_outputDe, width);
      } else if(BPB==2) {
        decompressRGB2<<<gridDe, blockDe>>>(d_inputDe, d_outputDe, width);
      } else {
        decompressRGB1<<<gridDe, blockDe>>>(d_inputDe, d_outputDe, width);
      }
      CUT_CHECK_ERROR("decompressRGB");
    } else if (numOfBytes == 4) {
      if(BPB==4){
        decompressRGBA4<<<gridDe, blockDe>>>(d_inputDe, d_outputDe, width);
      } else if(BPB==2){
        decompressRGBA2<<<gridDe, blockDe>>>(d_inputDe, d_outputDe, width);
      } else {
        decompressRGBA1<<<gridDe, blockDe>>>(d_inputDe, d_outputDe, width);
      }
      CUT_CHECK_ERROR("decompressRGBA");
    }


  // COPY BACK AND FREE MEMORY //
    CUDA_SAFE_CALL(cudaMemcpy((unsigned char *)uncompressedData, d_outputDe, uncompressedSize, cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaFree(d_inputDe));
    CUDA_SAFE_CALL(cudaFree(d_outputDe));

}

void init_cuda()
{
  int argc=0;
  char *argv[1];
  CUT_DEVICE_INIT(argc, argv); // init cuda
}

void close_cuda()
{
  ;
}
