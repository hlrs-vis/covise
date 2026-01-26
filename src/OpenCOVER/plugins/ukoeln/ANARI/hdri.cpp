/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION 1
#define TINYEXR_IMPLEMENTATION 1
#include <stb_image.h>
#include "tinyexr.h"

#include "hdri.h"

void HDRI::load(std::string fileName)
{
  // check the extension
  std::string extension = std::string(strrchr(fileName.c_str(), '.'));
  std::transform(extension.data(), extension.data() + extension.size(), 
      std::addressof(extension[0]), [](unsigned char c){ return std::tolower(c); });

  if (extension.compare(".hdr") != 0 && extension.compare(".exr") != 0) {
    throw std::runtime_error("Error: expected either a .hdr or a .exr file");
  }

  if (extension.compare(".hdr") == 0) {
    int w, h, n;
    float *imgData = stbi_loadf(fileName.c_str(), &w, &h, &n, STBI_rgb);
    width = w;
    height = h;
    numComponents = n;
    pixel.resize(w*h*n);
    memcpy(pixel.data(),imgData,w*h*n*sizeof(float));
    stbi_image_free(imgData);
  } else {
    int w, h, n;
    float* imgData;
    const char* err;
    int ret = LoadEXR(&imgData, &w, &h, fileName.c_str(), &err);
    if (ret != 0)
      throw std::runtime_error(std::string("Error, ") + std::string(err));
    n = 4;

    width = w;
    height = h;
    numComponents = n;
    pixel.resize(w*h*n);
    memcpy(pixel.data(),imgData,w*h*n*sizeof(float));
  }

}



