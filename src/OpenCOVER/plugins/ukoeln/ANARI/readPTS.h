#pragma once

// std
#include <cmath>
#include <numeric>
#include <cstdio>
// anari
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/glm.h>

anari::Surface readPTS(anari::Device device, std::string fn)
{
  const float radius = .5f;

  unsigned int numSpheres = 0;
  FILE* fp = fopen(fn.c_str(),"r");
  if(!fp) return NULL;
  
  char line[1024];
  char* res = fgets(line, sizeof(line), fp);
  if (res == NULL) return NULL;
  sscanf(line,"%u",&numSpheres);
  
  auto positionsArray =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, numSpheres);

  auto colorsArray =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, numSpheres);

  auto *positions = anari::map<glm::vec3>(device, positionsArray);
  auto *colors = anari::map<glm::vec3>(device, colorsArray);
 
  for (uint32_t i = 0; i < numSpheres; i++) {
    res = fgets(line, sizeof(line), fp);
    if (res == NULL) break;
    unsigned a,r,g,b;
    sscanf(line,"%f %f %f %u %u %u %u",&positions[i][0],&positions[i][1],&positions[i][2],&a,&r,&g,&b);
    colors[i][0] = r / 255.f;
    colors[i][1] = g / 255.f;
    colors[i][2] = b / 255.f;
  }

  anari::unmap(device, positionsArray);
  anari::unmap(device, colorsArray);
  

  // Create and parameterize geometry //

  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(
      device, geometry, "vertex.position", positionsArray);
  anari::setAndReleaseParameter(
      device, geometry, "vertex.color", colorsArray);
  anari::setParameter(device, geometry, "radius", radius);
  anari::commitParameters(device, geometry);

  // Create and parameterize material //

  auto material = anari::newObject<anari::Material>(device, "matte");
//  anari::setAndReleaseParameter(device, material, "color", texture);
  anari::commitParameters(device, material);

  // Create and parameterize surface //

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);
  return surface;
}
