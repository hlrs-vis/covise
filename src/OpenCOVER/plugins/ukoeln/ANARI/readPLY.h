#pragma once

// std
#include <cassert>
#include <iostream>
// anari
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/glm.h>
// rply
#include "rply.h"

inline int rplyReadVertexCallback(p_ply_argument arg)
{
  long axis;
  void *pointer;
  ply_get_argument_user_data(arg, &pointer, &axis);
  auto **positions = (glm::vec3 **)pointer;
  float val = ply_get_argument_value(arg);
  assert(axis==0 || axis==1 || axis==2);

  static unsigned long count = 0;
  (*positions)[count][axis] = val;
  if (axis==2) count++;

  return 1;
}

inline int rplyReadColorCallback(p_ply_argument arg)
{
  long axis;
  void *pointer;
  ply_get_argument_user_data(arg, &pointer, &axis);
  auto **colors = (glm::vec3 **)pointer;
  float val = ply_get_argument_value(arg);
  assert(axis==0 || axis==1 || axis==2);

  static unsigned long count = 0;
  (*colors)[count][axis] = val/255.f;
  if (axis==2) count++;

  return 1;
}

anari::Surface readPLY(anari::Device device, std::string fn, float radius)
{
  p_ply ply = ply_open(fn.c_str(), NULL, 0, NULL);
  if (!ply || !ply_read_header(ply)) {
    std::cerr << "Cound not open file: " << fn << '\n';
    return nullptr;
  }

  glm::vec3 *positions = nullptr, *colors = nullptr;
  
  long numSpheres = ply_set_read_cb(ply, "vertex", "x", rplyReadVertexCallback, &positions, 0);
  ply_set_read_cb(ply, "vertex", "y", rplyReadVertexCallback, &positions, 1);
  ply_set_read_cb(ply, "vertex", "z", rplyReadVertexCallback, &positions, 2);

  ply_set_read_cb(ply, "vertex", "red", rplyReadColorCallback, &colors, 0);
  ply_set_read_cb(ply, "vertex", "green", rplyReadColorCallback, &colors, 1);
  ply_set_read_cb(ply, "vertex", "blue", rplyReadColorCallback, &colors, 2);

  auto positionsArray =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, numSpheres);

  auto colorsArray =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, numSpheres);

  positions = anari::map<glm::vec3>(device, positionsArray);
  colors = anari::map<glm::vec3>(device, colorsArray);

  if (!ply_read(ply)) {
    std::cerr << "Cound not read file: " << fn << '\n';
    return nullptr;
  }

  ply_close(ply);

#if 0
  glm::vec3 min(1e30), max(-1e30);
  glm::vec3 minColor(1e30), maxColor(-1e30);
  for (int i=0; i<numSpheres; ++i) {
    min.x = fminf(min.x, positions[i].x);
    min.y = fminf(min.y, positions[i].y);
    min.z = fminf(min.z, positions[i].z);

    max.x = fmaxf(min.x, positions[i].x);
    max.y = fmaxf(min.y, positions[i].y);
    max.z = fmaxf(min.z, positions[i].z);

    minColor.x = fminf(minColor.x, colors[i].x);
    minColor.y = fminf(minColor.y, colors[i].y);
    minColor.z = fminf(minColor.z, colors[i].z);

    maxColor.x = fmaxf(minColor.x, colors[i].x);
    maxColor.y = fmaxf(minColor.y, colors[i].y);
    maxColor.z = fmaxf(minColor.z, colors[i].z);
  }

  std::cout << "min: " << min.x << ',' << min.y << ',' << min.z << '\n';
  std::cout << "max: " << max.x << ',' << max.y << ',' << max.z << '\n';

  std::cout << "minColor: " << minColor.x << ',' << minColor.y << ',' << minColor.z << '\n';
  std::cout << "maxColor: " << maxColor.x << ',' << maxColor.y << ',' << maxColor.z << '\n';
#endif

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
  anari::setParameter(device, material, "color", "color");
  anari::commitParameters(device, material);

  // Create and parameterize surface //

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);
  return surface;
}
