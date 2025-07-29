#pragma once

// std
#include <cmath>
#include <numeric>
#include <cstdio>
// anari
#include <anari/anari_cpp.hpp>
#include <anari/anari_cpp/ext/glm.h>
// ours

anari::Surface readCTF(anari::Device device, std::string fn, float radius)
{

  unsigned int num_atoms = 0;
  unsigned int num_bonds = 0;
  char label[24];

  FILE* fp = fopen(fn.c_str(),"r");
  if(!fp) return NULL;
  
  //read the first line with the number of atoms
  char line[1024];
  char* res = fgets(line, sizeof(line), fp);
  if (res == NULL) return NULL;
  sscanf(line,"%s %u", label, &num_atoms);
  printf("Label: %s, Number of atoms: %u\n", label, num_atoms);

  // Allocate arrays for positions and colors 
  auto atom_positions_array =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, num_atoms);

  auto atoms_colors_array =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, num_atoms);

  auto *atom_positions = anari::map<glm::vec3>(device, atom_positions_array);
  auto *atom_colors = anari::map<glm::vec3>(device, atoms_colors_array);
 
  // Read the positions and colors of the atoms from the file
  for (uint32_t i = 0; i < num_atoms; i++) {
    res = fgets(line, sizeof(line), fp);
    if (res == NULL) break;
    float radius;
    sscanf(line,"%f %f %f %f %f %f %f", &atom_positions[i][0],
                                        &atom_positions[i][1],
                                        &atom_positions[i][2],
                                        &radius,
                                        &atom_colors[i][0],
                                        &atom_colors[i][1],
                                        &atom_colors[i][2]);
  }
  anari::unmap(device, atom_positions_array);
  anari::unmap(device, atoms_colors_array);

  #if 0
  //read the second line with the number of bonds
  res = fgets(line, sizeof(line), fp);
  if (res == NULL) return NULL;
  sscanf(line,"%s %u", label, &num_bonds);
  printf("Label: %s, Number of bonds: %u\n", label, num_bonds);

  // Allocate arrays for bond positions and colors
  auto bond_positions_array =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, num_bonds * 2);

  auto bond_colors_array =
      anari::newArray1D(device, ANARI_FLOAT32_VEC3, num_bonds * 2);

  auto *bond_positions = anari::map<glm::vec3>(device, bond_positions_array);
  auto *bond_colors = anari::map<glm::vec3>(device, bond_colors_array);

  // Read the indices of the bonds from the file
  for (uint32_t i = 0; i < num_bonds; i++) {
    res = fgets(line, sizeof(line), fp);
    if (res == NULL) break;
    unsigned int idx1, idx2;
    float radius;
    sscanf(line,"%u %u %f", &idx1, &idx2, &radius);
    bond_positions[i * 2] = atom_positions[idx1];
    bond_positions[i * 2 + 1] = atom_positions[idx2];

    bond_colors[i * 2] = atom_colors[idx1];
    bond_colors[i * 2 + 1] = atom_colors[idx2];
  }
  anari::unmap(device, bond_positions_array);
  anari::unmap(device, bond_colors_array);
  #endif

  fclose(fp);  

  // Create and parameterize geometry //

  auto geometry = anari::newObject<anari::Geometry>(device, "sphere");
  anari::setAndReleaseParameter(
      device, geometry, "vertex.position", atom_positions_array);
  anari::setAndReleaseParameter(
      device, geometry, "vertex.color", atoms_colors_array);
  anari::setParameter(device, geometry, "radius", radius);
  anari::commitParameters(device, geometry);

  // Create and parameterize material //
  /*
  auto material = anari::newObject<anari::Material>(device, "matte");
  anari::setParameter(device, material, "color", "color");  
  anari::setParameter(device, material, "opacity", 0.7f);
  anari::setParameter(device, material, "alphaMode", "blend");
  anari::commitParameters(device, material);
  */

  // Create and parameterize material //
  auto material = anari::newObject<anari::Material>(device, "physicallyBased");
  anari::setParameter(device, material, "baseColor", "color");  

  //anari::setParameter(device, material, "opacity", 0.7f);
  //anari::setParameter(device, material, "alphaMode", "blend");

  anari::setParameter(device, material, "metallic", 0.1f);
  anari::setParameter(device, material, "roughness", 0.3f);
  anari::setParameter(device, material, "specular", 0.8f);
  anari::setParameter(device, material, "specularColor", "color");

  //anari::setParameter(device, material, "clearcoat", 1.0f);
  //anari::setParameter(device, material, "clearcoatRoughness", 0.8f);  

  anari::commitParameters(device, material);

  // Create and parameterize surface //

  auto surface = anari::newObject<anari::Surface>(device);
  anari::setAndReleaseParameter(device, surface, "geometry", geometry);
  anari::setAndReleaseParameter(device, surface, "material", material);
  anari::commitParameters(device, surface);
  return surface;
}
