// Copyright 2023 Stefan Zellmann and Jefferson Amstutz
// SPDX-License-Identifier: Apache-2.0

#pragma once

// std
#include <H5Cpp.h>
#include <array>
#include <cassert>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <vector>
#include <iostream>
// ours
#include "FieldTypes_anari.h"
#include "hdf5.h"
#define MAX_STRING_LENGTH 80


// Data structure of info field
// Kind of unneccessary information at the moment
struct sim_info_t
{
  int file_format_version;
  char setup_call[400];
  char file_creation_time[MAX_STRING_LENGTH];
  char flash_version[MAX_STRING_LENGTH];
  char build_date[MAX_STRING_LENGTH];
  char build_dir[MAX_STRING_LENGTH];
  char build_machine[MAX_STRING_LENGTH];
  char cflags[400];
  char fflags[400];
  char setup_time_stamp[MAX_STRING_LENGTH];
  char build_time_stamp[MAX_STRING_LENGTH];
};
#ifdef __GNUC__
#define PACK(...) __VA_ARGS__  __attribute__((__packed__))
#endif

#ifdef _MSC_VER
#define PACK(...) __pragma( pack(push, 1) ) __VA_ARGS__  __pragma( pack(pop))
#endif
// Data structure of grid
struct grid_t
{
  // Character type for field names
  typedef std::array<char, 4> char4;
  // Data structure of 3 component vector
  typedef PACK(struct
  {
    double x, y, z;
  }) vec3d;
  // Data structure for bounding boxes
  // lower and upper corner
  typedef struct
  {
    vec3d min, max;
  } aabbd;

  // Data structure for the gid
  // 6 neighbours, 1 parent, 8 children
  PACK(struct gid_t
  {
    int neighbors[6];
    int parent;
    int children[8];
  });

  // Vector of field names
  std::vector<char4> unknown_names;
  // Vector field of refinement levels
  std::vector<int> refine_level;
  // Vector of node types
  std::vector<int> node_type; // node_type 1 ==> leaf
  // Vector of global ID
  std::vector<gid_t> gid;
  // Vector of block cooridnates
  std::vector<vec3d> coordinates;
  // Vector of block size
  std::vector<vec3d> block_size;
  // Vector of bounding box
  std::vector<aabbd> bnd_box;
  // Vector of child id (1-8)
  std::vector<int> which_child;
  // Vector of sinkList
};

struct particle_t
{
  // Index of position, velocity and mass of particles
  int ind_posx;
  int ind_posy;
  int ind_posz;
  
  int ind_velx;
  int ind_vely;
  int ind_velz;
  
  int ind_mass;

  // Number of particles and properties
  int npart;
  int nprop;

  std::vector<double> data;
};

// Data structure of general information
struct variable_t
{
  size_t global_num_blocks;
  size_t nxb;
  size_t nyb;
  size_t nzb;

  std::vector<double> data;
};

// Read general information of the simulation
inline void read_sim_info(sim_info_t &dest, H5::H5File const &file)
{
  H5::StrType str80(H5::PredType::C_S1, 80);
  H5::StrType str400(H5::PredType::C_S1, 400);

  H5::CompType ct(sizeof(sim_info_t));
  ct.insertMember("file_format_version", 0, H5::PredType::NATIVE_INT);
  ct.insertMember("setup_call", 4, str400);
  ct.insertMember("file_creation_time", 404, str80);
  ct.insertMember("flash_version", 484, str80);
  ct.insertMember("build_date", 564, str80);
  ct.insertMember("build_dir", 644, str80);
  ct.insertMember("build_machine", 724, str80);
  ct.insertMember("cflags", 804, str400);
  ct.insertMember("fflags", 1204, str400);
  ct.insertMember("setup_time_stamp", 1604, str80);
  ct.insertMember("build_time_stamp", 1684, str80);

  H5::DataSet dataset = file.openDataSet("sim info");

  dataset.read(&dest, ct);
}

// Read grid data
inline void read_grid(grid_t &dest, H5::H5File const &file)
{
  // Initialize the dataset and dataspace
  H5::DataSet dataset;
  H5::DataSpace dataspace;

  // Open grid datasets
  // Field names
  {
    // Define str4 as 4 character string
    H5::StrType str4(H5::PredType::C_S1, 4);
    // Open dataset
    dataset = file.openDataSet("unknown names");
    // Get dataspace
    dataspace = dataset.getSpace();
    // Allocate 1D space for data (nblocks * nx * ny * nz)
    dest.unknown_names.resize(dataspace.getSimpleExtentNpoints());
    // Read data from dataset and store in dest as str4
    dataset.read(dest.unknown_names.data(), str4, dataspace, dataspace);
  }
  // Refinement levels
  {
    dataset = file.openDataSet("refine level");
    dataspace = dataset.getSpace();
    dest.refine_level.resize(dataspace.getSimpleExtentNpoints());
    // Read data from dataset and store in dest as integer
    dataset.read(dest.refine_level.data(),
        H5::PredType::NATIVE_INT,
        dataspace,
        dataspace);
  }
  // Node types
  {
    dataset = file.openDataSet("node type");
    dataspace = dataset.getSpace();
    dest.node_type.resize(dataspace.getSimpleExtentNpoints());
    // Read data from dataset and store in dest as integer
    dataset.read(
        dest.node_type.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
  }
  // Global identifier
  {
    dataset = file.openDataSet("gid");
    dataspace = dataset.getSpace();
    // Initialize dims array 
    hsize_t dims[2];
    // Store dimensions of gid in dims
    dataspace.getSimpleExtentDims(dims);
    // Allocate gid with size of number of blocks only
    dest.gid.resize(dims[0]);
    // Make sure that second dimension of dataset has length 15
    // 6 neighbours + 1 parent + 8 neighbours
    assert(dims[1] == 15);
    // Read data from dataset and store in dest as integer
    dataset.read(
        dest.gid.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
  }
  // Block coordinates
  {
    dataset = file.openDataSet("coordinates");
    dataspace = dataset.getSpace();

    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims);
    // Allocate gid with size of number of blocks only
    dest.coordinates.resize(dims[0]);
    // Make sure that second dimension of dataset has length 3
    assert(dims[1] == 3);
    // Read data from dataset and store in dest as float 8
    // Will cut to float 4 if data is not float 8
    dataset.read(dest.coordinates.data(),
        H5::PredType::NATIVE_DOUBLE,
        dataspace,
        dataspace);
  }
  // Block sizes
  {
    dataset = file.openDataSet("block size");
    dataspace = dataset.getSpace();

    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims);
    // Allocate gid with size of number of blocks only
    dest.block_size.resize(dims[0]);
    // Make sure that second dimension of dataset has length 3
    assert(dims[1] == 3);
    // Read data from dataset and store in dest as float 8
    // Will cut to float 4 if data is not float 8
    dataset.read(dest.block_size.data(),
        H5::PredType::NATIVE_DOUBLE,
        dataspace,
        dataspace);
  }
  // Bounding boxes
  {
    dataset = file.openDataSet("bounding box");
    dataspace = dataset.getSpace();

    hsize_t dims[3];
    dataspace.getSimpleExtentDims(dims);
    // Allocate bnd_box with size of 2 times the number of blocks
    // Resized again lower, unneccesary?
    dest.bnd_box.resize(dims[0] * 2);
    // Make sure that second and third dimension are 
    // of length 3 and 2 respectively
    assert(dims[1] == 3);
    assert(dims[2] == 2);

    // Allocate a 1D temporary vector of size (nblocks * ndim * 2)
    std::vector<double> temp(dims[0] * dims[1] * dims[2]);

    // Read data from dataset and store in temp as float 8
    // Will cut to float 4 if data is not float 8
    dataset.read(
      temp.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace
    );

    // Allocate bnd_box with the number of blocks only
    dest.bnd_box.resize(dims[0]);
    // Loop over all blocks
    for (size_t i = 0; i < dims[0]; ++i) {
      // Set position of lower and upper corner
      // Order xmin, xmax, ymin, ymax, zmin, zmax
      dest.bnd_box[i].min.x = temp[i * 6];
      dest.bnd_box[i].max.x = temp[i * 6 + 1];
      dest.bnd_box[i].min.y = temp[i * 6 + 2];
      dest.bnd_box[i].max.y = temp[i * 6 + 3];
      dest.bnd_box[i].min.z = temp[i * 6 + 4];
      dest.bnd_box[i].max.z = temp[i * 6 + 5];
      // std::cout << dest.bnd_box[i].min.x << ' ' << dest.bnd_box[i].min.y << '
      // ' << dest.bnd_box[i].min.z << '\n'; std::cout << dest.bnd_box[i].max.x
      // << ' ' << dest.bnd_box[i].max.y << ' ' << dest.bnd_box[i].max.z <<
      // '\n';
    }
  }
  // Child IDs
  {
    dataset = file.openDataSet("which child");
    dataspace = dataset.getSpace();
    dest.which_child.resize(dataspace.getSimpleExtentNpoints());
    // Read data from dataset and store in dest as integer
    dataset.read(dest.which_child.data(),
      H5::PredType::NATIVE_INT,
      dataspace,
      dataspace
    );
  }
}

inline void read_sinks(particle_t &part, H5::H5File const &file)
{
  // Initialize the dataset and dataspace
  H5::DataSet dataset;
  H5::DataSpace dataspace;

  // Open grid datasets
  // Field names
  {
    // Open dataset
    dataset = file.openDataSet("sinkList");
    // Get dataspace
    dataspace = dataset.getSpace();
    // Allocate 1D space for data (nsinks * nprops)
    int tot_entries = dataspace.getSimpleExtentNpoints();
    part.data.resize(tot_entries);
    // Read data from dataset and store in dest as sinkList
    dataset.read(part.data.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);

    // Set indices of datasets (position, velocity, mass)
    part.ind_posx = 0;
    part.ind_posy = 1;
    part.ind_posz = 2;
    part.ind_velx = 3;
    part.ind_vely = 4;
    part.ind_velz = 5;
    part.ind_mass = 59;

    // Loop over all possible number of particles
    // Particles are always allocated in chucks of 100
    for (size_t i=1; i < 100; ++i)
    {
      // Determine number of particles in current iteration step
      part.npart = 100 * i;
      // Check if we found the correct size
      // Fraction of total number of entries to number of particles
      // should result in number of properties
      // First condition checks that we are in the right ball park
      // second condition check that the fraction is round.
      if ((tot_entries / (100 * i) < 150) && (tot_entries % (100 * i) == 0))
      {
        // Determine number of properties
        // and exit loop
        part.nprop = tot_entries / (100 * i);
        break;
      }
    }
  }
  std::cout << "Number of particles: " << part.npart << "\n";
  std::cout << "Number of properties: " << part.nprop << "\n";
}

// Read variable field
inline void read_variable(
    variable_t &var, H5::H5File const &file, char const *varname)
{
  H5::DataSet dataset = file.openDataSet(varname);
  H5::DataSpace dataspace = dataset.getSpace();

  // std::cout << dataspace.getSimpleExtentNdims() << '\n';
  // Get dimension of field
  hsize_t dims[4];
  dataspace.getSimpleExtentDims(dims);
  // Store dimensions in dataset
  var.global_num_blocks = dims[0];
  var.nxb = dims[1];
  var.nyb = dims[2];
  var.nzb = dims[3];
  // Allocate data as 1D array
  var.data.resize(dims[0] * dims[1] * dims[2] * dims[3]);
  // Read data from dataset and store in var as float 8
  dataset.read(
    var.data.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace
  );
  // std::cout << dims[0] << ' ' << dims[1] << ' ' << dims[2] << ' ' << dims[3]
  // << '\n';
}

inline ParticleField toParticleField(particle_t particles)
{
  // Create output dataset
  ParticleField result;
  
  std::cout << particles.npart << "\t" << particles.nprop << "\n";
  
  // Count all particles that exist
  int cnt_part = 0;

  for (size_t part_ind=0; part_ind < particles.npart; ++part_ind)
  {
    bool particle_exist = false;
    for (size_t prop_ind=0; prop_ind < particles.nprop; ++prop_ind)
    {
      if (particles.data[part_ind * particles.nprop + prop_ind] != 0.0)
      {
        particle_exist = true;
        break;
      }
    }
    if (particle_exist)
    {
      partVec position;
      partVec velocity;

      // Get position of current particle
      position.x = particles.data[part_ind * particles.nprop + particles.ind_posx];
      position.y = particles.data[part_ind * particles.nprop + particles.ind_posy];
      position.z = particles.data[part_ind * particles.nprop + particles.ind_posz];
      
      // Get velocity of current particle
      velocity.x = particles.data[part_ind * particles.nprop + particles.ind_velx];
      velocity.y = particles.data[part_ind * particles.nprop + particles.ind_vely];
      velocity.z = particles.data[part_ind * particles.nprop + particles.ind_velz];

      // Store data of current particle
      result.particlePosition.push_back(position);
      result.particlePosition.push_back(position);
      result.particleMass.push_back(particles.data[part_ind * particles.nprop + particles.ind_mass]);

      // Increase counter of active particles
      cnt_part++;
    }

    // Store total number of particles
    result.nrpart = cnt_part;
  }
  return result;
}

// Convert gridData to AMRfield data structure
inline AMRField toAMRField(
  const grid_t &grid, const variable_t &var, BlockBoundsf &reg_roi,
  int usr_min_level, int usr_max_level, bool usr_set_log
  )
{
  AMRField result;

  bool *in_reg = new bool[var.global_num_blocks];
  size_t bid_first = 0, bid_last = 0;

  int common_ref = 1000;
  int max_ref = 0;

  for (size_t i = 0; i < var.global_num_blocks; ++i) {
    if (grid.node_type[i] != 1) {
      in_reg[i] = false;
      continue;
    }
    if (grid.bnd_box[i].max.x < reg_roi[0] || grid.bnd_box[i].min.x > reg_roi[1]) {
      in_reg[i] = false;
      continue;
    }
    if (grid.bnd_box[i].max.y < reg_roi[2] || grid.bnd_box[i].min.y > reg_roi[3]) {
      in_reg[i] = false;
      continue;
    }
    if (grid.bnd_box[i].max.z < reg_roi[4] || grid.bnd_box[i].min.z > reg_roi[5]) {
      in_reg[i] = false;
      continue;
    }
    in_reg[i] = true;

    // Determine minimum and maximum refinement level within region
    if (grid.refine_level[i] < common_ref) {
      common_ref = grid.refine_level[i];
    }
    if (grid.refine_level[i] > max_ref) {
      max_ref = grid.refine_level[i];
    }
  }
  
  std::cout << "Common refinement level: " << common_ref << "\n";
  std::cout << "Maximum refinement level: " << max_ref << "\n";
  // If user limit is lower than common refinement level
  // Adjust the common ref level
  if (common_ref > usr_max_level) {
    std::cout << "Common level was " << common_ref << " now set to " << usr_max_level << "\n";
    common_ref = usr_max_level;
  }

  if (usr_min_level > 0)
  {
    if (common_ref > usr_min_level)
    {
      common_ref = usr_min_level;
    }
  }

  // Find parent blocks at common refinement level
  for (size_t ri = 0; ri < max_ref - common_ref; ++ri) {
    for (size_t i = 0; i < var.global_num_blocks; ++i) {
      if (!in_reg[i]) continue;
      if (grid.refine_level[i] > common_ref) {
        in_reg[i] = false;
        in_reg[grid.gid[i].parent - 1] = true;
      }
    }
  }

  // Enforce maximum ref level to not be higher than user limit
  if (max_ref > usr_max_level) {
    std::cout << "Max level was " << max_ref << " now set to " << usr_max_level << "\n";
    max_ref = usr_max_level;
  }

  if (max_ref < usr_max_level) {
    std::cout << "Max level was " << max_ref << " now set to " << usr_max_level << "\n";
    max_ref = usr_max_level;
  }

  // Replacement parent blocks by children
  for (size_t ri = 0; ri < max_ref - common_ref; ++ri) {
    for (size_t i = 0; i < var.global_num_blocks; ++i) {
      if (!in_reg[i]) continue;
      if (grid.refine_level[i] >= max_ref) continue;
      if (grid.gid[i].children[0] != -1) {
        in_reg[i] = false;
        for (size_t ci = 0; ci < 8; ++ci){
          in_reg[grid.gid[i].children[ci] - 1] = true;
        }
      }
    }
  }

  // Get first and last block in list
  // Gives lower and upper corner on fully sampled cuboid
  bool find_first = true;
  for (size_t i = 0; i < var.global_num_blocks; ++i) {
    if (!in_reg[i]) continue;
    bid_last = i;
    if (find_first) {
      bid_first = i;
      find_first = false;
    }
  }

  // Store domain bounds lower, upper corner
  result.domainBounds[0] = grid.bnd_box[bid_first].min.x;
  result.domainBounds[1] = grid.bnd_box[bid_first].min.y;
  result.domainBounds[2] = grid.bnd_box[bid_first].min.z;
  result.domainBounds[3] = grid.bnd_box[bid_last].max.x;
  result.domainBounds[4] = grid.bnd_box[bid_last].max.y;
  result.domainBounds[5] = grid.bnd_box[bid_last].max.z;

  // Length of the sides of the bounding box
  double len_total[3] = {
    // var.global_num_blocks-1
    grid.bnd_box[bid_last].max.x - grid.bnd_box[bid_first].min.x,
    grid.bnd_box[bid_last].max.y - grid.bnd_box[bid_first].min.y,
    grid.bnd_box[bid_last].max.z - grid.bnd_box[bid_first].min.z};

  std::cout << "Size of simulation domain:\n";
  std::cout << "x\t\ty\t\tz\n";
  for (size_t i = 0; i < 3; ++i) {
    std::cout << len_total[i] << '\t';
  }
  std::cout << '\n';

  // Maximum refinement level
  int max_level = 0;
  // Smallest block size
  double len[3];
  // Loop over all blocks and update maximum refinement level
  // and determine corresponding block size
  for (size_t i = 0; i < var.global_num_blocks; ++i) {
    if (!in_reg[i]) continue;
    if (grid.refine_level[i] > max_level) {
      max_level = grid.refine_level[i];
      len[0] = grid.bnd_box[i].max.x - grid.bnd_box[i].min.x;
      len[1] = grid.bnd_box[i].max.y - grid.bnd_box[i].min.y;
      len[2] = grid.bnd_box[i].max.z - grid.bnd_box[i].min.z;
    }
  }
  std::cout << "Max ref level is " << max_level << "\n";

  // --- cellWidth
  // Loop over all refinement levels
  // And append corresponding halfing factor
  for (int l = 0; l <= max_level; ++l) {
    result.cellWidth.push_back(1 << l);
  }

  // Convert smallest block size to smallest cell size
  len[0] /= var.nxb;
  len[1] /= var.nyb;
  len[2] /= var.nzb;

  // This is the number of cells for the finest level (?)
  int vox[3];
  // Get number of cells needed for uniform grid of highest refinement level
  vox[0] = static_cast<int>(round(len_total[0] / len[0]));
  vox[1] = static_cast<int>(round(len_total[1] / len[1]));
  vox[2] = static_cast<int>(round(len_total[2] / len[2]));

  int corr_fac = 1;
  if (max_level < max_ref)
  {
    corr_fac = pow(2.0, max_ref-max_level);
    max_level = max_ref;

  }
  if (max_level > max_ref)
  {
    max_level = max_ref;
  }

  // Store size of domain in counts of cells at highest refinement level
  result.domainSize[0] = vox[0] * corr_fac;
  result.domainSize[1] = vox[1] * corr_fac;
  result.domainSize[2] = vox[2] * corr_fac;




  std::cout << vox[0]*corr_fac << ' ' << vox[1]*corr_fac << ' ' << vox[2]*corr_fac << '\n';

  // Set initial limits of dataset
  float max_scalar = -FLT_MAX;
  float min_scalar = FLT_MAX;

  // Count the number of leaf blocks
  size_t numLeaves = 0;
  for (size_t i = 0; i < var.global_num_blocks; ++i) {
    if (!in_reg[i]) continue;
    if (grid.node_type[i] == 1)
      numLeaves++;
  }
  
  // This is the slowest part
  for (size_t i = 0; i < var.global_num_blocks; ++i) {
    // if (grid.node_type[i] == 1) // leaf!
    if (!in_reg[i]) continue;
    {
      // Project min on vox grid
      // Get local refinement level difference to maximum refinement level
      int level = max_level - grid.refine_level[i];
      // Get difference factor due to difference in refinement level
      int cellsize = 1 << level;
      // Get lower index position in uniform grid
      int lower[3] = {
        static_cast<int>(
          round((grid.bnd_box[i].min.x - grid.bnd_box[bid_first].min.x) / len[0] * corr_fac)),
        static_cast<int>(
          round((grid.bnd_box[i].min.y - grid.bnd_box[bid_first].min.y) / len[1] * corr_fac)),
        static_cast<int>(
          round((grid.bnd_box[i].min.z - grid.bnd_box[bid_first].min.z) / len[2] * corr_fac))};

      // Bounding box in index reference frame assuming
      // uniform grid at lowest refinement
      // ORDER CHANGED
      // from: xmin, xmax, ymin, ymax, zmin, zmax
      // to:   xmin, ymin, zmin, xmax, ymax, zmax
      BlockBounds bounds = {
        {
          lower[0], lower[1], lower[2],
          lower[0] + (int)var.nxb * cellsize - 1,
          lower[1] + (int)var.nyb * cellsize - 1,
          lower[2] + (int)var.nzb * cellsize - 1}};
      
      // Initialise data object 
      BlockData data;
      // Set dimensions of block
      data.dims[0] = var.nxb;
      data.dims[1] = var.nyb;
      data.dims[2] = var.nzb;
      // Loop over all cells
      for (int z = 0; z < var.nzb; ++z) {
        for (int y = 0; y < var.nyb; ++y) {
          for (int x = 0; x < var.nxb; ++x) {
            // Determine index of cell in 1D dataset
            // i * nxb * nyb * nzb -- start of dataset
            // + x                 -- offset for x
            // + y * nxb           -- offset for y
            // + z * nyb * nyb     -- offset for z
            size_t index = i * var.nxb * var.nyb * var.nzb
                + z * var.nyb * var.nxb + y * var.nxb + x;
	          // Get cell data
            double val = var.data[index];
	          // If data is not 0.0 take log10
	          // Should probably connected to switch
            if (usr_set_log) {
              val = val == 0.0 ? 0.0 : log10(val);
            }
	          // Initialise valf = val
            float valf(val);
	          // Check for minimum and maximum of range
            min_scalar = fminf(min_scalar, valf);
            max_scalar = fmaxf(max_scalar, valf);
            // Store data as float in data
            data.values.push_back((float)val);
          }
        }
      }
      // Store level differences (!) in blockLevel
      result.blockLevel.push_back(level);
      // Store block boundaries (index on uniform grid) in blockBounds
      result.blockBounds.push_back(bounds);
      // Store data (log10) in blockData
      result.blockData.push_back(data);
      // Store range of data field (log10)
      result.voxelRange = {min_scalar, max_scalar};
    }
  }

  std::cout << "value range: [" << min_scalar << ',' << max_scalar << "]\n";
  delete[] in_reg;
  return result;
}

struct FlashReader
{
  bool open(const char *fileName)
  {
    if (!H5::H5File::isHdf5(fileName))
      return false;

    try {
      file = H5::H5File(fileName, H5F_ACC_RDONLY);
      // Read simulation info
      sim_info_t sim_info;
      read_sim_info(sim_info, file);

      // Read grid data
      read_grid(grid, file);

      std::cout << "Variables found:\n";
      for (std::size_t i = 0; i < grid.unknown_names.size(); ++i) {
        std::string uname(
            grid.unknown_names[i].data(), grid.unknown_names[i].data() + 4);
        std::cout << uname << '\n';
        fieldNames.push_back(uname);
      }
    } catch (H5::FileIException error) {
      error.printErrorStack();
      return false;
    }

    return true;
  }

  AMRField getField(int index)
  {
    try {
      std::cout << "Reading field \"" << fieldNames[index] << "\"\n";
      read_variable(currentField, file, fieldNames[index].c_str());

      return toAMRField(grid, currentField, cutting_reg, usr_min_level, usr_max_level, usr_set_log);
    } catch (H5::DataSpaceIException error) {
      error.printErrorStack();
      exit(EXIT_FAILURE);
    } catch (H5::DataTypeIException error) {
      error.printErrorStack();
      exit(EXIT_FAILURE);
    }

    return {};
  }

  // Function to determine index of field
  int getFieldIndex(std::string name)
  {
    for (std::size_t i = 0; i < fieldNames.size(); ++i) {
      if (strcmp(fieldNames[i].c_str(), name.c_str()) == 0) {
	      std::cout << name << " found at index " << i << "\n";
	      return i;
      }
    }
    std::cout << "No entry found for " << name << "\n";
    return -1;
  }

  // Function to get field data by fieldName
  AMRField getFieldByName(std::string fieldName)
  {
    try {
      std::cout << "Reading field \"" << fieldName << "\"\n";
      read_variable(currentField, file, fieldName.c_str());

      return toAMRField(grid, currentField, cutting_reg, usr_min_level, usr_max_level, usr_set_log);
    } catch (H5::DataSpaceIException error) {
      error.printErrorStack();
      exit(EXIT_FAILURE);
    } catch (H5::DataTypeIException error) {
      error.printErrorStack();
      exit(EXIT_FAILURE);
    }

    return {};
  }

  ParticleField getSinkList()
  {
    try {
      read_sinks(particle, file);
      return toParticleField(particle);

    } catch (H5::DataSpaceIException error) {
      error.printErrorStack();
      exit(EXIT_FAILURE);
    } catch (H5::DataTypeIException error) {
      error.printErrorStack();
      exit(EXIT_FAILURE);
    }

    return {};
  }

  // Set region of interest
  // Only blocks falling into this region are considered
  void setROI(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax) {
    cutting_reg[0] = xmin;
    cutting_reg[1] = xmax;
    cutting_reg[2] = ymin;
    cutting_reg[3] = ymax;
    cutting_reg[4] = zmin;
    cutting_reg[5] = zmax;

    std::cout << "Set cutting region:\n";
    std::cout << "\tx\t\ty\t\tz\n";
    std::cout << "min\t" << cutting_reg[0] << "\t\t" << cutting_reg[2] << "\t\t" << cutting_reg[4] << "\n";
    std::cout << "min\t" << cutting_reg[1] << "\t\t" << cutting_reg[3] << "\t\t" << cutting_reg[5] << "\n";
  }

  void setMinLevel(int rlvl_min) {
    usr_min_level = rlvl_min;
  }
  void setMaxLevel(int rlvl_max) {
    usr_max_level = rlvl_max;
  }
  void setLog(bool log) {
    usr_set_log = log;
  }

  // HDF5 file
  H5::H5File file;
  // Derived field names
  std::vector<std::string> fieldNames;
  // Grid structure
  grid_t grid;
  // Particle structure
  particle_t particle;
  // Current field being read
  variable_t currentField;
  // Cutting region
  // Initially covers all domain
  BlockBoundsf cutting_reg = {
    {-FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX}
  };

  // Initial values of user parameters
  // minimum and maximum refinement level of uniform grid
  int usr_min_level = -1;
  int usr_max_level = 1000;
  // flag to enable log space
  bool usr_set_log = false;
};
