#include "readFlash_anari.h"
#include <iostream>

int main() {
  //const char *pathtemplate = "/home/nuernberger/sims/5e8/DwarfGal_hdf5_plt_cnt_0029";
  const char *pathtemplate = "/home/nuernberger/sim_tmp/5e8/DwarfGal_hdf5_plt_cnt_0012";
  char path_fname[1024];

  sprintf(path_fname, "%s", pathtemplate);
  // Init data field
  AMRField data;
  ParticleField particles;
  // Init flashReader
  FlashReader flashReader;
  // Open File
  // flashReader.open(path_fname.c_str());
  flashReader.open(path_fname);
  // If region needs to be selected apply region of interest
  //if (pfSelectRegion->getValue()) {
  //  float xmin, xmax, ymin, ymax, zmin, zmax;
  //  pfRegionMin->getValue(xmin, ymin, zmin);
  //  pfRegionMax->getValue(xmax, ymax, zmax);
  //  flashReader.setROI(
  //  xmin, xmax, ymin, ymax, zmin, zmax
  //);

  flashReader.setMinLevel(-1);
  flashReader.setMaxLevel(3);

  bool use_log = true;
  // Read the data
  flashReader.setLog((bool) use_log);
  data = flashReader.getFieldByName("dens");

  auto &dat = data;
  
  int nr_sink_part;
  for (int i=0; i < flashReader.scalar_parameters.nr_integers; ++i)
  {
    std::string str(flashReader.scalar_parameters.dset_ints[i].name);
    if (str.find("hs_slcs") != std::string::npos)
    {
      nr_sink_part = flashReader.scalar_parameters.dset_ints[i].value;
    }
  }

  particles = flashReader.getSinkList(nr_sink_part);
  auto &part = particles;
  for (size_t i=0; i < 10; ++i)
  {
    std::cout << part.particleMass[i] << "\n";
    //std::cout << dat.domainBounds[i] << "\n";
  }
}

