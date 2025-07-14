/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2025 GER  **
 **                                                                        **
 ** Description: Read volume files in formats supported by Virvo.          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     Pierre Colin Nürnberger                            **
 **           University of Cologne - Institute of Physics I               **
 **                        Zülpicher Straße 77                             **
 **                            50937 Köln                                  **
 **                                                                        **
 ** Cration Date: 28.05.2025                                               **
\**************************************************************************/

#include <sstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <api/coModule.h>
#include <api/coFeedback.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoPoints.h>
#include <do/coDoSpheres.h>
#include <do/coDoSet.h>
//#include <virvo/vvfileio.h>
//#include <virvo/vvvoldesc.h>
//#include <virvo/vvtoolshed.h>
//#include <virvo/fileio/feature.h>
#include "readFlash_anari.h"
#include "ReadFlash.h"


/// Iterate folders either with a sequence (begin, end, int)
/// or randomly (sequence is determined from file header)
class FolderIterator
{
public:

    // Use either an int or a boost::filesystem::directory_iterator
    enum Type { Int, DirectoryIter };


    // constructors ---------------------------------------

    // Type Int
    FolderIterator(
            std::string pt,
            int i,
            bool dc_format = false,
            bool slice_format = false
            )
        : type_(Int)
        , pathtemplate_(pt)
        , int_it_(i)
        , dc_format_(dc_format)
        , slice_format_(slice_format)
    {
    }

    // Type DirectoryIter
    FolderIterator(
            boost::filesystem::path dir,
            bool dc_format = false,
            bool slice_format = false
            )
        : type_(DirectoryIter)
        , dir_it_(dir)
        , dc_format_(dc_format)
        , slice_format_(slice_format)
    {
    }

    // TypeDirectoryIter, create end()
    FolderIterator()
        : type_(DirectoryIter)
    {
    }


    // basic forward iterator interface -------------------

    FolderIterator& operator++()
    {
        if (type_ == Int)
            ++int_it_;
        else
            ++dir_it_;
        return *this;
    }

    bool operator!=(const FolderIterator& rhs)
    {
        if (type_ == Int)
            return int_it_ != rhs.int_it_;
        else
            return dir_it_ != rhs.dir_it_;
    }


    // ----------------------------------------------------

    boost::filesystem::path path() const
    {
        if (type_ == Int)
            return boost::filesystem::path(assemblePath(int_it_));
        else
            return dir_it_->path();
    }

    std::string path_string() const
    {
        if (type_ == Int)
            return assemblePath(int_it_);
        else
            return dir_it_->path().string();
    }

    boost::filesystem::file_status status() const
    {
        if (type_ == Int)
            return boost::filesystem::status(path());
        else
            return dir_it_->status();
    }

    int toInt() const
    {
        if (type_ == Int)
            return int_it_;
        else
            return -1;
    }

private:

    std::string assemblePath(int i) const
    {
        char *path = new char[strlen(pathtemplate_.c_str()) + 20];
        if (dc_format_)
        {
            int i1, i2;
            i2 = i % 1000;
            i1 = i / 1000;
            sprintf(path, pathtemplate_.c_str(), i1, i2);
        }
        else
        {
            sprintf(path, pathtemplate_.c_str(), i);
        }

        std::string result(path);
        delete[] path;
        return result;
    }

    Type type_;
    std::string pathtemplate_;
    bool dc_format_;
    bool slice_format_;

    int int_it_;
    boost::filesystem::directory_iterator dir_it_;

};


/// util class, provides folder iterators
struct FolderUtil
{
    struct Sequence
    {
        Sequence(int b, int e, int i)
            : begin(b)
            , end(e)
            , increment(i)
        {
        }

        int begin;
        int end;
        int increment;
    };


    // constructors ---------------------------------------

    // use integer sequence
    FolderUtil(std::string pathtemplate, Sequence seq)
        : type_(FolderIterator::Int)
        , pathtemplate_(pathtemplate)
        , sequence_(seq)
        , dc_format_(false)
        , slice_format_(false)
    {
        checkFormat();
    }

    // randomly traverse folder, sequence is
    // obtained from file headers
    FolderUtil(std::string path)
        : type_(FolderIterator::DirectoryIter)
        , pathtemplate_(path)
        , sequence_(0, 0, 0)
        , dc_format_(false)
        , slice_format_(true) // always assume we merge slices (TODO?)
    {
        boost::filesystem::path tmp(pathtemplate_);
        directory_ = tmp.parent_path();
    }


    // get iterators --------------------------------------

    FolderIterator begin()
    {
        if (type_ == FolderIterator::Int)
            return FolderIterator(
                pathtemplate_,
                sequence_.begin,
                dc_format_,
                slice_format_
                );
        else
            return FolderIterator(
                directory_,
                dc_format_,
                slice_format_
                );
    }

    FolderIterator end()
    {
        if (type_ == FolderIterator::Int)
            return FolderIterator(pathtemplate_, sequence_.end);
        else
            return FolderIterator();
    }


    // sequence format ------------------------------------

    bool dc_format() const
    {
        return dc_format_;
    }

    bool slice_format() const
    {
        return slice_format_;
    }

private:

    void checkFormat()
    {
        const char *s1 = strchr(pathtemplate_.c_str(), '%');
        if (s1)
            slice_format_ = true;
        if (s1 && strchr(s1 + 1, '%')) // we found a second %
            dc_format_ = true;
    }

    FolderIterator::Type type_;
    std::string pathtemplate_;
    Sequence sequence_;
    bool dc_format_;
    bool slice_format_;
    boost::filesystem::path directory_;
};


/// Constructor
// Creates the module in covise
// Such as ports for input and output as well as possible runtime settings
coReadFlash::coReadFlash(int argc, char *argv[])
    : coModule(argc, argv, "Read Flash files.")
{

  // Create ports:
  poGrid = addOutputPort("grid", "UniformGrid", "Grid for volume data");
  poGrid->setInfo("Grid for volume data");

  // Loop over all channels
  for (int c = 0; c < MAX_CHANNELS; c++)
  {
    // Create name of channel
    char buf[1024];
    sprintf(buf, "channel%d", c);
    // Create description of channel
    char buf2[1024];
    sprintf(buf2, "Scalar volume data channel %d", c);
    // Add port with name, data type and description
    // poVolume declared in header
    poVolume[c] = addOutputPort(buf, "Float|Byte", buf2);
    // Update info
    poVolume[c]->setInfo(buf2);
  }

  // String initialisation
  std::stringstream filetypes;
  // Store possible files types
  filetypes << "*hdf*;*plt*;*chk*/*";
  // Convert to string
  std::string ftstr = filetypes.str();
  // Create pointer to c string
  const char *ftptr = ftstr.c_str();

  // Create parameters:
  // Possible settings in module window
  // Also set initial value by setValue()
  // File selection
  pbrVolumeFile = addFileBrowserParam("FilePath", "FLASH file (or printf format string for sequence)");
  pbrVolumeFile->setValue("data/volume", ftptr);

  piSequence = addInt32VectorParam("Sequence", "First, last and increment in file number", 3);
  piSequence->setValue(0, 0, 1);

  // Particles
  pfGetParticles = addBooleanParam("Particles", "Return particle positions at the last output port");
  pfGetParticles->setValue(false);

  // Removed, now determined using data from the stored scalars
  //pfPartProperties = addInt32Param("NumPartProp", "Number of particle properties (-1 = guessing)");
  //pfPartProperties->setValue(-1);

  // Region selection
  pfSelectRegion = addBooleanParam("useRegion", "Enable a region selection with xmin, xmax...");
  pfSelectRegion->setValue(false);

  pfRegionMin = addFloatVectorParam("regMin", "lower corner of region", 3);
  pfRegionMin->setValue(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  pfRegionMax = addFloatVectorParam("regMax", "upper corner of region", 3);
  pfRegionMax->setValue(FLT_MAX, FLT_MAX, FLT_MAX);

  // Maximum level selection
  // pfMaxLevel = addInt32Param("rlvl_max", "Maximum allowed refinement level");
  // pfMaxLevel->setValue(20);

  pfLevels = addInt32VectorParam("ref_lvls", "Common lower and upper refinement level", 2);
  pfLevels->setValue(0, -1);
  pfLevels->setValue(1, 8);

  // Variables
  for (int i = 0; i < MAX_CHANNELS; ++i)
  {
    char buf[1024];
    sprintf(buf, "%s_%d", "var_name", i);
    var_names[i] = addStringParam(buf, "variable name");
    if (i == 0) {
      var_names[i]->setValue("dens");
    }
    else {
      var_names[i]->setValue("");
    }
    // Custom range
    sprintf(buf, "%s_%d", "Options", i);
    pfDataOpt[i] = addInt32VectorParam(buf, "use log(0/1), use vmin/vmax(0/1), return Float(0/1)", 3);
    pfDataOpt[i]->setValue(0, 0, 0);

    sprintf(buf, "%s_%d", "vmin_vmax", i);
    pfRange[i] = addFloatVectorParam(buf, "limits of value range", 2);
    pfRange[i]->setValue(0, 0.0);
    pfRange[i]->setValue(1, 1.0);
  }
}

/// This is our compute-routine
int coReadFlash::compute(const char *)
{
  // Create a pointer to the pointer of gridData
  coDoUniformGrid **gridData = NULL;
  coDoPoints **particleData = NULL;

  std::vector<AMRField> dataOut;
  std::vector<ParticleField> particleOut;
  std::vector<int> fieldOut;
  
  // Create pointers to volumeSet and gridSet
  coDoSet *volumeSet = NULL;
  coDoSet *particleSet = NULL;
  coDoSet *gridSet = NULL;

  // Path to the file
  const char *pathtemplate = pbrVolumeFile->getValue();
  // Init values for return value and number of files
  int retVal = CONTINUE_PIPELINE;
  int numFiles = 0;
  int numFields = 0;

  // Check if particles should be returned
  bool retPart = pfGetParticles->getValue();

  // Count how man files to process
  int iter = 1;
  long sbegin, send, sinc;
  piSequence->getValue(sbegin, send, sinc);
  if (sinc > 0){
    iter = (send - sbegin) / sinc + 1;
  }

  // Prepare file name to cutoff number at the end
  int len = strlen(pathtemplate);
  string path_temp = pathtemplate;
  path_temp = path_temp.substr(0, len-5);

  // Count number of iterations
  int niter = (send - sbegin) / sinc + 1;

  // Loop over all files
  char path_fname[1024];
  for (int it=sbegin; it < send + sinc; it = it + sinc)
  {
    // Create name of current file
    if (niter > 1)
    {
      sprintf(path_fname, "%s_%04d", path_temp.c_str(), it);
      std::cout << path_fname << "\n";
    }
    else
    {
      sprintf(path_fname, "%s", pathtemplate);
    }
    
    // Init data field
    AMRField data;
    ParticleField particles;
    
    // Init flashReader
    FlashReader flashReader;
    
    // Open File
    // flashReader.open(path_fname.c_str());
    flashReader.open(path_fname);
    
    // If region needs to be selected apply region of interest
    if (pfSelectRegion->getValue()) {
      float xmin, xmax, ymin, ymax, zmin, zmax;
      pfRegionMin->getValue(xmin, ymin, zmin);
      pfRegionMax->getValue(xmax, ymax, zmax);
      flashReader.setROI(
        xmin, xmax, ymin, ymax, zmin, zmax
      );
    
    // Set minimum and maximum refinement level of uniform grid.
    flashReader.setMinLevel(pfLevels->getValue(0));}
    flashReader.setMaxLevel(pfLevels->getValue(1));

    // Check if fields are active by checking string length
    for (int i = 0; i < MAX_CHANNELS; ++i)
    {
      if (strlen(var_names[i]->getValue()) > 0)
      {
        long use_log = pfDataOpt[i]->getValue(0);
      
        // Set flag to use linear or log space for data
        flashReader.setLog((bool) use_log);
      
        // Read the data
        data = flashReader.getFieldByName(var_names[i]->getValue());
      
        // Store the data
        dataOut.push_back(data);
      
        // Check if last channel is already used
        // when particle data is requested
        if (i + 1 == MAX_CHANNELS && retPart)
        {
          std::cout << "Last channel already occupied for a different field\n";
          retPart = false;
        }
      }
    }

    // Store particle if flag is still active
    if (retPart)
    {
      // Read number of particles in list from "integer scalar" field
      int nr_sink_part = -1;
      // Loop over all entries within the field
      for (int i = 0; i < flashReader.scalar_parameters.nr_integers; ++i)
      {
        // Convert string for comparison
        std::string str(flashReader.scalar_parameters.dset_ints[i].name);
        // Check if keyword exist in current string
        if (str.find("hs_slcs") != std::string::npos)
        {
          // Get number of sink entires
          nr_sink_part = flashReader.scalar_parameters.dset_ints[i].value;
        }
      }
      particles = flashReader.getSinkList(nr_sink_part);
      particleOut.push_back(particles);
    }

    // Increase file counter
    numFiles++;
  }

  // Count the active fields and store the index of it
  for (int i = 0; i < MAX_CHANNELS; ++i)
  {
    // Field is active if string exists
    if (strlen(var_names[i]->getValue()) > 0)
    {
      // Keep track of corresponding channel
      fieldOut.push_back(i);
      std::cout << "File out " << i << "\n";

      // Increase field counter
      numFields++;
    }
  }

  // Conversion factor for the coordinates
  // from parsec to cm
  float pc2cm = 3.086e18;
  
  // Determine the grid for each file
  if (numFiles > 0)
  {
    // Create set of uniform grids for each file
    // Requires one more entry than number of files 
    gridData = new coDoUniformGrid *[numFiles+1];
    std::cout << "numFiles " << numFiles << "\n";
   
    // Determine the bounding box of the first dataset
    // All other datasets should have the same grid
    for (int t = 0; t < numFiles; ++t) {

      // Select first dataset of current file
      auto &dat = dataOut[t * numFields];

      // Get bounding box, rescale from cm to pc
      float minX = dat.domainBounds[0] / pc2cm;
      float minY = dat.domainBounds[1] / pc2cm;
      float minZ = dat.domainBounds[2] / pc2cm;
      float maxX = dat.domainBounds[3] / pc2cm;
      float maxY = dat.domainBounds[4] / pc2cm;
      float maxZ = dat.domainBounds[5] / pc2cm;

      // Get shape of grid
      int npixx = dat.domainSize[0];
      int npixy = dat.domainSize[1];
      int npixz = dat.domainSize[2];
      std::cout << "Pixel? " << npixx << " " << npixy << " " << npixz << "\n";
      std::cout << "  min [pc] " << minX << " " << minY << " " << minZ << "\n";
      std::cout << "  max [pc] " << maxX << " " << maxY << " " << maxZ << "\n";

      // If more than 1 file is read, give timeseries
      if (numFiles > 1)
      {
        char buf[1024];
        sprintf(buf, "%s_%d", poGrid->getObjName(), t);
        gridData[t] = new coDoUniformGrid(buf, npixx, npixy, npixz, minX, maxX, minY, maxY, minZ, maxZ);
      }
      else
      {
        gridData[t] = new coDoUniformGrid(poGrid->getObjName(), npixx, npixy, npixz, minX, maxX, minY, maxY, minZ, maxZ);
      }
    }

    // Set last entry to NULL (?)
    // Tells end of timeseries 
    gridData[numFiles] = NULL;
    coFeedback browserFeedback("FileBrowserParam");
    browserFeedback.addPara(pbrVolumeFile);

    // If more than 1 file is read, make a set
    // and apply to output
    // Else just apply the current grid
    if (numFiles > 1)
    {
      char buf[1024];
      sprintf(buf, "0 %d", (int)numFiles - 1);

      gridSet = new coDoSet(poGrid->getObjName(), (coDistributedObject **)gridData);
      gridSet->addAttribute("TIMESTEP", buf);
      
      browserFeedback.apply(gridSet);
      poGrid->setCurrentObject(gridSet);

      for (size_t t = 0; t < numFiles; ++t)
      {
        delete gridData[t];
      }
    }
    else
    {
      browserFeedback.apply(gridData[0]);
      poGrid->setCurrentObject(gridData[0]);
      delete[] gridData;
    }

    // Loop over all channels
    for (int c = 0; c < numFields; ++c)
    {
      // Create output object for the data
      std::vector<coDistributedObject *> timesteps;

      // Initialise user defined value range
      long use_log = pfDataOpt[fieldOut[c]]->getValue(0);
      long use_range = pfDataOpt[fieldOut[c]]->getValue(1);
      long ret_float = pfDataOpt[fieldOut[c]]->getValue(2);
      
      // Limits to use by default
      float use_vmin = 0;
      float use_vmax = 1;
      
      // Check if user specified range needs to be applied
      if ((bool) use_range)
      {
        use_vmin = pfRange[fieldOut[c]]->getValue(0);
        use_vmax = pfRange[fieldOut[c]]->getValue(1);
        
        // Check if log is active
        // in that case automatically update values to log values
        if((bool) use_log)
        {
          use_vmin = log10(use_vmin);
          use_vmax = log10(use_vmax);
        }
      }

      // Loop over all files (times)
      for (int t = 0; t < numFiles; ++t)
      {
        // Get dataset of current channel (dataset)
        auto &dat = dataOut[numFields * t + c];

        // Get bounds of dataset
        // May be time-dependent
        float minX = dat.domainBounds[0];
        float minY = dat.domainBounds[1];
        float minZ = dat.domainBounds[2];
        float maxX = dat.domainBounds[3];
        float maxY = dat.domainBounds[4];
        float maxZ = dat.domainBounds[5];
        
        // Get size of uniform grid
        // May be time-dependent
        int npixx = dat.domainSize[0];
        int npixy = dat.domainSize[1];
        int npixz = dat.domainSize[2];
        
        // Get object name for covise
        std::stringstream name(poVolume[fieldOut[c]]->getObjName());
        // For time series append id of current "time"
        if (numFiles > 1) name << "_" << t;

        // Get total size of array
        unsigned long long int size_arr = npixx * npixy * npixz;

        // Initialise float and byte data address
        coDoFloat *dof = NULL;
        float *fdata = NULL;
        coDoByte *dob = NULL;
        unsigned char *bdata = NULL;
        
        // Allocate new float/byte array 
        // get start address of dataset
        // and append to timeseries
        if ((bool) ret_float)
        {
          dof = new coDoFloat(name.str(), int(npixx * npixy * npixz));
          fdata = dof->getAddress();
          timesteps.push_back(dof);
        }
        else
        {
          dob = new coDoByte(name.str(), size_arr);
          bdata = dob->getAddress();
          timesteps.push_back(dob);
        }

        
        int cnt = 0;
        
        // Get dataset value limits
        float vmin = dat.voxelRange.x;
        float vmax = dat.voxelRange.y;
        
        // Update to user defined values if set
        if ((bool) use_range)
        {
          vmin = use_vmin;
          vmax = use_vmax;
        }
        
        // Get total range of data values
        // Needed to create linear space for byte data
        float range = vmax - vmin;
        
        // Not sure yet what this does
        std::string mapping_min_str = boost::lexical_cast<std::string>(0);
        std::string mapping_max_str = boost::lexical_cast<std::string>(1);
        std::string range_min_str = boost::lexical_cast<std::string>(0);
        std::string range_max_str = boost::lexical_cast<std::string>(1);

        timesteps.back()->addAttribute("MAPPING_MIN", mapping_min_str.c_str());
        timesteps.back()->addAttribute("MAPPING_MAX", mapping_max_str.c_str());
        timesteps.back()->addAttribute("RANGE_MIN", range_min_str.c_str());
        timesteps.back()->addAttribute("RANGE_MAX", range_max_str.c_str());
        
        // Map data onto uniform grid
        // Loop over all blocks in dataset
        for (int i = 0; i < dat.blockData.size(); ++i)
        {
          // Get shape of current block
          int ncx = dat.blockData[i].dims[0];
          int ncy = dat.blockData[i].dims[1];
          int ncz = dat.blockData[i].dims[2];

          // Get level difference to highest refined block
          int lvl_diff = 1 << dat.blockLevel[i];
          
          // Get lower and upper corner of block
          // in cell space (not physical space)
          BlockBounds bnds = dat.blockBounds[i];

          // Loop over all cells in block
          for (int kk = 0; kk < ncz; ++kk)
          {
            for (int jj = 0; jj < ncy; ++jj)
            {
              for (int ii = 0; ii < ncx; ++ii)
              {
                // Get cell position in block
                unsigned long long int blk_ind = ii + ncx * jj + ncy * ncx * kk;
                
                // Data in current cell
                float cdata_val;
                unsigned char cdata_uc;

                // Apply limits onto data set
                if ((bool) ret_float)
                {
                  // Data in current cell
                  cdata_val = dat.blockData[i].values[blk_ind];

                  // Apply limits
                  cdata_val = fmax(cdata_val, vmin);
                  cdata_val = fmin(cdata_val, vmax);
                }
                else
                {
                  // Data in current cell remapped on to [0, 1] space
                  cdata_val = (dat.blockData[i].values[blk_ind] - vmin) / range;

                  // Apply limits
                  cdata_val = fmax(cdata_val, 0.0);
                  cdata_val = fmin(cdata_val, 1.0);
                  
                  // Convert to unsigned char range [0, 255]
                  cdata_uc = cdata_val * 255;
                }
                
                // Loop over all duplicates due to level mismatch
                for (int ok = 0; ok < lvl_diff; ++ok)
                {
                  unsigned long long int dkk = bnds[2] + kk * lvl_diff + ok;
                  for (int oj = 0; oj < lvl_diff; ++oj)
                  {
                    unsigned long long int djj = bnds[1] + jj * lvl_diff + oj;
                    for (int oi = 0; oi < lvl_diff; ++oi)
                    {
                      unsigned long long int dii = bnds[0] + ii * lvl_diff + oi;
                      
                      // Index in output dataset
                      unsigned long long int dof_ind = dkk + npixz * djj + npixz * npixy * dii;
                      
                      // Store data in output
                      if ((bool) ret_float)
                      {
                        fdata[dof_ind] = *(float *) &cdata_val;
                      }
                      else
                      {
                        bdata[dof_ind] = (unsigned char) cdata_uc;
                      }

                      cnt = cnt + 1;
                    } // oi
                  } // oj
                } // ok
              } // ii
            } // jj
          } // kk
        } // block
      } // files

      // Place dataset on the correct output ports
      if (numFiles > 1)
      {
        // Create set objects:
        volumeSet = new coDoSet(poVolume[fieldOut[c]]->getObjName(), int(timesteps.size()), &timesteps[0]);

        // Set timestep attribute:
        char buf[1024];
        sprintf(buf, "0 %d", (int)numFiles - 1);
        volumeSet->addAttribute("TIMESTEP", buf);

        // Assign sets to output ports:
        poVolume[fieldOut[c]]->setCurrentObject(volumeSet);

        // Cleanup?
        for (size_t t = 0; t < numFiles; ++t)
        {
            delete timesteps[t];
        }
      }
      else
      {
        std::cout << "At channel " << c << " output to " << fieldOut[c] << "\n";
        poVolume[fieldOut[c]]->setCurrentObject(timesteps[0]);
      }
    }
    
    // Get particle information
    if (retPart)
    {
      // Timeseries of particle data
      //std::vector<coDistributedObject *> timePart;
      particleData = new coDoPoints *[numFiles+1];
      // Loop over times
      for (size_t t = 0; t < numFiles; ++t)
      {
        auto &dat = particleOut[t];
        int npart = dat.nrpart;

        vector<float> px;
        vector<float> py;
        vector<float> pz;

        // Get object name for covise
        std::stringstream name(poVolume[MAX_CHANNELS-1]->getObjName());
        // For time series append id of current "time"
        if (numFiles > 1) name << "_" << t;

        for (size_t pid = 0; pid < npart; ++pid)
        {
          std::cout << "Particle position\n";
          std::cout << dat.particlePosition[pid].x / pc2cm;
          std::cout << " " << dat.particlePosition[pid].y / pc2cm;
          std::cout << " " << dat.particlePosition[pid].z / pc2cm << "\n";
          px.push_back(dat.particlePosition[pid].x / pc2cm);
          py.push_back(dat.particlePosition[pid].y / pc2cm);
          pz.push_back(dat.particlePosition[pid].z / pc2cm);
        }

        float *ppx = &px[0];
        float *ppy = &py[0];
        float *ppz = &pz[0];

        // If more than 1 file is read, give timeseries
        if (numFiles > 1)
        {
          char buf[1024];
          sprintf(buf, "%s_%d", poVolume[MAX_CHANNELS-1]->getObjName(), (int) t);
          particleData[t] = new coDoPoints(buf, npart, ppx, ppy, ppz);
        }
        else
        {
          particleData[t] = new coDoPoints(poVolume[MAX_CHANNELS-1]->getObjName(), npart, ppx, ppy, ppz);
        }

        // coDoPoints *new_part = new coDoPoints(name.str(), npart, ppx, ppy, ppz);
        // particleData[t] = new coDoPoints(name.str(), npart, ppx, ppy, ppz);
        // timePart.push_back(new_part);
      }

      particleData[numFiles] = NULL;

      // If more than 1 file is read, make a set
      // and apply to output
      // Else just apply the current particles
      if (numFiles > 1)
      {
        char buf[1024];
        sprintf(buf, "0 %d", (int)numFiles - 1);

        particleSet = new coDoSet(poVolume[MAX_CHANNELS-1]->getObjName(), (coDistributedObject **)particleData);
        particleSet->addAttribute("TIMESTEP", buf);
        
        browserFeedback.apply(particleSet);
        poVolume[MAX_CHANNELS-1]->setCurrentObject(particleSet);

        for (size_t t = 0; t < numFiles; ++t)
        {
            delete particleData[t];
        }
      }
      else
      {
        browserFeedback.apply(particleData[0]);
        poVolume[MAX_CHANNELS-1]->setCurrentObject(particleData[0]);
        delete[] particleData;
      }
    }
    
  }

  return retVal;

}

MODULE_MAIN(IO, coReadFlash)
