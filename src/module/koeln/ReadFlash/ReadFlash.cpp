/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Read volume files in formats supported by Virvo.          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                     Juergen Schulze-Doebold                            **
 **     High Performance Computing Center University of Stuttgart          **
 **                         Allmandring 30                                 **
 **                         70550 Stuttgart                                **
 **                                                                        **
 ** Cration Date: 28.10.2000                                               **
\**************************************************************************/

#include <sstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <api/coModule.h>
#include <api/coFeedback.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
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
  pbrVolumeFile = addFileBrowserParam("FilePath", "Volume file (or printf format string for sequence)");
  pbrVolumeFile->setValue("data/volume", ftptr);

  piSequenceBegin = addInt32Param("SequenceBegin", "First file number in sequence");
  piSequenceBegin->setValue(0);

  piSequenceEnd = addInt32Param("SequenceEnd", "Last file number in sequence");
  piSequenceEnd->setValue(0);

  piSequenceInc = addInt32Param("SequenceInc", "Increment in sequence");
  piSequenceInc->setValue(1);

  // Region selection
  pfSelectRegion = addBooleanParam("useRegion", "Enable a region selection with xmin, xmax...");
  pfSelectRegion->setValue(false);

  pfXmin = addFloatParam("xmin", "lower corner x-coordinate");
  pfXmin->setValue(-FLT_MAX);

  pfXmax = addFloatParam("xmax", "upper corner x-coordinate");
  pfXmax->setValue(FLT_MAX);

  pfYmin = addFloatParam("ymin", "lower corner y-coordinate");
  pfYmin->setValue(-FLT_MAX);

  pfYmax = addFloatParam("ymax", "upper corner y-coordinate");
  pfYmax->setValue(FLT_MAX);

  pfZmin = addFloatParam("zmin", "lower corner z-coordinate");
  pfZmin->setValue(-FLT_MAX);

  pfZmax = addFloatParam("zmax", "upper corner z-coordinate");
  pfZmax->setValue(FLT_MAX);

  // Variables
  //var_name = addStringParam("var_name", "variable name");
  //var_name->setValue("dens");
  //std::cout << var_name_0->getValue() << " " << strlen(var_name_0->getValue()) << "\n";
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
  }

  // Custom range
  pfSelectRange = addBooleanParam("useRange", "Enable a range selection with vmin, vmax");
  pfSelectRange->setValue(false);
  
  pfSelectClipping = addBooleanParam("useClipping", "Enables clipping of values outside vmin, vmax");
  pfSelectClipping->setValue(false);

  pfRangeLow = addFloatParam("vmin", "lower limit of range");
  pfRangeLow->setValue(0.0);
  
  pfRangeLow = addFloatParam("vmax", "upper limit of range");
  pfRangeLow->setValue(1.0);
}

/// This is our compute-routine
int coReadFlash::compute(const char *)
{

  // Create a pointer to the pointer of gridData
  coDoUniformGrid **gridData = NULL;
  std::vector<AMRField> dataOut;
  std::vector<int> fieldOut;
  
  // Create pointers to volumeSet and gridSet
  coDoSet *volumeSet = NULL;
  coDoSet *gridSet = NULL;

  // Path to the file
  const char *pathtemplate = pbrVolumeFile->getValue();
  // Init values for return value and number of files
  int retVal = CONTINUE_PIPELINE;
  int numFiles = 0;
  int numFields = 0;

  // Create folder sequence
  FolderUtil folder_util = FolderUtil(
    pathtemplate,
    FolderUtil::Sequence(
      piSequenceBegin->getValue(),
      piSequenceEnd->getValue() + piSequenceInc->getValue(),
      piSequenceInc->getValue()
    )
  );

  // Loop over all items in folder_util
  for (
    FolderIterator it = folder_util.begin();
    it != folder_util.end();
    ++it
    ) {
    // Change namespace (more like a shorthand)
    namespace fs = boost::filesystem;
    // Get path of current file
    std::string path_fname = it.path_string();
    
    // Init data field
    AMRField data;
    
    // Init flashReader
    FlashReader flashReader;
    // Open File
    flashReader.open(path_fname.c_str());
    // If region needs to be selected apply region of interest
    if (pfSelectRegion->getValue()) {
      flashReader.setROI(
        pfXmin->getValue(), pfXmax->getValue(),
        pfYmin->getValue(), pfYmax->getValue(),
        pfZmin->getValue(), pfZmax->getValue()
      );
    }
    
    for (int i = 0; i < MAX_CHANNELS; ++i) {
      if (strlen(var_names[i]->getValue()) > 0) {
        // Read the data
        data = flashReader.getFieldByName(var_names[i]->getValue());
        // Store the data
        dataOut.push_back(data);
      }
    }

    // Increase file counter
    numFiles++;
  }

  // Count the active fields and store the index of it
  for (int i = 0; i < MAX_CHANNELS; ++i) {
    if (strlen(var_names[i]->getValue()) > 0) {
      // Keep track of corresponding channel
      fieldOut.push_back(i);
      std::cout << "File out " << i << "\n";
      // Increase field counter
      numFields++;
    }
  }

  float pc2cm = 3.086e18;
  // Determine the grid for each file
  if (numFiles > 0) {
    gridData = new coDoUniformGrid *[numFiles+1];
    std::cout << "numFiles " << numFiles << "\n";
    // Determine the bounding box of the first dataset
    // All other datasets should have the same grid
    for (int t = 0; t < numFiles; ++t) {
      // Select first dataset of current file
      auto &dat = dataOut[t*numFiles];
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
      std::cout << "min " << minX << " " << minY << " " << minZ << "\n";
      std::cout << "max " << maxX << " " << maxY << " " << maxZ << "\n";

      // If more than 1 file is read give timeseries
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
    gridData[numFiles] = NULL;
    coFeedback browserFeedback("FileBrowserParam");
    browserFeedback.addPara(pbrVolumeFile);

    // If more than 1 file is read, make a set
    // and apply to output
    // Else just apply the current grid
    if (numFiles > 1) {
      char buf[1024];
      sprintf(buf, "0 %d", (int)numFiles - 1);
      gridSet = new coDoSet(poGrid->getObjName(), (coDistributedObject **)gridData);
      gridSet->addAttribute("TIMESTEP", buf);
      browserFeedback.apply(gridSet);
      poGrid->setCurrentObject(gridSet);

      for (size_t t = 0; t < numFiles; t++)
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

    for (int c = 0; c < numFields; ++c) {
      // Create output object for the data
      std::vector<coDistributedObject *> timesteps;

      for (int t = 0; t < numFiles; ++t)
      {
        //std::cout << "t = " << t << "\n";
        auto &dat = dataOut[numFiles * t + c];

        float minX = dat.domainBounds[0];
        float minY = dat.domainBounds[1];
        float minZ = dat.domainBounds[2];
        float maxX = dat.domainBounds[3];
        float maxY = dat.domainBounds[4];
        float maxZ = dat.domainBounds[5];
        
        int npixx = dat.domainSize[0];
        int npixy = dat.domainSize[1];
        int npixz = dat.domainSize[2];
        
        
        std::stringstream name(poVolume[fieldOut[c]]->getObjName());
        if (numFiles > 1) name << "_" << t;

        //coDoByte *dob = NULL;
        coDoFloat *dof = NULL;
        float *fdata = NULL;
        //uchar *bdata = NULL;

        dof = new coDoFloat(name.str(), int(npixx * npixy * npixz));
        
        size_t size_arr = npixx * npixy * npixz;

        //float * dat_1d = new float[size_arr];
        
        int cnt = 0;
        fdata = dof->getAddress();
        timesteps.push_back(dof);
        float range = dat.voxelRange.y - dat.voxelRange.x;
        
        std::string mapping_min_str = boost::lexical_cast<std::string>(0);
        std::string mapping_max_str = boost::lexical_cast<std::string>(1);
        std::string range_min_str = boost::lexical_cast<std::string>(0);
        std::string range_max_str = boost::lexical_cast<std::string>(1);
        timesteps.back()->addAttribute("MAPPING_MIN", mapping_min_str.c_str());
        timesteps.back()->addAttribute("MAPPING_MAX", mapping_max_str.c_str());
        timesteps.back()->addAttribute("RANGE_MIN", range_min_str.c_str());
        timesteps.back()->addAttribute("RANGE_MAX", range_max_str.c_str());

        for (int i = 0; i < dat.blockData.size(); ++i) {
          int ncx = dat.blockData[i].dims[0];
          int ncy = dat.blockData[i].dims[1];
          int ncz = dat.blockData[i].dims[2];
          int lvl_diff = 1 << dat.blockLevel[i];
          
          BlockBounds bnds = dat.blockBounds[i];
          // Loop over all cells in block
          for (int kk = 0; kk < ncx; ++kk) {
            for (int jj = 0; jj < ncy; ++jj) {
              for (int ii = 0; ii < ncz; ++ii) {
                // Cell position in block
                size_t blk_ind = ii + ncx * jj + ncy * ncx * kk;
                // Data in current cell
                float cdata_val = (dat.blockData[i].values[blk_ind] - dat.voxelRange.x) / range;

                // Loop over all duplicates due to level mismatch
                for (int ok = 0; ok < lvl_diff; ++ok) {
                  size_t dkk = bnds[2] + kk * lvl_diff + ok;
                  for (int oj = 0; oj < lvl_diff; ++oj) {
                    size_t djj = bnds[1] + jj * lvl_diff + oj;
                    for (int oi = 0; oi < lvl_diff; ++oi) {
                      size_t dii = bnds[0] + ii * lvl_diff + oi;
                      // Index in output
                      size_t dof_ind = dkk + npixz * djj + npixz * npixy * dii;
                      
                      // Add data to output
                      fdata[dof_ind] = *(float *) &cdata_val;
                      //dat_1d[dof_ind] = cdata_val;
                      cnt = cnt + 1;
                    }
                  }
                }
              }
            }
          }
        }


        //std::cout << npixx * npixy * npixz << " " << cnt <<"\n";
        // Create set objects:
        //volumeSet = new coDoSet(poVolume[t]->getObjName(), int(timesteps.size()), &timesteps[0]);
        //std::cout << "Hello2\n";
        // Set timestep attribute:
        //char buf[1024];
        //sprintf(buf, "%d %d", 0, (int)1 - 1);
        //volumeSet->addAttribute("TIMESTEP", buf);
        //std::cout << "Hello3\n";
        // Assign sets to output ports:
        //poVolume[t]->setCurrentObject(volumeSet);
        //std::cout << "Hello4\n";
      }
      std::cout << "At channel " << c << " output to " << fieldOut[c] << "\n";
      poVolume[fieldOut[c]]->setCurrentObject(timesteps[0]);
    }
  }

  return retVal;

//    // Creata a pointer to the pointer of gridData
//    coDoUniformGrid **gridData = NULL;
//
//    // Create pointers to volumeSet and gridSet
//    coDoSet *volumeSet = NULL;
//    coDoSet *gridSet = NULL;
//
//    // Create pointer to rawData
//    unsigned char *rawData = NULL;
//    // Shape of volume
//    float width = 0.0, height = 0.0, depth = 0.0;
//    // Iteration variable for time
//    int t;
//    // Number of voxels (total)
//    size_t numVoxels;
//    // Path to the file
//    const char *pathtemplate = pbrVolumeFile->getValue();
//
//    // Create pointer to new File loader from VIRVO
//    // Will need to be changed to new custom loader with HDF5 support
//    vvFileIO *fio = new vvFileIO();
//    // Create pointer to new Volume description
//    // Contains volume information
//    vvVolDesc *vd = new vvVolDesc();
//
//    // Create new string
//    std::ostringstream skipped;
//    // Write to skipped
//    skipped << "Skipped";
//
//    // Init values for return value and number of files
//    int retVal = CONTINUE_PIPELINE;
//    int numFiles = 0;
//
//    // Create folder sequence from Header
//    // If True use only path
//    // Otherwise include Sequence
//    FolderUtil folder_util = pboSequenceFromHeader->getValue()
//        ? FolderUtil(pathtemplate)
//        : FolderUtil(pathtemplate, FolderUtil::Sequence(
//                piSequenceBegin->getValue(),
//                piSequenceEnd->getValue() + piSequenceInc->getValue(),
//                piSequenceInc->getValue()
//                )
//                )
//        ;
//    // Loop over folders from sequence
//    for (FolderIterator it =  folder_util.begin();
//                        it != folder_util.end();
//                        ++it
//                        )
//    {
//        // Change namespace
//        namespace fs = boost::filesystem;
//        // Check if it is a directory
//        // Does not do anything
//        if (fs::is_directory(it.status()))
//        {
//            continue;
//        }
//        // Create pointer to volume description
//        // and volume description of current frame
//        vvVolDesc *vdread = vd, vdframe;
//        // Set merge to false
//        bool merge = false;
//        // If third dimensions is larger than 1 and
//        // Volume Description has more than 1 frame
//        if (vd->vox[2] > 1 && vd->frames > 0)
//        {
//            // Set merge to True
//            merge = true;
//            // Update vdread pointer from vd to vdframe
//            vdread = &vdframe;
//        }
//        // Get path of current file
//        std::string path = it.path_string();
//        // Set path for current volume description
//        vdread->setFilename(path.c_str());
//        // Set results to OK
//        int result = vvFileIO::OK;
//        // Check if we read raw data
//        if (pboReadRaw->getValue())
//        {
//            // Load raw data into results
//            // Requires correct information to be set in module
//            // for number of pixels in x, y and z
//            // for bits per channel
//            // for number of channels
//            // for header size (skip)
//            result = fio->loadRawFile(vdread,
//                                      piVoxelsX->getValue(), piVoxelsY->getValue(), piVoxelsZ->getValue(),
//                                      piBPC->getValue(), piChannels->getValue(), piHeader->getValue());
//        }
//        else
//        {
//            // Load volume into results
//            // Add frame is set if not first volume and merge is False (?)
//            result = fio->loadVolumeData(vdread, vvFileIO::ALL_DATA, it != folder_util.begin() && !merge);
//        }
//        // If result is OK
//        if (result == vvFileIO::OK)
//        {
//            // Increase counter for number of files
//            numFiles++;
//            //vdread->printVolumeInfo();
//
//            // Determine volume dimensions:
//            // If custom size is given
//            // Get volume dimensions
//            if (pboCustomSize->getValue())
//            {
//                width = pfsVolWidth->getValue();
//                height = pfsVolHeight->getValue();
//                depth = pfsVolDepth->getValue();
//            }
//            // Otherwise derive from volume
//            else
//            {
//                // Width = number of voxels in x * size of voxel in x
//                width = vdread->vox[0] * vdread->getDist()[0];
//                // Height = number of voxels in y * size of voxel in y
//                height = vdread->vox[1] * vdread->getDist()[1];
//
//                // If slice format and merge is False
//                if (folder_util.slice_format() && !merge)
//                {
//                    // Depth = number of fules * size of voxel in z direction
//                    depth = numFiles * vdread->getDist()[2];
//                }
//                else
//                {
//                    // Depth = number of voxels in z * size of voxel in z
//                    depth = vdread->vox[2] * vdread->getDist()[2];
//                }
//            }
//        }
//        // If file is not found, write to string and skip
//        else if (result == vvFileIO::FILE_NOT_FOUND)
//        {
//            skipped << " " << it.toInt();
//        }
//        // Otherwise stop pipeline
//        else
//        {
//            sendError("Failed to load file %d from sequence: %s", it.toInt(), path.c_str());
//            retVal = STOP_PIPELINE;
//            break;
//        }
//
//        // If merge is True
//        if (merge)
//        {
//            // Add frame to to volume description
//            vd->merge(&vdframe, vvVolDesc::VV_MERGE_VOL2ANIM);
//        }
//    }
//
//    if (numFiles > 0)
//    {
//        // If slices were read merge frames
//        if (vd->vox[2] == 1 && vd->frames > 0)
//        {
//            vd->mergeFrames(); // merge single slices into one volume
//        }
//        // Print information from volume
//        vd->printVolumeInfo();
//        // convert data from Virov formation to Covise formaton
//        vd->convertVirvoToCovise();
//
//        // Create set elements:
//        // important: allocate one more than frames are present!
//        gridData = new coDoUniformGrid *[vd->frames + 1];
//
//        // Create time steps:
//        for (t = 0; t < vd->frames; ++t)
//        {
//            float posX = vd->pos[0] * vd->getDist()[0];
//            float posY = vd->pos[1] * vd->getDist()[1];
//            float posZ = vd->pos[2] * vd->getDist()[2];
//            float maxX = posX + 0.5f * width;
//            float maxY = posY + 0.5f * height;
//            float maxZ = posZ + 0.5f * depth;
//            float minX = posX - 0.5f * width;
//            float minY = posY - 0.5f * height;
//            float minZ = posZ - 0.5f * depth;
//            int vox[] = { static_cast<int>(vd->vox[0]), static_cast<int>(vd->vox[1]), static_cast<int>(vd->vox[2]) };
//            if (vd->frames > 1)
//            {
//                char buf[1024];
//                sprintf(buf, "%s_%d", poGrid->getObjName(), t);
//                gridData[t] = new coDoUniformGrid(buf, vox[0], vox[1], vox[2], minX, maxX, minY, maxY, minZ, maxZ);
//            }
//            else
//            {
//                gridData[t] = new coDoUniformGrid(poGrid->getObjName(), vox[0], vox[1], vox[2], minX, maxX, minY, maxY, minZ, maxZ);
//            }
//        }
//        gridData[vd->frames] = NULL;
//
//        coFeedback browserFeedback("FileBrowserParam");
//        browserFeedback.addPara(pbrVolumeFile);
//
//        if (vd->frames > 1)
//        {
//            char buf[1024];
//            sprintf(buf, "0 %d", (int)vd->frames - 1);
//            gridSet = new coDoSet(poGrid->getObjName(), (coDistributedObject **)gridData);
//            gridSet->addAttribute("TIMESTEP", buf);
//            browserFeedback.apply(gridSet);
//            poGrid->setCurrentObject(gridSet);
//
//            for (t = 0; t < vd->frames; t++)
//            {
//                delete gridData[t];
//            }
//        }
//        else if (vd->vox[0] * vd->vox[1] * vd->vox[2] > 0)
//        {
//            browserFeedback.apply(gridData[0]);
//            poGrid->setCurrentObject(gridData[0]);
//
//            delete[] gridData;
//
//            // Copy raw volume data to shared memory:
//            numVoxels = vd->getFrameVoxels();
//
//            for (int c = 0; c < vd->getChan() && c < MAX_CHANNELS; c++)
//            {
//                std::vector<coDistributedObject *> timesteps;
//                for (t = 0; t < vd->frames; ++t)
//                {
//                    std::stringstream name(poVolume[c]->getObjName());
//                    if (vd->frames > 1)
//                        name << "_" << t;
//                    coDoByte *dob = NULL;
//                    coDoFloat *dof = NULL;
//                    float *fdata = NULL;
//                    uchar *bdata = NULL;
//                    if (vd->bpc == 1 && pboPreferByteData->getValue())
//                    {
//                        dob = new coDoByte(name.str(), int(vd->vox[0] * vd->vox[1] * vd->vox[2]));
//                        bdata = dob->getAddress();
//                        timesteps.push_back(dob);
//                    }
//                    else
//                    {
//                        dof = new coDoFloat(name.str(), int(vd->vox[0] * vd->vox[1] * vd->vox[2]));
//                        fdata = dof->getAddress();
//                        timesteps.push_back(dof);
//                    }
//
//                    std::string mapping_min_str = boost::lexical_cast<std::string>(vd->mapping(c)[0]);
//                    std::string mapping_max_str = boost::lexical_cast<std::string>(vd->mapping(c)[1]);
//                    std::string range_min_str = boost::lexical_cast<std::string>(vd->mapping(c)[0]);
//                    std::string range_max_str = boost::lexical_cast<std::string>(vd->mapping(c)[1]);
//                    timesteps.back()->addAttribute("MAPPING_MIN", mapping_min_str.c_str());
//                    timesteps.back()->addAttribute("MAPPING_MAX", mapping_max_str.c_str());
//                    timesteps.back()->addAttribute("RANGE_MIN", range_min_str.c_str());
//                    timesteps.back()->addAttribute("RANGE_MAX", range_max_str.c_str());
//
//                    rawData = vd->getRaw((size_t)t);
//
//                    bool bs = pboReadBS->getValue();
//                    float minV = (float)minValue->getValue();
//                    float range = (float)(maxValue->getValue() - minValue->getValue());
//                    {
//                        if (vd->bpc == 1)
//                        {
//                            if (fdata)
//                            {
//                                for (size_t i = 0; i < numVoxels; ++i)
//                                    fdata[i] = float(rawData[i * vd->getChan() + c]) / 255.0f;
//                            }
//                            else
//                            {
//                                for (size_t i = 0; i < numVoxels; ++i)
//                                    bdata[i] = rawData[i * vd->getChan() + c];
//                            }
//                        }
//                        else if (vd->bpc == 2)
//                        {
//                            if (bs) //   Big Endian
//                            {
//                                //low
//                                for (size_t i = 0; i < numVoxels; ++i)
//                                {
//                                    fdata[i] = (((256.0f * ((float)rawData[(vd->getChan() * i + c) * 2]))
//                                                 + ((float)rawData[(vd->getChan() * i + c) * 2 + 1]))
//                                                - minV) / range;
//                                    fdata[i] = ts_clamp(fdata[i], 0.f, 1.f);
//                                }
//                            }
//                            else //    Little Endian
//                            {
//                                for (size_t i = 0; i < numVoxels; ++i)
//                                {
//                                    //high
//                                    fdata[i] = (((256.0f * ((float)rawData[(vd->getChan() * i + c) * 2 + 1]))
//                                                 + ((float)rawData[(vd->getChan() * i + c) * 2]))
//                                                - minV) / range;
//                                    fdata[i] = ts_clamp(fdata[i], 0.f, 1.f);
//                                }
//                            } //
//                        }
//                        else if (vd->bpc == 4)
//                        {
//                            if (bs)
//                            {
//                                for (size_t i = 0; i < numVoxels; ++i)
//                                {
//                                    uint32_t d = *(uint32_t *)&rawData[(vd->getChan() * i + c) * 4];
//                                    byteSwap(d);
//                                    fdata[i] = *(float *)&d;
//                                }
//                            }
//                            else
//                            {
//                                for (size_t i = 0; i < numVoxels; ++i)
//                                {
//                                    fdata[i] = *(float *)&rawData[(vd->getChan() * i + c) * 4];
//                                }
//                            }
//                        }
//                        else
//                        {
//                            for (size_t i = 0; i < numVoxels; ++i)
//                                fdata[i] = 0.0f;
//                        }
//                    }
//                }
//
//                if (vd->frames > 1)
//                {
//                    // Create set objects:
//                    volumeSet = new coDoSet(poVolume[c]->getObjName(), int(timesteps.size()), &timesteps[0]);
//
//                    // Set timestep attribute:
//                    char buf[1024];
//                    sprintf(buf, "%d %d", 0, (int)vd->frames - 1);
//                    volumeSet->addAttribute("TIMESTEP", buf);
//
//                    // Assign sets to output ports:
//                    poVolume[c]->setCurrentObject(volumeSet);
//
//                    for (t = 0; t < vd->frames; ++t)
//                    {
//                        delete timesteps[t];
//                    }
//                }
//                else
//                {
//                    // Assign sets to output ports:
//                    poVolume[c]->setCurrentObject(timesteps[0]);
//                }
//            }
//
//            // Print message:
//            if (
//                !pboSequenceFromHeader->getValue()
//             && (numFiles != (piSequenceEnd->getValue() - piSequenceBegin->getValue() + piSequenceInc->getValue())/piSequenceInc->getValue())
//                )
//            {
//                skipped << " in sequence.";
//                sendInfo("%s", skipped.str().c_str());
//            }
//            sendInfo("Volume data loaded: %d x %d x %d voxels, %d channels, %d bytes per channel, %d time %s.",
//                     static_cast<int>(vd->vox[0]), static_cast<int>(vd->vox[1]), static_cast<int>(vd->vox[2]),
//                     vd->getChan(), static_cast<int>(vd->bpc),
//                     static_cast<int>(vd->frames), ((vd->frames == 1) ? "step (no set)" : "steps"));
//        }
//        else
//        {
//            sendError("Cannot load volume data!");
//            retVal = STOP_PIPELINE;
//        }
//    }
//    else
//    {
//        sendError("Cannot load volume data!");
//        retVal = STOP_PIPELINE;
//    }
//    delete fio;
//    delete vd;
//    return retVal;
}

MODULE_MAIN(IO, coReadFlash)
