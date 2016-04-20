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

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include <api/coModule.h>
#include <api/coFeedback.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoSet.h>
#include <virvo/vvfileio.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvtoolshed.h>
#include <virvo/fileio/feature.h>
#include "ReadVolume.h"


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
coReadVolume::coReadVolume(int argc, char *argv[])
    : coModule(argc, argv, "Read volume files or create a volume from 2D slice images.")
{

    // Create ports:
    poGrid = addOutputPort("grid", "UniformGrid", "Grid for volume data");
    poGrid->setInfo("Grid for volume data");

    for (int c = 0; c < MAX_CHANNELS; c++)
    {
        char buf[1024];
        sprintf(buf, "channel%d", c);
        char buf2[1024];
        sprintf(buf2, "Scalar volume data channel %d", c);
        poVolume[c] = addOutputPort(buf, "Float|Byte", buf2);
        poVolume[c]->setInfo(buf2);
    }

    std::stringstream filetypes;
    filetypes << "*.xvf;*.rvf;*.avf/*.dcm;*.dcom/";
    if (virvo::fileio::hasFeature("nifti"))
    {
        filetypes << "*.nii;*.nii.gz/";
    }
    filetypes << "*.tif;*.tiff/*.rgb/*.pgm;*.ppm/*";
    std::string ftstr = filetypes.str();
    const char *ftptr = ftstr.c_str();

    // Create parameters:
    pbrVolumeFile = addFileBrowserParam("FilePath", "Volume file (or printf format string for sequence)");
    pbrVolumeFile->setValue("data/volume", ftptr);

    piSequenceBegin = addInt32Param("SequenceBegin", "First file number in sequence");
    piSequenceBegin->setValue(0);

    piSequenceEnd = addInt32Param("SequenceEnd", "Last file number in sequence");
    piSequenceEnd->setValue(0);

    piSequenceInc = addInt32Param("SequenceInc", "Increment in sequence");
    piSequenceInc->setValue(1);

    pboSequenceFromHeader = addBooleanParam("SequenceFromHeader", "Off: read single file or file sequence, on: process all files in folder, determine order from file header");
    pboSequenceFromHeader->setValue(false);

    pboPreferByteData = addBooleanParam("PreferByteData", "Off: never create byte data, on: use byte data for volumes stored with 1 byte/channel");
    pboPreferByteData->setValue(false);

    pboCustomSize = addBooleanParam("CustomSize", "Off: use size values from volume file, on: use size values from below");
    pboCustomSize->setValue(false);

    pfsVolWidth = addFloatParam("VolumeWidth", "Volume width");
    pfsVolWidth->setValue(1.0);

    pfsVolHeight = addFloatParam("VolumeHeight", "Volume height");
    pfsVolHeight->setValue(1.0);

    pfsVolDepth = addFloatParam("VolumeDepth", "Volume depth");
    pfsVolDepth->setValue(1.0);

    pboReadRaw = addBooleanParam("ReadRaw", "Off: read data according to file format guessed from extension, on: read raw data as specified below");
    pboReadRaw->setValue(false);

    pboReadBS = addBooleanParam("ReadBS", "Swap Bytes");
    pboReadBS->setValue(false);

    piVoxelsX = addInt32Param("NumVoxelX", "Number of voxels in x direction (width)");
    piVoxelsX->setValue(512);

    piVoxelsY = addInt32Param("NumVoxelY", "Number of voxels in y direction (height)");
    piVoxelsY->setValue(512);

    piVoxelsZ = addInt32Param("NumVoxelZ", "Number of voxels in z direction (slices)");
    piVoxelsZ->setValue(1);

    piBPC = addInt32Param("BytePerChannel", "Byte per channel");
    piBPC->setValue(1);

    piChannels = addInt32Param("NumberOfChannels", "Number of channels");
    piChannels->setValue(1);

    piHeader = addInt32Param("HeaderSize", "Offset of raw volume data from file beginning");
    piHeader->setValue(0);

    minValue = addInt32Param("minValue", "Minimum values for 16 bit to 0.0-1.0 float values");
    minValue->setValue(0);

    maxValue = addInt32Param("maxValue", "Maximum values for 16 bit to 0.0-1.0 float values");
    maxValue->setValue(65535);
}

/// This is our compute-routine
int coReadVolume::compute(const char *)
{
    coDoUniformGrid **gridData = NULL;

    coDoSet *volumeSet = NULL;
    coDoSet *gridSet = NULL;

    unsigned char *rawData = NULL;
    float width = 0.0, height = 0.0, depth = 0.0;
    int t;
    size_t numVoxels;
    const char *pathtemplate = pbrVolumeFile->getValue();

    vvFileIO *fio = new vvFileIO();
    vvVolDesc *vd = new vvVolDesc();

    std::ostringstream skipped;
    skipped << "Skipped";

    int retVal = CONTINUE_PIPELINE;
    int numFiles = 0;

    FolderUtil folder_util = pboSequenceFromHeader->getValue()
        ? FolderUtil(pathtemplate)
        : FolderUtil(pathtemplate, FolderUtil::Sequence(
                piSequenceBegin->getValue(),
                piSequenceEnd->getValue() + piSequenceInc->getValue(),
                piSequenceInc->getValue()
                )
                )
        ;

    for (FolderIterator it =  folder_util.begin();
                        it != folder_util.end();
                        ++it
                        )
    {
        namespace fs = boost::filesystem;

        if (fs::is_directory(it.status()))
        {
            continue;
        }

        vvVolDesc *vdread = vd, vdframe;
        bool merge = false;
        if (vd->vox[2] > 1 && vd->frames > 0)
        {
            merge = true;
            vdread = &vdframe;
        }
        std::string path = it.path_string();
        vdread->setFilename(path.c_str());
        int result = vvFileIO::OK;
        if (pboReadRaw->getValue())
        {
            result = fio->loadRawFile(vdread,
                                      piVoxelsX->getValue(), piVoxelsY->getValue(), piVoxelsZ->getValue(),
                                      piBPC->getValue(), piChannels->getValue(), piHeader->getValue());
        }
        else
        {
            result = fio->loadVolumeData(vdread, vvFileIO::ALL_DATA, it != folder_util.begin() && !merge);
        }

        if (result == vvFileIO::OK)
        {
            numFiles++;
            //vdread->printVolumeInfo();

            // Determine volume dimensions:
            if (pboCustomSize->getValue())
            {
                width = pfsVolWidth->getValue();
                height = pfsVolHeight->getValue();
                depth = pfsVolDepth->getValue();
            }
            else
            {
                width = vdread->vox[0] * vdread->dist[0];
                height = vdread->vox[1] * vdread->dist[1];

                if (folder_util.slice_format() && !merge)
                {
                    depth = numFiles * vdread->dist[2];
                }
                else
                {
                    depth = vdread->vox[2] * vdread->dist[2];
                }
            }
        }
        else if (result == vvFileIO::FILE_NOT_FOUND)
        {
            skipped << " " << it.toInt();
        }
        else
        {
            sendError("Failed to load file %d from sequence: %s", it.toInt(), path.c_str());
            retVal = STOP_PIPELINE;
            break;
        }

        if (merge)
        {
            vd->merge(&vdframe, vvVolDesc::VV_MERGE_VOL2ANIM);
        }
    }

    if (numFiles > 0)
    {

        if (vd->vox[2] == 1 && vd->frames > 0)
        {
            vd->mergeFrames(); // merge single slices into one volume
        }

        vd->printVolumeInfo();
        vd->convertVirvoToCovise();

        // Create set elements:
        // important: allocate one more than frames are present!
        gridData = new coDoUniformGrid *[vd->frames + 1];

        // Create time steps:
        for (t = 0; t < vd->frames; ++t)
        {
            float posX = vd->pos[0] * vd->dist[0];
            float posY = vd->pos[1] * vd->dist[1];
            float posZ = vd->pos[2] * vd->dist[2];
            float maxX = posX + 0.5f * width;
            float maxY = posY + 0.5f * height;
            float maxZ = posZ + 0.5f * depth;
            float minX = posX - 0.5f * width;
            float minY = posY - 0.5f * height;
            float minZ = posZ - 0.5f * depth;
            int vox[] = { static_cast<int>(vd->vox[0]), static_cast<int>(vd->vox[1]), static_cast<int>(vd->vox[2]) };
            if (vd->frames > 1)
            {
                char buf[1024];
                sprintf(buf, "%s_%d", poGrid->getObjName(), t);
                gridData[t] = new coDoUniformGrid(buf, vox[0], vox[1], vox[2], minX, maxX, minY, maxY, minZ, maxZ);
            }
            else
            {
                gridData[t] = new coDoUniformGrid(poGrid->getObjName(), vox[0], vox[1], vox[2], minX, maxX, minY, maxY, minZ, maxZ);
            }
        }
        gridData[vd->frames] = NULL;

        coFeedback browserFeedback("FileBrowser");
        browserFeedback.addPara(pbrVolumeFile);

        if (vd->frames > 1)
        {
            char buf[1024];
            sprintf(buf, "0 %d", (int)vd->frames - 1);
            gridSet = new coDoSet(poGrid->getObjName(), (coDistributedObject **)gridData);
            gridSet->addAttribute("TIMESTEP", buf);
            browserFeedback.apply(gridSet);
            poGrid->setCurrentObject(gridSet);

            for (t = 0; t < vd->frames; t++)
            {
                delete gridData[t];
            }
        }
        else
        {
            browserFeedback.apply(gridData[0]);
            poGrid->setCurrentObject(gridData[0]);
        }

        delete[] gridData;

        // Copy raw volume data to shared memory:
        numVoxels = vd->getFrameVoxels();

        for (int c = 0; c < vd->chan && c < MAX_CHANNELS; c++)
        {
            std::vector<coDistributedObject *> timesteps;
            for (t = 0; t < vd->frames; ++t)
            {
                std::stringstream name(poVolume[c]->getObjName());
                if (vd->frames > 1)
                    name << "_" << t;
                coDoByte *dob = NULL;
                coDoFloat *dof = NULL;
                float *fdata = NULL;
                uchar *bdata = NULL;
                if (vd->bpc == 1 && pboPreferByteData->getValue())
                {
                    dob = new coDoByte(name.str(), vd->vox[0] * vd->vox[1] * vd->vox[2]);
                    bdata = dob->getAddress();
                    timesteps.push_back(dob);
                }
                else
                {
                    dof = new coDoFloat(name.str(), vd->vox[0] * vd->vox[1] * vd->vox[2]);
                    fdata = dof->getAddress();
                    timesteps.push_back(dof);
                }

                std::string min_str = boost::lexical_cast<std::string>(vd->real[c][0]);
                std::string max_str = boost::lexical_cast<std::string>(vd->real[c][1]);
                timesteps.back()->addAttribute("MIN", min_str.c_str());
                timesteps.back()->addAttribute("MAX", max_str.c_str());

                rawData = vd->getRaw((size_t)t);

                bool bs = pboReadBS->getValue();
                float minV = (float)minValue->getValue();
                float range = (float)(maxValue->getValue() - minValue->getValue());
                {
                    if (vd->bpc == 1)
                    {
                        if (fdata)
                        {
                            for (size_t i = 0; i < numVoxels; ++i)
                                fdata[i] = float(rawData[i * vd->chan + c]) / 255.0f;
                        }
                        else
                        {
                            for (size_t i = 0; i < numVoxels; ++i)
                                bdata[i] = rawData[i * vd->chan + c];
                        }
                    }
                    else if (vd->bpc == 2)
                    {
                        if (bs) //   Big Endian
                        {
                            //low
                            for (size_t i = 0; i < numVoxels; ++i)
                            {
                                fdata[i] = (((256.0f * ((float)rawData[(vd->chan * i + c) * 2]))
                                             + ((float)rawData[(vd->chan * i + c) * 2 + 1]))
                                            - minV) / range;
                                fdata[i] = ts_clamp(fdata[i], 0.f, 1.f);
                            }
                        }
                        else //    Little Endian
                        {
                            for (size_t i = 0; i < numVoxels; ++i)
                            {
                                //high
                                fdata[i] = (((256.0f * ((float)rawData[(vd->chan * i + c) * 2 + 1]))
                                             + ((float)rawData[(vd->chan * i + c) * 2]))
                                            - minV) / range;
                                fdata[i] = ts_clamp(fdata[i], 0.f, 1.f);
                            }
                        } //
                    }
                    else if (vd->bpc == 4)
                    {
                        if (bs)
                        {
                            for (size_t i = 0; i < numVoxels; ++i)
                            {
                                uint32_t d = *(uint32_t *)&rawData[(vd->chan * i + c) * 4];
                                byteSwap(d);
                                fdata[i] = *(float *)&d;
                            }
                        }
                        else
                        {
                            for (size_t i = 0; i < numVoxels; ++i)
                            {
                                fdata[i] = *(float *)&rawData[(vd->chan * i + c) * 4];
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < numVoxels; ++i)
                            fdata[i] = 0.0f;
                    }
                }
            }

            if (vd->frames > 1)
            {
                // Create set objects:
                volumeSet = new coDoSet(poVolume[c]->getObjName(), timesteps.size(), &timesteps[0]);

                // Set timestep attribute:
                char buf[1024];
                sprintf(buf, "%d %d", 0, (int)vd->frames - 1);
                volumeSet->addAttribute("TIMESTEP", buf);

                // Assign sets to output ports:
                poVolume[c]->setCurrentObject(volumeSet);

                for (t = 0; t < vd->frames; ++t)
                {
                    delete timesteps[t];
                }
            }
            else
            {
                // Assign sets to output ports:
                poVolume[c]->setCurrentObject(timesteps[0]);
            }
        }

        // Print message:
        if (
            !pboSequenceFromHeader->getValue()
         && (numFiles != piSequenceEnd->getValue() - piSequenceBegin->getValue() + 1)
            )
        {
            skipped << " in sequence.";
            sendInfo("%s", skipped.str().c_str());
        }
        sendInfo("Volume data loaded: %d x %d x %d voxels, %d channels, %d bytes per channel, %d time %s.",
                 static_cast<int>(vd->vox[0]), static_cast<int>(vd->vox[1]), static_cast<int>(vd->vox[2]),
                 static_cast<int>(vd->chan), static_cast<int>(vd->bpc),
                 static_cast<int>(vd->frames), ((vd->frames == 1) ? "step (no set)" : "steps"));
    }
    else
    {
        sendError("Cannot load volume data!");
        retVal = STOP_PIPELINE;
    }
    delete fio;
    delete vd;
    return retVal;
}

MODULE_MAIN(IO, coReadVolume)
