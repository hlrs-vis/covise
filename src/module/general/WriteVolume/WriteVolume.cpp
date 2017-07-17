/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2002 RUS  **
 **                                                                        **
 ** Description: Write volume files in formats supported by Virvo.         **
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
 ** Cration Date: 26.06.02                                                 **
\**************************************************************************/

#include <api/coModule.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoUniformGrid.h>
#include <virvo/math/math.h>
#include <virvo/vvfileio.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvtoolshed.h>
#include "WriteVolume.h"

/// Constructor
coWriteVolume::coWriteVolume(int argc, char *argv[])
    : coModule(argc, argv, "Write volume files or create 2D slice images.")
{
    const char *fileTypes[] = {
        "Extended volume file (xvf)",
        "Raw volume file (rvf)",
        "ASCII volume file (avf)",
        "Raw volume data (dat)",
        "2D slice images (pgm/ppm)"
    };
    const char *formatTypes[] = {
        "8 bits per voxel",
        "16 bits per voxel"
    };

    // Create ports:
    piGrid = addInputPort("grid", "UniformGrid", "Grid for volume data");
    piGrid->setInfo("Grid for volume data");

    piChan[0] = addInputPort("channel0",
                             "Byte|Float|Vec3|RGBA", "Scalar/vector volume data");
    piChan[0]->setInfo("Scalar volume data (channel 0)/vector volume data");

    for (int i = 1; i < MAX_CHANNELS; i++)
    {
        char name[1024];
        char desc[1024];
        sprintf(name, "channel%d", i);
        sprintf(desc, "Scalar volume data channel %d", i);
        piChan[i] = addInputPort(name, "Byte|Float", desc);
        piChan[i]->setRequired(0);
    }

    // Create parameters:
    pbrVolumeFile = addFileBrowserParam("FileName", "Volume file or first slice of sequence");
    pbrVolumeFile->setValue("data/volumeFile", "*.xvf/*.rvf/*.avf/*.dat/*.pgm;*.ppm/*");

    pboOverwrite = addBooleanParam("OverwriteExisting", "Overwrite existing files?");
    pboOverwrite->setValue(false);

    pchFileType = addChoiceParam("FileType", "File type of volume or slice files");
    pchFileType->setValue(4, fileTypes, 0);

    pchDataFormat = addChoiceParam("DataFormat", "Volume data format");
    pchDataFormat->setValue(2, formatTypes, 0);

    pfsMin = addFloatParam("MinimumValue", "Minimum scalar input value");
    pfsMin->setValue(0.0);

    pfsMax = addFloatParam("MaximumValue", "Maximum scalar input value");
    pfsMax->setValue(1.0);
}

/** Determine grid size (= volume size).
  This function calls itself recursively to unwrap sets. The first occurrence
  of a uniform grid determines the volume size used for all time steps.
  @return true if the grid size was found
*/
bool coWriteVolume::getGridSize(const coDistributedObject *object, int *size, float *min, float *max)
{
    const coDistributedObject *const *elem; // set elements
    int numElem; // number of set elements
    int i; // counter
    float ix, ax, iy, ay, iz, az; // volume extent [mm]

    if (const coDoSet *set = dynamic_cast<const coDoSet *>(object))
    {
        elem = set->getAllElements(&numElem);
        for (i = 0; i < numElem; i++)
        {
            if (getGridSize(elem[i], size, min, max))
                return true;
        }
        return false;
    }

    else if (const coDoUniformGrid *grid = dynamic_cast<const coDoUniformGrid *>(object))
    {
        grid->getGridSize(&size[0], &size[1], &size[2]);
        grid->getMinMax(&ix, &ax, &iy, &ay, &iz, &az);
        min[0] = ix;
        max[0] = ax;
        min[1] = iy;
        max[1] = ay;
        min[2] = iz;
        max[2] = az;
        return true;
    }

    return false;
}

/** Read a data object from the volume data input port and add it to the
  list of volumes. This routine calls itself recursively to process sets.
*/
void coWriteVolume::getTimestepData(const coDistributedObject **object, int no_channels)
{
    const char *string1 = "All time steps must have equal grid sizes!";
    int x = 0, y = 0, z = 0; // grid size
    float *fData[MAX_CHANNELS];
    uchar *uData;

    const coDoSet *set0 = dynamic_cast<const coDoSet *>(object[0]);
    if (set0)
    {
        int no_elem = set0->getNumElements();

        const coDistributedObject **elem = new const coDistributedObject *[MAX_CHANNELS];
        for (int i = 0; i < no_elem; i++)
        {
            for (int c = 0; c < no_channels; c++)
            {
                const coDoSet *set = dynamic_cast<const coDoSet *>(object[c]);
                if (set)
                {
                    if (set->getNumElements() != no_elem)
                    {
                        sendError("timestep data must be compatible\n");
                        elem[c] = NULL;
                    }
                    else
                    {
                        elem[c] = set->getElement(i);
                    }
                }
                else
                {
                    elem[c] = NULL;
                }
            }
            getTimestepData(elem, no_channels);
        }
    }
    else if (const coDoRGBA *rgba = dynamic_cast<const coDoRGBA *>(object[0]))
    {
        if (volSize[0] * volSize[1] * volSize[2] != rgba->getNumPoints())
        {
            fprintf(stderr, "x=%d, y=%d, z=%d, elem=%d\n",
                    volSize[0], volSize[1], volSize[2],
                    rgba->getNumPoints());
            sendError("%s", string1);
        }
        else
        {
            int *data = NULL;
            rgba->getAddress(&data);
            uData = new uchar[vd->getFrameBytes()];
            memcpy(uData, data, vd->getFrameBytes());
#ifdef BYTESWAP
            for (int i = 0; i < vd->getFrameVoxels(); i++)
            {
                byteSwap(((uint32_t *)uData)[i]);
            }
#endif
            vd->addFrame(uData, vvVolDesc::ARRAY_DELETE);
            ++vd->frames;
        }
    }
    else if (const coDoVec3 *vec = dynamic_cast<const coDoVec3 *>(object[0]))
    {
        int nelem = vec->getNumPoints();
        if (volSize[0] * volSize[1] * volSize[2] != nelem)
        {
            sendError("%s", string1);
        }
        else
        {
            vec->getAddresses(&fData[0], &fData[1], &fData[2]);
            uData = new uchar[vd->getFrameBytes()];
            switch (pchDataFormat->getValue())
            {
            default:
            case 0:
            {
                for (int i = 0; i < vd->getFrameVoxels(); i++)
                {
                    vvToolshed::convertFloat2UCharClamp(fData[0] + i, uData + 3 * i, 1, pfsMin->getValue(), pfsMax->getValue());
                    vvToolshed::convertFloat2UCharClamp(fData[1] + i, uData + 3 * i + 1, 1, pfsMin->getValue(), pfsMax->getValue());
                    vvToolshed::convertFloat2UCharClamp(fData[2] + i, uData + 3 * i + 2, 1, pfsMin->getValue(), pfsMax->getValue());
                }
                break;
            }
            case 1:
            {
                for (int i = 0; i < vd->getFrameVoxels(); i++)
                {
                    vvToolshed::convertFloat2ShortClamp(fData[0] + i, uData + 6 * i, 1, pfsMin->getValue(), pfsMax->getValue());
                    vvToolshed::convertFloat2ShortClamp(fData[1] + i, uData + 6 * i + 2, 1, pfsMin->getValue(), pfsMax->getValue());
                    vvToolshed::convertFloat2ShortClamp(fData[2] + i, uData + 6 * i + 4, 1, pfsMin->getValue(), pfsMax->getValue());
                }
                break;
            }
            }
            vd->addFrame(uData, vvVolDesc::ARRAY_DELETE);
            ++vd->frames;
        }
    }
    else if (dynamic_cast<const coDoFloat *>(object[0]))
    {
        const coDoFloat *scalar[MAX_CHANNELS]; // scalar volume data
        int no_channels = 0;
        for (int c = 0; c < MAX_CHANNELS; c++)
        {
            scalar[c] = dynamic_cast<const coDoFloat *>(object[c]);
            if (!scalar[c])
            {
                no_channels = c;
                break;
            }
        }

        for (int c = 0; c < no_channels; c++)
        {
            int nelem = scalar[c]->getNumPoints();
            if (volSize[0] * volSize[1] * volSize[2] != nelem)
            {
                sendError("grid size: (%d %d %d)\n", x, y, z);
                return;
            }
            else
            {
                scalar[c]->getAddress(&fData[c]);
            }
        }
        uData = new uchar[vd->getFrameBytes()];
        switch (pchDataFormat->getValue())
        {
        default:
        case 0:
            for (int c = 0; c < no_channels; c++)
            {
                for (int i = 0; i < vd->getFrameVoxels(); i++)
                {
                    vvToolshed::convertFloat2UCharClamp(fData[c] + i, uData + no_channels * i + c, 1, pfsMin->getValue(), pfsMax->getValue());
                }
            }
            break;
        case 1:
            for (int c = 0; c < no_channels; c++)
            {
                for (int i = 0; i < vd->getFrameVoxels(); i++)
                {
                    vvToolshed::convertFloat2ShortClamp(fData[c] + i, uData + no_channels * i * 2 + c, 1, pfsMin->getValue(), pfsMax->getValue(), virvo::serialization::getEndianness());
                }
            }
            break;
        }
        vd->addFrame(uData, vvVolDesc::ARRAY_DELETE);
        ++vd->frames;
    }
    else if (dynamic_cast<const coDoByte *>(object[0]))
    {
        const coDoByte *scalar[MAX_CHANNELS]; // scalar volume data
        uchar *bData[MAX_CHANNELS];
        int no_channels = 0;
        for (int c = 0; c < MAX_CHANNELS; c++)
        {
            scalar[c] = dynamic_cast<const coDoByte *>(object[c]);
            if (!scalar[c])
            {
                no_channels = c;
                break;
            }
        }

        for (int c = 0; c < no_channels; c++)
        {
            int nelem = scalar[c]->getNumPoints();
            if (volSize[0] * volSize[1] * volSize[2] != nelem)
            {
                sendError("grid size: (%d %d %d)\n", x, y, z);
                return;
            }
            else
            {
                scalar[c]->getAddress(&bData[c]);
            }
        }
        uData = new uchar[vd->getFrameBytes()];
        switch (pchDataFormat->getValue())
        {
        default:
        case 0:
            for (int c = 0; c < no_channels; c++)
            {
                for (int i = 0; i < vd->getFrameVoxels(); i++)
                {
                    uData[no_channels * i + c] = *(bData[c] + i);
                }
            }
            break;
        case 1:
            for (int c = 0; c < no_channels; c++)
            {
                for (int i = 0; i < vd->getFrameVoxels(); i++)
                {
                    uData[(no_channels * i + c) * 2] = *(bData[c] + i);
                    uData[(no_channels * i + c) * 2 + 1] = 0;
                }
            }
            break;
        }
        vd->addFrame(uData, vvVolDesc::ARRAY_DELETE);
        ++vd->frames;
    }
}

/// The compute routine is called whenever the module is executed in Covise.
int coWriteVolume::compute(const char *)
{
    const coDistributedObject *tmpObj;
    const coDistributedObject *inObj[MAX_CHANNELS];
    const char *extension[] = { "xvf", "rvf", "avf", "dat", "pgm" };
    const char *string1 = "Volume saved to file: ";
    const char *string2 = "Destination file exists: ";
    const char *string3 = "Cannot write file: ";
    const char *string4 = "No grid found at input port.";
    const char *string5 = "Invalid grid size.";
    const char *path;
    char *newPath;
    vvFileIO *fio;
    virvo::vec3 position;
    int retVal = CONTINUE_PIPELINE;
    bool overwrite;
    float minPos[3]; // minimum volume extent [mm]
    float maxPos[3]; // maximum volume extent [mm]
    int bytesPerChannel;
    int i;

    // Process grid input data:
    tmpObj = piGrid->getCurrentObject();
    if (getGridSize(tmpObj, volSize, minPos, maxPos) == false)
    {
        sendError("%s", string4);
        return STOP_PIPELINE;
    }
    if (volSize[0] <= 0 || volSize[1] <= 0 || volSize[2] <= 0)
    {
        sendError("%s", string5);
        return STOP_PIPELINE;
    }

    bool haveVectorData = false;
    bool haveRGBAData = false;
    int no_channels = 0;
    for (i = 0; i < MAX_CHANNELS; i++)
    {
        inObj[i] = piChan[i]->getCurrentObject();
        if (!inObj[i])
            break;
        if (i == 0 && dynamic_cast<const coDoVec3 *>(inObj[i]))
        {
            haveVectorData = true;
            no_channels = 3;
            break;
        }
        if (i == 0 && dynamic_cast<const coDoRGBA *>(inObj[i]))
        {
            haveRGBAData = true;
            no_channels = 4;
            break;
        }
        no_channels++;
    }

    // Create volume description:
    path = pbrVolumeFile->getValue();
    newPath = new char[strlen(path) + 10];
    vvToolshed::replaceExtension(newPath, extension[pchFileType->getValue()], path);
    bytesPerChannel = pchDataFormat->getValue();
    //cerr << "WriteVolume: volSize[0], volSize[1], volSize[2] " << volSize[0] << " " << volSize[1] << " " <<  volSize[2] << " " << bytesPerChannel  <<endl;
    vd = new vvVolDesc(newPath, volSize[0], volSize[1], volSize[2], 0, bytesPerChannel + 1, no_channels, NULL);
    for (i = 0; i < 3; ++i)
    {
        position[i] = fabs(0.5f * (maxPos[i] + minPos[i]));
        vd->dist[i] = (maxPos[i] - minPos[i]) / volSize[i];
    }
    vd->pos = position;

    // Read timestep data from input port:
    if (haveRGBAData || haveVectorData)
    {
        getTimestepData(inObj, 1);
    }
    else
    {
        getTimestepData(inObj, no_channels);
    }
    vd->printVolumeInfo();
    // Convert data to Virvo format:
    vd->convertCoviseToVirvo();

    // Save volume file:
    fio = new vvFileIO();
    overwrite = (bool)pboOverwrite->getValue();
    switch (fio->saveVolumeData(vd, overwrite, vvFileIO::ALL_DATA))
    {
    case vvFileIO::OK:
        sendInfo("%s%s", string1, newPath);
        retVal = CONTINUE_PIPELINE;
        break;
    case vvFileIO::FILE_EXISTS:
        sendError("%s%s", string2, newPath);
        retVal = STOP_PIPELINE;
        break;
    default:
        sendError("%s%s", string3, newPath);
        retVal = STOP_PIPELINE;
        break;
    }
    delete[] newPath;
    delete fio;
    delete vd;
    return retVal;
}

MODULE_MAIN(IO, coWriteVolume)
