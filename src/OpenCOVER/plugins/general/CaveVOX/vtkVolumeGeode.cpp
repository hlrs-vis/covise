/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// VTK:
#include <vtk/vtkFloatArray.h>
#include <vtk/vtkPointData.h>
#include <vtk/vtkPolyDataNormals.h>
#include <vtk/vtkProperty.h>

// Virvo:
#include <vvvoldesc.h>

// OSG:
#include <osg/Geode>

// Local:
#include "vtkVolumeGeode.H"
#include "vtkActorToOSG.H"

#include "config/config.H"

using namespace osg;

vtkVolumeGeode::vtkVolumeGeode(vvVolDesc *vd, float isovalue, float color, int frame)
{
    const int MAX_VOL_SIZE_FOR_ISOSURFACE = CONFIGval("MAX_VOL_SIZE_FOR_ISOSURFACE", 64, 0); // voxels
    int i;

    // Create temporary volume to be able to make changes:
    _vd = new vvVolDesc(vd, frame); // copy first frame to new volume
    if (_vd->chan > 1)
    {
        _vd->convertChannels(1); // and convert to single channel
    }

    // Resample if too big:
    if (_vd->vox[0] > MAX_VOL_SIZE_FOR_ISOSURFACE || _vd->vox[1] > MAX_VOL_SIZE_FOR_ISOSURFACE || _vd->vox[2] > MAX_VOL_SIZE_FOR_ISOSURFACE)
    {
        int i, newSize[3];
        for (i = 0; i < 3; ++i)
        {
            newSize[i] = min(MAX_VOL_SIZE_FOR_ISOSURFACE, _vd->vox[i]);
        }
        _vd->resize(newSize[0], newSize[1], newSize[2], vvVolDesc::NEAREST);
    }

    int numPts = _vd->getFrameVoxels();
    vtkStructuredPoints *output = vtkStructuredPoints::New();
    output->SetDimensions(_vd->vox);
    int numCells = output->GetNumberOfCells();

    //     vtkDataSetAttributes *a=output->GetPointData();
    // ReadScalarData(a,npts)
    //     this->SetScalarLut("default");  // How is this handled??

    vtkDataArray *array = vtkFloatArray::New();

    int numComp = 1;
    array->SetNumberOfComponents(numComp);
    int numTuples = numPts;
    float *ptr = ((vtkFloatArray *)array)->WritePointer(0, numTuples * numComp);

    unsigned char *raw = _vd->getRaw();
    for (i = 0; i < numPts; ++i)
    {
        switch (_vd->bpc)
        {
        case 1:
            ptr[i] = _vd->real[0] + (_vd->real[1] - _vd->real[0]) * float(*raw) / 256.0f;
            break;
        case 2:
            ptr[i] = _vd->real[0] + (_vd->real[1] - _vd->real[0]) * (float(*raw) * 256.0f + float(*(raw + 1))) / 65535.0f;
            break;
        case 4:
            ptr[i] = *((float *)raw);
            break;
        }
        raw += _vd->bpc;
    }

    array->SetName("intensity");
    output->GetPointData()->SetScalars(array);
    _points = output;
    _iso = NULL;
    _isoActor = NULL;
    Vec3 size(1, 1, 1);
    setSize(size);
    makeIsosurface(_points, &_iso, &_isoActor, isovalue);
    _isoActor->GetProperty()->SetColor(0, color, 0);
    _node = vtkActorToOSG(_isoActor, true);

    // Set state set so as to avoid lighting errors when scaling:
    StateSet *stateSet = _node->getOrCreateStateSet();
    stateSet->setMode(GL_RESCALE_NORMAL, StateAttribute::ON);

    cerr << "Iso-surfaces created for volume: " << vd->getFilename() << endl;
}

vtkVolumeGeode::~vtkVolumeGeode()
{
    delete _vd;
    // TODO: clean up VTK memory
}

/** Assumes that setSize() is called after this.
*/
void vtkVolumeGeode::setIsoValue(float iso)
{
    _iso->SetValue(0, iso);
}

void vtkVolumeGeode::setIsoColor(float col)
{
    _isoActor->GetProperty()->SetColor(0, col, 0);
}

void vtkVolumeGeode::setSize(Vec3 &size)
{
    int i;
    float spacing[3];
    for (i = 0; i < 3; ++i)
        spacing[i] = size[i] / _vd->vox[i];
    _points->SetSpacing(spacing);

    // Update position so that object stays centered at origin:
    float pos[3];
    for (i = 0; i < 3; ++i)
        pos[i] = -size[i] / 2.0f;
    _points->SetOrigin(pos);
}

Vec3 vtkVolumeGeode::getSize()
{
    Vec3 size;
    int i;

    float *spacing = _points->GetSpacing();
    for (i = 0; i < 3; ++i)
        size[i] = spacing[i] * _vd->vox[i];
    return size;
}

void vtkVolumeGeode::makeIsosurface(vtkStructuredPoints *data,
                                    vtkContourFilter **iso, vtkActor **isoActor, float isovalue)
{
    *iso = vtkContourFilter::New();
    (*iso)->SetInput(data);
    (*iso)->SetValue(0, isovalue);
    (*iso)->ComputeScalarsOff();

    vtkPolyDataNormals *normals = vtkPolyDataNormals::New();
    normals->SetInput((*iso)->GetOutput());
    normals->SetFeatureAngle(45);

    vtkPolyDataMapper *isoMapper = vtkPolyDataMapper::New();
    isoMapper->SetInput(normals->GetOutput());
    isoMapper->ScalarVisibilityOn();
    //     isoMapper->SetScalarRange(0, 100);
    //     isoMapper->SetScalarModeToUsePointFieldData();
    // //     isoMapper->ColorByArraycomponent("VelocityMagnitude", 0);

    //     vtkLODActor *isoActor = vtkLODActor::New();
    //     isoActor->SetMapper(isoMapper);
    //     isoActor->SetNumberOfCloudPoints(1000);
    *isoActor = vtkActor::New();
    (*isoActor)->SetMapper(isoMapper);
    (*isoActor)->GetProperty()->SetColor(0, 1, 0);
}

osg::Node *vtkVolumeGeode::getNode()
{
    return _node.get();
}
