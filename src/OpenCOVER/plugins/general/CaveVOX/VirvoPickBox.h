/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VIRVO_PICK_BOX_H_
#define _VIRVO_PICK_BOX_H_

#include "vvarray.h"

// Local:
#include "Virvo.H"
#include "VolumePickBox.H"

// gluebase
#include "std/list.H"

class vtkVolumeGeode;

/**
    This class provides a Virvo object in a PickBox. The pick box
    provides a wireframe box around the volume, and it allows the volume
    to be moved around by the user. VirvoPickBox is to be used _alternatively_
    to VirvoNode, they shouldn't be interdependent.
*/
class VirvoPickBox : public VolumePickBox
{
protected:
    ref_ptr<Virvo> _virvoDrawable;
    ref_ptr<Geode> _geode;
    osg::ref_ptr<osg::Switch> _vtkSwitch;
    osg::ref_ptr<osg::MatrixTransform> _vtkThickness;
    ARRAY<vtkVolumeGeode *> _vtkVolumes;
    osg::ref_ptr<osg::MatrixTransform> _vtkNode;

    float _isoValue;
    float _isoColor;
    osg::Matrix _localToVoxel;

    /*
      Fills vvArray<int*> with the voxeldata
      on the line given by _measure.
    */
    void getLineData(vvArray<float *> &);

    void computeLocalToVoxel();

public:
    enum DataType
    {
        VIRVO,
        VTK
    };

    VirvoPickBox(cui::Interaction *, osgDrawObj *, const osg::Vec3 &, const osg::Vec3 &,
                 const osg::Vec4 &, const osg::Vec4 &, const osg::Vec4 &, Virvo::AlgorithmType);
    virtual ~VirvoPickBox();
    virtual Virvo *getDrawable();
    virtual vvRenderer *getRenderer();
    virtual vvVolDesc *getVD();
    virtual bool loadVolumeFile(const char *);
    virtual void updateBoundingBox();
    virtual void cursorUpdate(cui::InputDevice *);
    virtual void setThickness(float);
    virtual float getThickness();
    virtual void setVisible(DataType, bool);
    virtual void setIsoValue(float);
    virtual float getIsoValue();
    virtual void setIsoColor(float);
    virtual float getIsoColor();
    virtual void updateIsoGeometry();
    virtual int nextTimeStep();
    virtual int prevTimeStep();
    virtual void recreateVTKIsosurfaces();
    virtual void setCurrentFrame(int frame);
    virtual void makeDynamicVolume(const char *, int, int, int);
    virtual void updateDynamicVolume(double *);

    /*
    Computes an intensity diagram of the line given
    by _measure. Changes the _intensImage.
  */
    void computeIntensityDiagram();

    /*
    Computes a histogram of the line given by
    _measure. Changes the _histoImage.
  */
    void computeHistogram();

    void computeRectIntensityDiagram();
    void computeRectHistogram();
    void getRectHistogramData(vvArray<float *> &);
    unsigned char *getRectData(int &, int &);
};
#endif
