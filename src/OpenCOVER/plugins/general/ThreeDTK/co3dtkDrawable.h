/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-c++-*-
#ifndef CO_3DTK_DRAWABLE
#define CO_3DTK_DRAWABLE

#include <osg/Drawable>
#include <osg/RenderInfo>
#include "show/scancolormanager.h"
#include "show/vertexarray.h"

namespace osg
{
class State;
};

/* This vector contains the pointer to a vertex array for
 * all colors (inner vector) and all scans (outer vector)
 */
extern vector<vector<vertexArray *> > vvertexArrayList;

/**
 * Storing of AlgoType for all frames
 */
extern vector<vector<Scan::AlgoType> > MetaAlgoType;

/**
 * Storing of all transformation (frames for animation) of all scans
 */
extern vector<vector<double *> > MetaMatrix;

class co3dtkDrawable : public osg::Drawable
{
public:
    co3dtkDrawable();
    virtual ~co3dtkDrawable();

    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;

    //void  setCurrentFrame(int);

    float LevelOfDetail;
    ScanColorManager *cm;
    /**
	* the octrees that store the points for each scan
	*/
    //Show_BOctTree **octpts;
    vector<colordisplay *> octpts;

    void setScansColored(int colorScanVal);
    void calcPointSequence(vector<int> &sequence, int frameNr) const;
    void mapColorToValue(int listboxColorVal);
    void changeColorMap(int listboxColorMapVal);

    void minmaxChanged();
    void resetMinMax();
    int currentFrame; // last frame
    int frameNr; // current animation frame
    int pointmode;
    float voxelSize;

private:
    void DrawPoints() const;

    virtual osg::Object *cloneType() const
    {
        return new co3dtkDrawable();
    }
    virtual osg::Object *clone(const osg::CopyOp &copyop) const
    {
        return new co3dtkDrawable(*this, copyop);
    }

    co3dtkDrawable(const co3dtkDrawable &, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);
    void init();

    double X, Y, Z; // origin

    struct ContextState
    {
        ContextState();
        ~ContextState();
        //vvRenderer *renderer;
        //bool distributedRendering;
        //int transFuncCnt;                   ///< draw process should update the transfer function if this is not equal to myUserData->transFuncCnt
        //int lastDiscreteColors;
    };

    mutable std::vector<ContextState> contextState;
    mutable int pointSize;
};
#endif
