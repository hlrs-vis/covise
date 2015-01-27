/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VIRVO_H_
#define _VIRVO_H_

#include <osg/Drawable>
#include <osg/Vec3>
#include <osg/Image>
#include <vvrenderer.h>

class vvVolDesc;

using namespace osg;

/** This class encapsulates volume rendering with the Virvo API and makes
  it available as a Drawable for OpenSceneGraph.
  (C) 2004 Jurgen P. Schulze (schulze@cs.brown.edu)
  @author Jurgen Schulze
*/
class Virvo : public Drawable
{
public:
    enum AlgorithmType /// type of rendering algorithm
    {
        VV_MEMORY,
        VV_TEXREND,
        VV_BRICKS,
        VV_STINGRAY
    };

    Virvo(AlgorithmType);
    /** Copy constructor using CopyOp to manage deep vs shallow copy.*/
    Virvo(const Virvo &drawimage, const CopyOp &copyop = CopyOp::SHALLOW_COPY);
    virtual Object *cloneType() const
    {
        return new Virvo(_algorithm);
    }
    virtual Object *clone(const CopyOp &copyop) const
    {
        return new Virvo(*this, copyop);
    }
    virtual bool isSameKindAs(const Object *obj) const
    {
        return dynamic_cast<const Virvo *>(obj) != NULL;
    }
    virtual const char *libraryName() const
    {
        return "Virvo";
    }
    virtual const char *className() const
    {
        return "Virvo";
    }
    virtual void drawImplementation(State &state) const;

    virtual vvRenderer *getRenderer();
    virtual vvVolDesc *getVD();
    virtual bool loadVolumeFile(const char *);
    virtual void updateBoundingBox();
    virtual void setVisible(bool);
    virtual bool getVisible();
    virtual void setRenderer(AlgorithmType, vvRenderState);
    virtual void makeDynamicVolume(const char *, int, int, int);
    virtual void updateDynamicVolume(double *);

protected:
    vvRenderer *_renderer;
    vvVolDesc *_vd;
    AlgorithmType _algorithm; ///< rendering algorithm
    bool _visible;

    Virvo &operator=(const Virvo &)
    {
        return *this;
    }
    virtual ~Virvo();
    virtual BoundingBox computeBound() const;
    void initStateset();
};
#endif
