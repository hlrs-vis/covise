/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VIRVO_NODE_
#define _VIRVO_NODE_ 1

// OSG:
#include <osg/MatrixTransform>
#include <osg/Geode>

class Virvo;
class vvRenderer;
class vvVolDesc;

using namespace osg;

/** VirvoNode: A node in OSG which contains a Geode with a 
  Virvo volume drawable. This class was derived from 
  osg::MatrixTransform to contain a transformation matrix.
  (C) 2004 Jurgen P. Schulze (schulze@cs.brown.edu)
  @author Jurgen Schulze
*/
class VirvoNode : public MatrixTransform
{
public:
    ref_ptr<Virvo> _drawable;
    ref_ptr<Geode> _geode;

    VirvoNode(Virvo::AlgorithmType);
    virtual const char *libraryName() const
    {
        return "Virvo";
    }
    virtual const char *className() const
    {
        return "VirvoNode";
    }
    virtual ~VirvoNode();
    virtual Virvo *getDrawable();
    virtual vvRenderer *getRenderer();
    virtual vvVolDesc *getVD();
    virtual bool loadVolumeFile(const char *);
};

#endif
