/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS SmokeGeneratorSolutions
//
//  Auxiliary class of the SmokeGenerator plugin.
//  SmokeGeneratorSolutions gets a set of streamlines
//  as argument of the function set, and produces
//  arrays in lengths_ and linepoints_ which may be
//  used for the construction of lines in the format
//  specified by Performer.
//
//  Initial version: 24.06.2004 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _SMOKE_GENERATOR_SOLUTIONS_H_
#define _SMOKE_GENERATOR_SOLUTIONS_H_

#include <util/common.h>

#include <osg/Vec3>
#include <alg/coUniTracer.h>
#include <osg/Geometry>

class COVISEPLUGINEXPORT SmokeGeneratorSolutions
{
public:
    /// constructor
    SmokeGeneratorSolutions();
    /// destructor
    virtual ~SmokeGeneratorSolutions();
    /// clear throws away old solutions
    void clear();
    /// set builds up internal arrays based on the argument solus (streamlines)
    void set(const vector<vector<covise::coUniState> > &solus);
    /// lengths outputs the array of line lengths
    osg::DrawArrayLengths *lengths() const;
    /// lengths outputs the array of streamline points
    osg::Vec3Array *linepoints() const;
    /// size outputs the array of streamlines (number of
    /// primitives from the perspective of Performer)
    int size() const;

private:
    osg::ref_ptr<osg::DrawArrayLengths> lengths_; // line lengths
    osg::ref_ptr<osg::Vec3Array> linepoints_; // line points
    int size_; // number of lines
};

#endif
