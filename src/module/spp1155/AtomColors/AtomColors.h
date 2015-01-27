/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ATOMCOLORS_H
#define ATOMCOLORS_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2004 ZAIK/RRZK  ++
// ++ Description: Atom Properties module                                 ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                       Thomas van Reimersdahl                        ++
// ++               Institute for Computer Science (Prof. Lang)           ++
// ++                        University of Cologne                        ++
// ++                         Robert-Koch-Str. 10                         ++
// ++                             50931 Köln                              ++
// ++                                                                     ++
// ++ Date:  26.12.2004                                                   ++
// ++**********************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <config/coConfig.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>

struct AtomColor
{
    enum
    {
        TypeLength = 3
    };
    char type[TypeLength];
    float color[4];
};

class AtomColors : public coSimpleModule
{
public:
    AtomColors(int argc, char *argv[]);

private:
    //////////  inherited mamber functions
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    virtual void preHandleObjects(coInputPort **);

#ifdef ATOMRADII
    typedef coDoFloat coDoResult;
#else
    typedef coDoRGBA coDoResult;
#endif

    coDoResult *getOutputData(const coDoSet *inData);
    coDoResult *getOutputData(const coDoInt *inData);
    coDoResult *getOutputData(const coDoFloat *inData);

    ////////// ports
    coInputPort *m_portInData;
    coInputPort *m_portInPoints; // for setting port leader (if handling deeper nested coDoText objects)
    coOutputPort *m_portOutColor;

    coConfigGroup *m_mapConfig;

#ifdef ATOMRADII
    std::vector<coFloatParam *> m_atom;
#else
    std::vector<coColorParam *> m_atom;
#endif
    std::vector<AtomColor> m_rgb;
    std::vector<float> m_radius;
};
#endif
