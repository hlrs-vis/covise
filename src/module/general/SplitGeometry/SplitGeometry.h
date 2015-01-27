/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: extract element of a set                               ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 04.10.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef SPLITGEOMETRY_H
#define SPLITGEOMETRY_H

#include <do/coDoSet.h>
#include <api/coSimpleModule.h>
using namespace covise;

class SplitGeometry : public coSimpleModule
{
public:
    SplitGeometry(int argc, char *argv[]);

    virtual ~SplitGeometry();

private:
    /// compute call-back
    virtual int compute(const char *port);

    /// ports
    coInputPort *m_inPort;
    coOutputPort *m_outPortGrid, *m_outPortData, *m_outPortNormals, *m_outPortTexture;
};
#endif
