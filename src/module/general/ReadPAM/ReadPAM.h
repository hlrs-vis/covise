/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_PAM_H_
#define _READ_PAM_H_
/**************************************************************************\ 
 **                                                     (C)2001 Vircinity  **
 **                                                                        **
 ** Description: Reads DSY and THP files. DSY is used for grid and         **
 **              visualisation data, and optionally also for plots.        **
 **              The THP file is only used for plots.                      **
 **                                                                        **
 ** Author:                                                                **
 **                            Sergio Leseduarte                           **
 **                            Vircinity GmbH                              **
 **                            Nobelstr. 15                                **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  Ende Juni 2001?  (coding begins)                                **
 ** Sergio Leseduarte:                                                     **
 **                   Ende Juli 2001: Conditionally compiled are the       **
 **                           lines for creating local reference systems.  **
 **                           Additional information is needed             **
 **                           for this task that is not available in the   **
 **                           documentation of the DSY library.            **
\**************************************************************************/
#ifndef _WIN32
#define COIDENT "$Header: /vobs/covise/src/application/general/READ_PAM/ReadPAM.h /main/vir_main/1 18-Dec-2001.11:12:59 we_te $"
#include <util/coIdent.h>
#endif

#include <api/coModule.h>
#include "ReadDSY.h"
#include "auxiliary.h"

using namespace covise;

// ReadPam handles the interface with the user.
// Requests from the user are comunicated to the readDSY (ReadDSY.h)
// object, which reads the files and creates the output objects.
// The communication of the requests is supported by auxiliary
// classes.
class ReadPam : public coModule
{
private:
    enum
    {
        NODAL_PORTS = 2,
        CELL_PORTS = 2,
        GLOBAL_PORTS = 2,
        TENSOR_PORTS = 1
    };
    int dsy_ok_;
    int thp_ok_;
    virtual int compute(const char *port);
    virtual void param(const char *, bool inMapLoading);
    // Parameters
    coFileBrowserParam *p_dsy;
    coFileBrowserParam *p_thp;
    coFloatParam *p_scale;
    coIntVectorParam *p_times;
    coChoiceParam *p_nodal_ch[NODAL_PORTS];
    coChoiceParam *p_cell_ch[CELL_PORTS];
    coBooleanParam *p_file;
    coChoiceParam *p_global_ch[GLOBAL_PORTS];
    // parameters for tensors
    coChoiceParam *p_Tport;
    coChoiceParam *p_Tcomponents[TENSOR_PORTS][9];
    // Output ports
    coOutputPort *p_grid;
    coOutputPort *p_nodal_obj[NODAL_PORTS];
    coOutputPort *p_cell_obj[CELL_PORTS];
    coOutputPort *p_global_obj[GLOBAL_PORTS];
    coOutputPort *p_materials;
    coOutputPort *p_elementL;
#ifdef _LOCAL_REFERENCES_
    coOutputPort *p_references;
#endif
    coOutputPort *p_tensor_obj[TENSOR_PORTS];

    int noStates_;

    int fillDescriptions(TensDescriptions &);
    int isNull(int);
    int isS2D(int);
    int isS3D(int);
    int isF2D(int);
    int isF3D(int);

    void setTensorObj(coDoSet **, TensDescriptions &);

    // use for tensor component options
    whichContents cell_contents_old_[TENSOR_PORTS];

    ReadDSY readDSY;
    void postInst();

public:
    ReadPam(int argc, char *argv[]);
    virtual ~ReadPam();
};
#endif
