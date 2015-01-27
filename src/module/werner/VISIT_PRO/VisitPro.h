/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _STAR_GEOMETRY_H_
#define _STAR_GEOMETRY_H_
/**************************************************************************\ 
 **                                                     (C)2002 VirCinity  **
 **                                                                        **
 ** Description: Read Star Geometry for DC Simulation Coupling             **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** History:                                                               **
 **      07/2002    A. Werner      Initial version                         **
 *\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <api/coModule.h>
using namespace covise;

class AddGeom;
class Attachable;
class MultiFileParam;

class VisitPro : public coModule
{

private:
    enum
    {
        MAX_BASE_FILES = 256,
        MAX_ADD_GEOM = 16,
        MAX_ATTACHABLES = 16
    };

    //////////////////////////////////////// basic member functions
    virtual int compute();
    virtual void param(const char *pname);
    virtual void postInst();

    //////////////////////////////////////// Parameters

    // Input files - also directories for add. parts
    MultiFileParam *d_baseGeo;
    MultiFileParam *d_addPart;

    // Mode Switch
    coBooleanParam *p_createGrid;

    // Choice which type if attached
    coChoiceParam *p_name[MAX_ADD_GEOM];

    // Unit in Choice files
    coChoiceParam *d_objUnit;

    // Ports
    coOutputPort *p_baseOut;
    coOutputPort *p_attachOut;
    coOutputPort *p_starOut;

    // Utility functions

    // create the base geometry at the output port
    void createBaseGeo();

    // update information for additional parts
    void updateAttachables();

    // create attached objects
    void createAttachedGeo();

    // Create Star coupling module steering object
    void createStarObj();

    // Create Star coupling module dummy object for NO EXEC
    void createNoExecObj();

protected:
    // this class contains all info about attached channels
    AddGeom *d_addGeo[MAX_ADD_GEOM];

    // this array contains information about all attachable objects
    Attachable *d_attachable[MAX_ATTACHABLES];

    // check whether already initialized Attachables - start from Map
    bool d_attachablesInitialized;

public:
    //////////////////////////////////////// Utility funcs
    // read file into Object objname_id
    static coDistributedObject *readOBJ(const char *filename,
                                        const char *objName,
                                        int id, const char *unit);

    VisitPro();
    virtual ~VisitPro();
};
#endif
