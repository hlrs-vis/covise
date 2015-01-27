/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1999 RUS  **
 ** Example module for Covise API 2.0 User-interface functions             **
 **                                                                        **
 ** Author:                                                                **
 **                             Andreas Werner                             **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 ** Date:  23.09.99  V1.0                                                  **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#endif

#include <util/coviseCompat.h>
#include <do/coDoUnstructuredGrid.h>
#include "ParamTest.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ParamTest::ParamTest(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Param Program: Show all parameter types")
{
    //autoInitParam(0);

    // Immediate-mode String parameter
    stringImm = addStringParam("stringImm", "Immediate string");
    stringImm->setValue("This is an immediate String Parameter");

    // Immediate-mode Boolean parameter, pre-set to FALSE
    boolImm = addBooleanParam("boolImm", "Immediate coBooleanParam");
    boolImm->setValue(0);

    iScalImm = addInt32Param("iScalImm", "Immediate coIntScalarParam");
    iScalImm->setValue(123);

    fScalImm = addFloatParam("fScalImm", "Immediate coFloatParam");
    fScalImm->setValue(-12.56f);

    // integer sliders: immediate and non-immediate
    iSlidImm = addIntSliderParam("iSlidImm", "Immediate coIntSliderParam");
    iSlidImm->setValue(1, 27, 16);

    // float sliders: immediate and non-immediate
    fSlidImm = addFloatSliderParam("fSlidImm", "Immediate coFloatSliderParam");
    fSlidImm->setValue(-10.0, 30.0, 0.0);

    // float vector: use default size of 3 and set with 3D setValue function
    fVectImm = addFloatVectorParam("fVectImm", "Immediate coFloatVectorParam");
    fVectImm->setValue(1.34f, 1.889f, -99.87f);

    // it makes no sense to put a file selector in the switch, since
    // it is not displayed in the control panel
    browseImm = addFileBrowserParam("myFile", "a file browser");
    browseImm->setValue("/var/tmp/whatever.txt", "*.txt");
    browseImm->show();

    browse = addFileBrowserParam("my2File", "a file browser");
    browse->setValue("/var/tmp/whatever2.txt", "*.txt");
    browse->show();

    // Now this is a choice : we have the choice between 6 values
    const char *choiceVal[] = {
        "left lower inlet", "left upper inlet",
        "left center inlet", "right center inlet",
        "right lower Inlet", "right upper Inlet"
    };
    choImm = addChoiceParam("choImm", "Nun auch noch Choices");
    choImm->setValue(6, choiceVal, 1);

    // add an input port for 'coDoUnstructuredGrid' objects
    inPortReq = addInputPort("inputReq", "StructuredGrid", "Required input port");

    // add another input port for 'coDoUnstructuredGrid' objects
    inPortNoReq = addInputPort("inputNoReq", "UnstructuredGrid", "Not required input port");

    // tell that this port does not have to be connected
    inPortNoReq->setRequired(0);

    // add an output port for this type
    outPort = addOutputPort("outPort", "coDoUnstructuredGrid", "Output Port");

    // and that's all ... no init() or anything else ... that's done in the lib
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Compute callback: Called when the module is executed
// ++++
// ++++  NEVER use input/output ports or distributed objects anywhere
// ++++        else than inside this function
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ParamTest::compute(const char *port)

{
    (void)port;
    // we send an error message
    coModule::sendError("This is an error message ");

    // we send an info message
    coModule::sendInfo("This is an information message ");

    // we send an warning message
    coModule::sendWarning("This is an warning message ");

    // set a ne title
    coModule::setTitle("New title");

    cerr << "\n ------- COMPUTE" << endl;

    const char *name = browse->getValue();
    cerr << "The normal browser parameter has the value "
         << " Name = " << name << endl;
    const char *name2 = browseImm->getValue();
    cerr << "The normal browser parameter has the value "
         << " Name = " << name2 << endl;

    // here we try to retrieve the data object from the required port
    const coDistributedObject *reqObj = inPortReq->getCurrentObject();
    if (reqObj)
        cerr << "req: " << reqObj->getType() << endl;
    else
    {
        cerr << "req: NULL" << endl;
        return FAIL;
    }

    // here we try to retrieve the data object from the not required port
    const coDistributedObject *volObj = inPortNoReq->getCurrentObject();
    if (volObj)
        cerr << "vol: " << volObj->getType() << endl;
    else
    {
        cerr << "vol: NULL" << endl;
        return FAIL;
    }

    // now we create an object for the output port: get the name and make the Obj
    const char *outObjName = outPort->getObjName();
    coDoUnstructuredGrid *usg = new coDoUnstructuredGrid(outObjName, 5, 6, 7, 1);

    // ... do whatever you like with in- or output objects,
    // BUT: do NOT delete them !!!!

    // tell the output port that this is his object
    outPort->setCurrentObject(usg);

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Parameter callback: This one is called whenever an immediate
// ++++                      mode parameter is changed, but NOT for
// ++++                      non-immediate ports
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ParamTest::param(const char *name, bool /*inMapLoading*/)
{
    cerr << "\n ------- Parameter Callback for '"
         << name
         << "'" << endl;

    // check whether this was the 'boolImm' parameter
    if (strcmp(name, boolImm->getName()) == 0)
        cerr << "The immediate bool parameter arrived with the value "
             << ((boolImm->getValue()) ? "TRUE" : "FALSE")
             << endl;

    // if not, it might be the float slider...
    else if (strcmp(name, fSlidImm->getName()) == 0)
    {
        float minV, maxV, value; // guess why not min,max ??
        // -> some people use macros overloading these ;-)
        fSlidImm->getValue(minV, maxV, value);
        cerr << "The immediate float slider parameter has the values "
             << " Min=" << minV << " Max=" << minV << " Value=" << value
             << endl;
    }

    // if not, it might be the float slider...
    else if (strcmp(name, browseImm->getName()) == 0)
    {
        const char *name = browseImm->getValue();
        cerr << "The immediate browserparameter has the value "
             << " Name = " << name << endl;
    }

    else if (strcmp(name, choImm->getName()) == 0)
    {
        const char *val = choImm->getActLabel();
        if (val)
        {
            cerr << "The immediate choiceparameter has the value "
                 << "'" << val << "'" << endl;
        }
        else
        {
            cerr << "The immediate choiceparameter has the value "
                 << "null" << endl;
        }
    }

    /// and we are always allowed to change a parameter's value
    fScalImm->setValue(12345.678f);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Quit callback: as the name tells...
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ParamTest::quit()
{
    cerr << "Param test ended..." << endl;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  postInst() is called once after we contacted Covise, but before
// ++++             getting into the main loop
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ParamTest::postInst()
{
    cerr << "after Contruction" << endl;
}

MODULE_MAIN(Examples, ParamTest)
