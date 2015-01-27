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
#include "TestUIF.h"
#include "api/coFeedback.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TestUIF::TestUIF(int argc, char *argv[])
    : coModule(argc, argv, "TestUIF Program: Show all parameter types")
{
    // Timer parameter
    timer = addTimerParam("timer", "Timer Parameter");

    // Immediate-mode String parameter
    stringImm = addStringParam("stringImm", "Immediate string");
    stringImm->setValue("This is an immediate String Parameter");

    // Non-immediate String parameter: delivers new value only at compute()
    stringCom = addStringParam("stringCom", "Compute string");

    // Declare a switching part of parameters
    choMaster = paraSwitch("top_switch", "Now switch Parameters");

    // First switch possibility: Boolean parameters
    paraCase("Boolean Parameters");

    // Immediate-mode Boolean parameter, pre-set to FALSE
    boolImm = addBooleanParam("boolImm", "Immediate coBooleanParam");
    boolImm->setValue(0);

    // Non-immediate Boolean parameter, pre-set to TRUE
    boolCom = addBooleanParam("boolCom", "Compute coBooleanParam");
    boolCom->setValue(1);

    // the 'boolean' switch case is over here...
    paraEndCase();

    // Next switch possibility: Scalar parameters
    paraCase("Scalar");

    // second level of switching
    paraSwitch("ScalarSw", "Sub-Cases of Scalar");

    // one case
    paraCase("Int Scalars");
    iScalImm = addInt32Param("iScalImm", "Immediate coIntScalarParam");
    iScalCom = addInt32Param("iScalCom", "Compute coIntScalarParam");
    paraEndCase();

    paraCase("Float");
    fScalImm = addFloatParam("fScalImm", "Immediate coFloatParam");
    fScalCom = addFloatParam("fScalCom", "Compute coFloatParam");
    paraEndCase();

    // second level of switching ends here
    paraEndSwitch();

    // The 'Scalar' case ends here
    paraEndCase();

    // Next switch possibility: Slider parameters
    paraCase("Sliders");

    // integer sliders: immediate and non-immediate
    iSlidImm = addIntSliderParam("iSlidImm", "Immediate coIntSliderParam");
    iSlidImm->setValue(1, 27, 16);

    iSlidCom = addIntSliderParam("iSlidCom", "Compute coIntSliderParam");
    iSlidImm->setMax(32);

    // float sliders: immediate and non-immediate
    fSlidImm = addFloatSliderParam("fSlidImm", "Immediate coFloatSliderParam");
    fSlidImm->setValue(0, 3, 5);

    fSlidCom = addFloatSliderParam("fSlidCom", "Compute coFloatSliderParam");
    fSlidCom->setMax(10);

    paraEndCase();

    paraCase("Vectors");

    // float vector: use default size of 3 and set with 3D setValue function
    fVectImm = addFloatVectorParam("fVectImm", "Immediate coFloatVectorParam");
    fVectImm->setValue(0, 3, 5);

    // float vector: pre-set with own array and non-default size
    fVectCom = addFloatVectorParam("fVectCom", "Compute coFloatVectorParam");
    float preset[] = { 1.34f, 1.889f };
    fVectCom->setValue(2, preset);

    // int vector: use default size of 3 and set with 3D setValue function
    iVectImm = addInt32VectorParam("iVectImm", "Immediate coIntVectorParam");
    iVectImm->setValue(11, 22, 33);

    // int vector: pre-set with own array and non-default size
    iVectCom = addInt32VectorParam("iVectCom", "Compute coIntVectorParam");
    long presetInt[] = { 111, 222 };
    iVectCom->setValue(2, presetInt);

    paraEndCase();

    paraEndSwitch();

    // it makes no sense to put a file selector in the switch, since
    // it is not displayed in the control panel
    browseImm = addFileBrowserParam("myFile", "a file browser");
    browseImm->setValue("/var/tmp/whatever.txt", "*.txt");
    browseImm->show();

    // Now this is a choice : we have the choice between 6 values
    const char *choiceVal[] = {
        "left lower inlet", "left upper inlet",
        "left center inlet", "right center inlet",
        "right lower Inlet", "right upper Inlet"
    };
    choImm = addChoiceParam("choImm", "Nu auch noch Choices");
    choImm->setValue(6, choiceVal, 2);

    // add an output port for this type
    outPort = addOutputPort("outPort", "Polygons", "Output Port");

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

int TestUIF::compute(const char *port)

{
    (void)port;
    //cerr << "\n ------- COMPUTE" << endl;

    // now we create an object for the output port: get the name and make the Obj
    const char *outObjName = outPort->getObjName();

    // make a little shape
    float xc[] = { 0.0, 0.0, 1.0, 1.0 };
    float yc[] = { 0.0, 1.0, 1.0, 0.0 };
    float zc[] = { 0.0, 0.5, 1.0, 0.5 };
    int elem[] = { 0 };
    int conn[] = { 0, 1, 2, 3 };

    // create object for Module Feedback
    coDoPolygons *poly = new coDoPolygons(outObjName, 4, xc, yc, zc, 4, conn, 1, elem);
    coFeedback feed("cpFeedback-Test");
    feed.addPara(boolImm);
    feed.addPara(iScalImm);
    feed.addPara(fScalImm);
    feed.addPara(iSlidImm);
    feed.addPara(fSlidImm);
    feed.addPara(fVectImm);
    feed.addPara(iVectImm);
    feed.addPara(choImm);
    feed.addPara(stringImm);
    feed.addString("User string #1");
    feed.addString("User string #2");
    feed.addString("This uses the masking feature!?");
    feed.apply(poly);

    // tell the output port that this is his object
    outPort->setCurrentObject(poly);

    // ... do whatever you need with in- or output objects,
    // BUT: do NOT delete them !!!!

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

void TestUIF::param(const char *name, bool /*inMapLoading*/)
{
    //cerr << "\n ------- Parameter Callback for '"
    //     << name
    //     << "'" << endl;

    // check whether this was the 'boolImm' parameter
    /*if (strcmp(name,boolImm->getName())==0)
      cerr << "The immediate bool parameter arrived with the value "
           << ( (boolImm->getValue()) ? "TRUE" : "FALSE" )
           << endl;*/

    // if not, it might be the float slider...
    if (strcmp(name, fSlidImm->getName()) == 0)
    {
        float minV, maxV, value; // guess why not min,max ??
        // -> some people use macros overloading these ;-)
        fSlidImm->getValue(minV, maxV, value);
        /*cerr << "The immediate float slider parameter has the values "
           << " Min=" << minV << " Max=" << minV << " Value=" << value
           << endl;*/
    }

    /// But: we can always access ALL port variables, only that the
    ///      non-immediate are only updated with every compute() call
    //cerr << *fSlidCom << endl;
    //cerr << *fSlidImm << endl;

    /// and we are always allowed to change a parameter's value
    fScalCom->setValue(12345.678f);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Quit callback: as the name tells...
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void TestUIF::quit()
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  postInst() is called once after we contacted Covise, but before
// ++++             getting into the main loop
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void TestUIF::postInst()
{
    //cerr << "after Contruction" << endl;
}

MODULE_MAIN(Examples, TestUIF)
