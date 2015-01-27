/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                                          **
 **                                                                          **
 ** Description: Assemble blocks of unstructured grids( per timestep)        **
 **                                                                          **
 ** Name:        TextIv                                                 **
 ** Category:    Tools                                                       **
 **                                                                          **
 ** Author: Sven Kufer		                                            **
 **         (C)  VirCinity IT- Consulting GmbH                               **
 **         Nobelstrasse 15                               		    **
 **         D- 70569 Stuttgart    			       		    **
 **                                                                          **
 **  19.02.2001                                                              **
\****************************************************************************/

#include "TextIv.h"
#include <stdlib.h>
#include <stdio.h>

void main(int argc, char *argv[])
{
    TextIv *application = new TextIv;
    application->start(argc, argv);
}

TextIv::TextIv() //  :coModule("")
{
    p_textOut = addOutputPort("TextOut", "coDoText", "text");
}

int TextIv::compute()
{
    static char *text_begin = "Separator {\n"
                              "    LightModel { model BASE_COLOR }\n"
                              "    MaterialBinding { value PER_FACE }\n"
                              "    Separator {\n"
                              "        BaseColor{ rgb 1 0 0  }\n"
                              "        Translation { translation 0 0.01 0 }\n"
                              "        Text2 { string ";
    static char *text_end = " }\n"
                            "    }\n"
                            "}\n";
    char *my_text = "Hallo Juergen";

    coDoText *text_obj = new coDoText(p_textOut->getObjName(), strlen(text_begin) + strlen(my_text) + strlen(text_end) + 3);
    char *text_pt;
    text_obj->getAddress(&text_pt);

    sprintf(text_pt, "%s\"%s\"%s", text_begin, my_text, text_end);

    p_textOut->setCurrentObject(text_obj);
    return CONTINUE_PIPELINE;
}
