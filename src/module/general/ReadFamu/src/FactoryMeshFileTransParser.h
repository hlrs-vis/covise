/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file factoryresultsfileconverter.hxx
 * a factory for results file parsers.
 */

// #include "FactoryMeshFileTransParser.h" // a factory for results file parsers.

#ifndef __FactoryMeshFileTransParser_h__
#define __FactoryMeshFileTransParser_h__

#include "MeshFileTransParser.h" // a results file parser.
#include "OutputHandler.h" // an output handler for displaying information on the screen.

/** 
 * a factory for results file converters.
 */
class FactoryMeshFileTransParser
{
public:
    static FactoryMeshFileTransParser *getInstance(void);
    static void deleteInstance(void);

    virtual MeshFileTransParser *create(
        std::string filename,
        OutputHandler *outputHandler);

private:
    static FactoryMeshFileTransParser *_instance;
    FactoryMeshFileTransParser(void){};
    virtual ~FactoryMeshFileTransParser()
    {
        _instance = NULL;
    };
};

#endif // __factoryresultsfileconverter_hxx__
