/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file factoryresultsfileconverter.hxx
 * a factory for results file parsers.
 */

#include "FactoryMeshFileTransParser.h" // a factory for results file parsers.
#include "MeshFileTransParserBinary.h" // a mesh file parser for changing (transient) meshes in Famu binary format.
#include "MeshFileTransParserHmAscii.h" // a mesh file parser for changing (transient) meshes in HyperMesh ASCII format.
#include "Tools.h" // some helpful tools.

MeshFileTransParser *FactoryMeshFileTransParser::create(
    std::string filename,
    OutputHandler *outputHandler)
{
    std::string suffix = parserTools::getSuffix(filename);
    MeshFileTransParser *retval = NULL;
    if (suffix == ".cvm")
    {
        retval = new MeshFileTransParserBinary(outputHandler);
    }
    else if (suffix == ".hmascii")
    {
        retval = new MeshFileTransParserHmAscii(outputHandler);
    }
    else
    {
        ERROR0("Error: unknown mesh file suffix.", outputHandler);
    }
    return retval;
}

// ----------------- singleton stuff --------------------
FactoryMeshFileTransParser *FactoryMeshFileTransParser::_instance = NULL;

FactoryMeshFileTransParser *FactoryMeshFileTransParser::getInstance()
{
    if (_instance == NULL)
    {
        _instance = new FactoryMeshFileTransParser();
    }
    return _instance;
}

void FactoryMeshFileTransParser::deleteInstance(void)
{
    if (_instance != NULL)
    {
        delete _instance;
        _instance = NULL;
    }
}
