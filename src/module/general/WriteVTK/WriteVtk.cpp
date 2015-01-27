/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <sstream>
#include <vtkPointSet.h>
#include <vtkPolyData.h>
#include <vtkStructuredGrid.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSet.h>
#include <vtkDataSetAttributes.h>
#include <vtkPointData.h>
#include <vtkImageData.h>
#include <vtkDataSet.h>
#include <vtkInformation.h>
#include <vtkMultiPieceDataSet.h>
#include <do/coDoGeometry.h>
#include <do/coDoPixelImage.h>
#include <do/coDoTexture.h>
#include <do/coDoData.h>
#include <alg/coVtk.h>
#include "WriteVtk.h"

#undef DEBUG

#if VTK_MAJOR_VERSION > 5 || (VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION >= 8)
#define HAVE_COMPOSITE_WRITER
#endif

#ifdef HAVE_COMPOSITE_WRITER
#include <vtkCompositeDataWriter.h>
#endif

#if VTK_MAJOR_VERSION < 6
#define SetInputData SetInput
#endif

WriteVtk::WriteVtk(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Write any type of vtk dataset to file")
{
#define curport (m_inPorts.size() - 1)

    // ports
    char portname[1000], portdesc[1000];
    snprintf(portname, sizeof(portname), "GridIn0");
    snprintf(portdesc, sizeof(portdesc), "Grid input");
    m_inPorts.push_back(addInputPort(portname, "Geometry|Points|Lines|Polygons|TriangleStrips|StructuredGrid|UnstructuredGrid|UniformGrid|RectilinearGrid", portdesc));
    m_inPorts[curport]->setRequired(true);

    int ndata = 0;

    snprintf(portname, sizeof(portname), "DataIn%d", ndata);
    snprintf(portdesc, sizeof(portdesc), "Normals");
    m_inPorts.push_back(addInputPort(portname, "Vec3", portdesc));
    m_inPorts[curport]->setRequired(false);
    ++ndata;

    for (int i = 0; i < NumFields; ++i)
    {
        snprintf(portname, sizeof(portname), "DataIn%d", ndata);
        snprintf(portdesc, sizeof(portdesc), "Field %d", i);
        m_inPorts.push_back(addInputPort(portname, "Int|Float|Vec2|Vec3|RGBA", portdesc));
        m_inPorts[curport]->setRequired(false);
        ++ndata;
    }

    snprintf(portname, sizeof(portname), "TextureIn0");
    snprintf(portdesc, sizeof(portdesc), "Textures");
    m_inPorts.push_back(addInputPort(portname, "Texture|PixelImage|Float|Vec2|Vec3", portdesc));
    m_inPorts[curport]->setRequired(false);

#undef curport

    // params
    m_FileNameParam = addFileBrowserParam("FileName", "Specify file name of vtk polygon data file to write.");
    m_FileNameParam->setValue(".", "*.vtk/*");

    m_BinaryFileTypeParam = addBooleanParam("BinaryFileType", "Specify file type (ASCII or BINARY) for vtk data file. Binary may not be portable.");
    m_BinaryFileTypeParam->setValue(false);

    m_filePerTimeStep = addBooleanParam("FilePerTimeStep", "Write separate files for each time step");
    m_filePerTimeStep->setValue(true);

    m_NormalsNameParam = addStringParam("NormalsName", "Give a name to the normals data.");
    m_NormalsNameParam->setValue("normals");

    for (int i = 0; i < NumFields; ++i)
    {
        char buf[1024];
        snprintf(buf, sizeof(buf), "FieldName%d", i);
        m_FieldNameParam[i] = addStringParam(buf, "Give a name to the scalar data.");
        snprintf(buf, sizeof(buf), "scalars%d", i);
        m_FieldNameParam[i]->setValue(buf);
    }

    m_TCoordsNameParam = addStringParam("TCoordsName", "Give a name to the texture coordinates data.");
    m_TCoordsNameParam->setValue("textureCoords");
}

WriteVtk::~WriteVtk()
{
}

void WriteVtk::preHandleObjects(coInputPort **ports)
{
    m_savedInputPorts = NULL;

    if (!*ports)
        return;

    const coDoGeometry *geo = dynamic_cast<const coDoGeometry *>((*ports)->getCurrentObject());
    if (!geo)
        return;

    for (size_t i = 0; ports[i]; ++i)
    {
        coInputPort *p = ports[i];
        if (p->getCurrentObject() && i > 0)
            return;
        m_savedInput.push_back(p->getCurrentObject());
    }

    m_savedInputPorts = ports;

    std::cerr << "unpacking geometry" << std::endl;

    ports[0]->setCurrentObject(geo->getGeometry());
    ports[1]->setCurrentObject(geo->getNormals());
    ports[2]->setCurrentObject(geo->getColors());
    ports[3]->setCurrentObject(geo->getVertexAttribute());
    ports[NumFields]->setCurrentObject(geo->getTexture());
}

void WriteVtk::postHandleObjects(coOutputPort **)
{
    if (m_savedInputPorts)
    {
        for (size_t i = 0; i < m_savedInput.size(); ++i)
            m_savedInputPorts[i]->setCurrentObject(m_savedInput[i]);
    }
    m_savedInput.clear();

    m_collection.clear();
}

int WriteVtk::compute(const char *)
{
    std::vector<std::string> attribNames;
    for (int i = 0; i < NumFields; ++i)
    {
        std::string name(m_FieldNameParam[i]->getValue());
        if (std::find(attribNames.begin(), attribNames.end(), name) != attribNames.end())
        {
            sendError("duplicate scalar parameter name %s", name.c_str());
            return STOP_PIPELINE;
        }
        attribNames.push_back(name);
    }

    vtkDataSet *vtk = NULL;
    if (const coDoGeometry *geom = dynamic_cast<const coDoGeometry *>(m_inPorts[0]->getCurrentObject()))
    {
        vtk = coVtk::coviseGeometry2Vtk(geom);
        if (!vtk)
        {
            sendError("conversion to VTK format failed for input port %s", m_inPorts[0]->getName());
            return STOP_PIPELINE;
        }
    }
    else if (const coDoGrid *grid = dynamic_cast<const coDoGrid *>(m_inPorts[0]->getCurrentObject()))
    {
        std::vector<int> inputType, inputTexType;
        int portnum = 0;
        vtkDataSet *dataset = coVtk::covise2Vtk(m_inPorts[portnum]->getCurrentObject());
        coVtk::Flags flags = coVtk::None;
        if (dynamic_cast<vtkImageData *>(dataset))
            flags = coVtk::RequireDouble;
        int type = 0;
        if (dynamic_cast<const coDoTexture *>(m_inPorts[portnum]->getCurrentObject()))
            type = 1;
        if (dynamic_cast<const coDoPixelImage *>(m_inPorts[portnum]->getCurrentObject()))
            type = 2;
        inputType.push_back(type);
        if (!dataset && m_inPorts[portnum]->getCurrentObject())
        {
            sendError("conversion to VTK format failed for input port %s", m_inPorts[portnum]->getName());
            return STOP_PIPELINE;
        }
        vtk = dynamic_cast<vtkDataSet *>(dataset);
        if (!vtk)
        {
            dataset->Delete();
            sendError("conversion to required type vtkDataSet failed for input port %s", m_inPorts[portnum]->getName());
            return STOP_PIPELINE;
        }
        vtkDataSetAttributes *vattr = vtk->GetPointData();

        ++portnum;
        if (const coDoAbstractData *data = dynamic_cast<const coDoAbstractData *>(m_inPorts[portnum]->getCurrentObject()))
        {
            vtkDataArray *vdata = coVtk::coviseData2Vtk(grid, data, coVtk::Flags(flags | coVtk::Normalize));
            vattr->SetNormals(vdata);
        }

        for (int i = 0; i < NumFields; ++i)
        {
            ++portnum;
            if (const coDoAbstractData *data = dynamic_cast<const coDoAbstractData *>(m_inPorts[portnum]->getCurrentObject()))
            {
                vtkDataArray *vdata = coVtk::coviseData2Vtk(grid, data, flags);
                if (!vdata && m_inPorts[portnum]->getCurrentObject())
                {
                    sendError("conversion to VTK format failed for input port %s", m_inPorts[portnum]->getName());
                    return STOP_PIPELINE;
                }
                vdata->SetName(m_FieldNameParam[i]->getValue());
                vattr->AddArray(vdata);
                vattr->SetActiveScalars(m_FieldNameParam[i]->getValue());
            }
        }

        ++portnum;
        const coDistributedObject *texobj = m_inPorts[portnum]->getCurrentObject();
        int textype = 0;
        if (dynamic_cast<const coDoTexture *>(texobj))
            textype = 1;
        if (dynamic_cast<const coDoPixelImage *>(texobj))
            textype = 2;
        inputTexType.push_back(textype);
    }
    else
    {
        sendError("incompatible data type on input GridIn0");
        return STOP_PIPELINE;
    }

    if (vtk && vtk->CheckAttributes())
        sendInfo("VTK object integrity check failed");

    bool writeMultiPiece = false;
    vtkMultiPieceDataSet *mp = NULL;
    if (isPartOfMultiblock())
    {
#ifdef DEBUG
        std::cerr << "part " << getElementNumber() << " of " << getNumberOfElements() << " (level " << getObjectLevel() << ")" << std::endl;

#endif
        if (m_collection.size() > getObjectLevel())
            mp = m_collection[getObjectLevel()];
        else
            m_collection.resize(getObjectLevel() + 1);
        if (!mp)
        {
            mp = vtkMultiPieceDataSet::New();
            mp->SetNumberOfPieces(getNumberOfElements());
            m_collection[getObjectLevel()] = mp;
        }
        mp->SetPiece(getElementNumber(), vtk);

        if (getElementNumber() == getNumberOfElements() - 1)
            writeMultiPiece = true;
    }

    if (isTimestep() && (!mp || writeMultiPiece))
    {
        std::string filename = m_FileNameParam->getValue();
        std::stringstream s;
        if (filename.size() >= 4 && filename.substr(filename.size() - 4, std::string::npos) == ".vtk")
            filename = filename.substr(0, filename.size() - 4);
        s << filename << "." << getElementNumber() << ".vtk";
        if (mp)
        {
            write(mp, s.str().c_str());
            m_collection.clear();
        }
        else
            write(vtk, s.str().c_str());
    }
    else if (vtk)
    {
        write(vtk, m_FileNameParam->getValue());
    }

    return CONTINUE_PIPELINE;
}

void WriteVtk::write(vtkDataObject *obj, const char *filename)
{
    if (vtkDataSet *d = dynamic_cast<vtkDataSet *>(obj))
    {
        vtkDataSetWriter *m_vtkDataSetWriterInstance = vtkDataSetWriter::New();
        if (!m_vtkDataSetWriterInstance)
        {
            sendError("failed to create instance of \"vtkDataSetWriter\"");
            return;
        }

        m_vtkDataSetWriterInstance->SetFileName(filename);
        if (m_BinaryFileTypeParam->getValue())
            m_vtkDataSetWriterInstance->SetFileTypeToBinary();
        else
            m_vtkDataSetWriterInstance->SetFileTypeToASCII();
        m_vtkDataSetWriterInstance->SetNormalsName(m_NormalsNameParam->getValue());
        m_vtkDataSetWriterInstance->SetTCoordsName(m_TCoordsNameParam->getValue());

        m_vtkDataSetWriterInstance->SetInputData(0, d);
        m_vtkDataSetWriterInstance->Update();
        m_vtkDataSetWriterInstance->Delete();
    }
    else if (vtkCompositeDataSet *d = dynamic_cast<vtkCompositeDataSet *>(obj))
    {
#ifdef HAVE_COMPOSITE_WRITER
        vtkCompositeDataWriter *m_vtkDataSetWriterInstance = vtkCompositeDataWriter::New();
        if (!m_vtkDataSetWriterInstance)
        {
            sendError("failed to create instance of \"vtkCompositeDataWriter\"");
            return;
        }

        m_vtkDataSetWriterInstance->SetFileName(filename);
        if (m_BinaryFileTypeParam->getValue())
            m_vtkDataSetWriterInstance->SetFileTypeToBinary();
        else
            m_vtkDataSetWriterInstance->SetFileTypeToASCII();
        m_vtkDataSetWriterInstance->SetNormalsName(m_NormalsNameParam->getValue());
        m_vtkDataSetWriterInstance->SetTCoordsName(m_TCoordsNameParam->getValue());

        m_vtkDataSetWriterInstance->SetInputData(0, d);
        m_vtkDataSetWriterInstance->Update();
        m_vtkDataSetWriterInstance->Delete();
#else
        (void)d;
        sendError("sets require at least VTK 5.8 for vtkCompositeDataWriter");
#endif
    }
}

MODULE_MAIN(IO, WriteVtk)
