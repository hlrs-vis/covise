/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *  (C) 1996-2020                                                       *
 *  High Performance Computer Centre Stuttgart    (HLRS)                *
 *  Nobelstrasse 19                                                     *
 *  D-70550 Stuttgart                                                   *
 *  Germany                                                             *
 *                                                                      *
 *	Description		center Paged LOD coordinates                        *
 *                                                                      *
 *	Author			Uwe Woessner                                        *
 *                                                                      *
 *                                                                      *
 ************************************************************************/
#include <stdio.h>

#include <osg/ArgumentParser>
#include <osg/ApplicationUsage>
#include <osg/Group>
#include <osg/Notify>
#include <osg/Vec3>
#include <osg/ProxyNode>
#include <osg/Geometry>
#include <osg/Texture2D>
#include <osg/Texture3D>
#include <osg/BlendFunc>
#include <osg/Timer>
#include <osg/PagedLOD>
#include <osg/MatrixTransform>

#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileNameUtils>
#include <osgDB/ReaderWriter>
#include <osgDB/PluginQuery>

#include <osgUtil/Optimizer>
#include <osgUtil/Simplifier>
#include <osgUtil/SmoothingVisitor>

#include <osgViewer/GraphicsWindow>
#include <osgViewer/Version>

#include <iostream>

typedef std::vector<std::string> FileNameList;

typedef std::vector< osg::ref_ptr<osg::Node> > Nodes;
typedef std::vector< osg::ref_ptr<osg::Object> > Objects;

void processNode(osg::Transform* t, osg::Node* node);

osg::BoundingSphere bs;


static void usage(const char* prog, const char* msg)
{
    if (msg)
    {
        osg::notify(osg::NOTICE) << std::endl;
        osg::notify(osg::NOTICE) << msg << std::endl;
    }

    // basic usage
    osg::notify(osg::NOTICE) << std::endl;
    osg::notify(osg::NOTICE) << "usage:" << std::endl;
    osg::notify(osg::NOTICE) << "    " << prog << " [options] infile1 outfile" << std::endl;
    osg::notify(osg::NOTICE) << std::endl;

}

void convertGeometry(osg::Geometry* geometry)
{

    osg::Vec3Array* verts3 = dynamic_cast<osg::Vec3Array*>(geometry->getVertexArray());
    if (verts3)
    {
        for (unsigned int j = 0; j < verts3->size(); j++)
        {
            (*verts3)[j] = (*verts3)[j] - bs.center();
        }
    }
    else
    {
        osg::Vec4Array* verts = dynamic_cast<osg::Vec4Array*>(geometry->getVertexArray());
        if (verts)
        {
            for (unsigned int j = 0; j < verts->size(); j++)
            {
                (*verts)[j][0] = (*verts)[j][0] - bs.center()[0];
                (*verts)[j][1] = (*verts)[j][1] - bs.center()[1];
                (*verts)[j][2] = (*verts)[j][2] - bs.center()[2];
            }
        }
    }

}
void convertDrawable(osg::Drawable* drawable)
{
    osg::Geometry* geometry = drawable->asGeometry();

    osg::Vec3Array* verts3 = dynamic_cast<osg::Vec3Array*>(geometry->getVertexArray());
    if (verts3)
    {
        for (unsigned int j = 0; j < verts3->size(); j++)
        {
            (*verts3)[j] = (*verts3)[j] - bs.center();
        }
    }
    else
    {
        osg::Vec4Array* verts = dynamic_cast<osg::Vec4Array*>(geometry->getVertexArray());
        if (verts)
        {
            for (unsigned int j = 0; j < verts->size(); j++)
            {
                (*verts)[j][0] = (*verts)[j][0] - bs.center()[0];
                (*verts)[j][1] = (*verts)[j][1] - bs.center()[1];
                (*verts)[j][2] = (*verts)[j][2] - bs.center()[2];
            }
        }
    }

}

void convertGeode(osg::Geode * geode)
{
    for (unsigned int i = 0; i < geode->getNumChildren(); i++)
    {
        if (dynamic_cast<osg::Geometry*>(geode->getChild(i)))
            convertGeometry(dynamic_cast<osg::Geometry*>(geode->getChild(i)));
        else if (dynamic_cast<osg::Drawable*>(geode->getChild(i)))
            convertDrawable(dynamic_cast<osg::Drawable*>(geode->getChild(i)));
    }
}
void convertGroup(osg::Group* g)
{
    for (unsigned int i = 0; i < g->getNumChildren(); i++)
    {
        if (dynamic_cast<osg::Geometry*>(g->getChild(i)))
            convertGeometry(dynamic_cast<osg::Geometry*>(g->getChild(i)));
        else if (dynamic_cast<osg::Drawable*>(g->getChild(i)))
            convertDrawable(dynamic_cast<osg::Drawable*>(g->getChild(i)));
    }
}
void convertLOD(osg::PagedLOD* pLOD)
{
    pLOD->setCenter(pLOD->getCenter() - bs.center());
    for (unsigned int i = 0; i < pLOD->getNumChildren(); i++)
    {
        processNode(nullptr,pLOD->getChild(i));
    }
}
void processNode(osg::Transform* t,osg::Node* node)
{
    if (t)
    {
        t->addChild(node);
    }
    if (dynamic_cast<osg::PagedLOD*>(node))
    {
        convertLOD(dynamic_cast<osg::PagedLOD*>(node));
    }
    if (dynamic_cast<osg::Geode*>(node))
    {
        convertGeode(dynamic_cast<osg::Geode*>(node));
    }
    if (dynamic_cast<osg::Geometry*>(node))
    {
        convertGeometry(dynamic_cast<osg::Geometry*>(node));
    }
    if (dynamic_cast<osg::Group*>(node))
    {
        convertGroup(dynamic_cast<osg::Group*>(node));
    }
}

osg::MatrixTransform*convert(osg::Node* node)
{
    osg::MatrixTransform* t = new osg::MatrixTransform();
    t->setName("rootTransform");
    t->setMatrix(osg::Matrixd::translate(bs.center()));
    osg::Group* g = dynamic_cast<osg::Group*>(node);
    if (g)
    {
        for (unsigned int i = 0; i < g->getNumChildren(); i++)
        {
            processNode(t,g->getChild(i));
        }
    }
    else
    {
        processNode(t,node);
    }
    return t;

}

int main(int argc, char *argv[])
{
    // use an ArgumentParser object to manage the program arguments.
    osg::ArgumentParser arguments(&argc, argv);

    // set up the usage document, in case we need to print out how to use this program.
    arguments.getApplicationUsage()->setApplicationName(arguments.getApplicationName());
    arguments.getApplicationUsage()->setDescription(arguments.getApplicationName() + " is a utility for transforming the coordinate origin to the center of the object.");
    arguments.getApplicationUsage()->setCommandLineUsage(arguments.getApplicationName() + " [options] filenameIn filenameOut");
    arguments.getApplicationUsage()->addCommandLineOption("-h or --help", "Display command line parameters");
    arguments.getApplicationUsage()->addCommandLineOption("--help-env", "Display environmental variables available");
    // if user request help write it out to cout.
    if (arguments.read("-h") || arguments.read("--help"))
    {
        osg::setNotifyLevel(osg::NOTICE);
        usage(arguments.getApplicationName().c_str(), 0);
        //arguments.getApplicationUsage()->write(std::cout);
        return 1;
    }

    if (arguments.argc() <= 1)
    {
        arguments.getApplicationUsage()->write(std::cout, osg::ApplicationUsage::COMMAND_LINE_OPTION);
        return 1;
    }

    FileNameList fileNames;
    Nodes nodes;
    Objects objects;

    // any option left unread are converted into errors to write out later.
    arguments.reportRemainingOptionsAsUnrecognized();

    // report any errors if they have occurred when parsing the program arguments.
    if (arguments.errors())
    {
        arguments.writeErrorMessages(std::cout);
        return 1;
    }

    for (int pos = 1; pos < arguments.argc(); ++pos)
    {
        if (!arguments.isOption(pos))
        {
            fileNames.push_back(arguments[pos]);
        }
    }



    std::string fileNameOut("converted.osgt");
    if (fileNames.size() > 1)
    {
        fileNameOut = fileNames.back();
        fileNames.pop_back();
    }

    for (FileNameList::iterator itr = fileNames.begin();
        itr != fileNames.end();
        ++itr)
    {
        osg::ref_ptr<osg::Object> object = osgDB::readObjectFile(*itr);
        if (object.valid())
        {
            if (object->asNode()) nodes.push_back(object->asNode());
            else objects.push_back(object);
        }
    }

    osg::ref_ptr<osg::Node> root;

    if (nodes.size() == 1)
    {
        bs = nodes.front()->getBound();
        root = convert(nodes.front());
    }
    else if (nodes.size() > 1)
    {
        osg::ref_ptr<osg::Group> group = new osg::Group;
        for (Nodes::iterator itr = nodes.begin();
            itr != nodes.end();
            ++itr)
        {
            group->addChild(itr->get());
        }

        bs = group->getBound();
        root = convert(group);
    }


    osgDB::ReaderWriter::WriteResult result = osgDB::Registry::instance()->writeNode(*root, fileNameOut, osgDB::Registry::instance()->getOptions());
    if (result.success())
    {
        osg::notify(osg::NOTICE) << "Data written to '" << fileNameOut << "'." << std::endl;
    }
    else if (result.message().empty())
    {
        osg::notify(osg::NOTICE) << "Warning: file write to '" << fileNameOut << "' not supported." << std::endl;
    }
    else
    {
        osg::notify(osg::NOTICE) << result.message() << std::endl;
    }

    return 0;
}
