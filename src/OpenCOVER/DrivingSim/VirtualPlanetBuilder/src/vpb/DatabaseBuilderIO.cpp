/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <vpb/DatabaseBuilder>

#include <iostream>
#include <string>
#include <map>

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/io_utils>

#include <osgDB/ReadFile>
#include <osgDB/Registry>
#include <osgDB/Input>
#include <osgDB/Output>
#include <osgDB/ParameterOutput>

#include <osgDB/FileNameUtils>
#include <osgDB/FileUtils>

using namespace vpb;


//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  DatabaseBuilder IO support
//

bool DatabaseBuilder_readLocalData(osg::Object &obj, osgDB::Input &fr);
bool DatabaseBuilder_writeLocalData(const osg::Object &obj, osgDB::Output &fw);

osgDB::RegisterDotOsgWrapperProxy DatabaseBuilder_Proxy
(
    new vpb::DatabaseBuilder,
    "DatabaseBuilder",
    "DatabaseBuilder Object",
    DatabaseBuilder_readLocalData,
    DatabaseBuilder_writeLocalData
);


bool DatabaseBuilder_readLocalData(osg::Object& obj, osgDB::Input &fr)
{
    vpb::DatabaseBuilder& db = static_cast<vpb::DatabaseBuilder&>(obj);
    bool itrAdvanced = false;
    
    osg::ref_ptr<osg::Object> readObject = fr.readObjectOfType(osgDB::type_wrapper<BuildOptions>());
    if (readObject.valid())
    {
        db.setBuildOptions(dynamic_cast<BuildOptions*>(readObject.get()));
    }
    
    return itrAdvanced;
}

bool DatabaseBuilder_writeLocalData(const osg::Object& obj, osgDB::Output& fw)
{
    const vpb::DatabaseBuilder& db = static_cast<const vpb::DatabaseBuilder&>(obj);

    if (db.getBuildOptions())
    {
        fw.writeObject(*db.getBuildOptions());
    }


    return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
//
// New wrappers
//

#include <osgDB/ObjectWrapper>
#include <osgDB/InputStream>
#include <osgDB/OutputStream>

REGISTER_OBJECT_WRAPPER( DatabaseBuilder,
                         new vpb::DatabaseBuilder,
                         vpb::DatabaseBuilder,
                         "osg::Object osgTerrain::TerrainTechnique vpb::DatabaseBuilder" )
{
    ADD_OBJECT_SERIALIZER( BuildOptions, vpb::BuildOptions, NULL );
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  VPBReaderWriter
//
                        

class VPBReaderWriter : public osgDB::ReaderWriter
{
    public:
    
        VPBReaderWriter()
        {
            supportsExtension("vpb","VirtualPlanetBuilder source format");
            supportsExtension("source","VirtualPlanetBuilder source format");
        }
        
        virtual const char* className() const { return "VPB Reader/Writer"; }

        virtual bool acceptsExtension(const std::string& extension) const
        {
            return osgDB::equalCaseInsensitive(extension,"vpb") || osgDB::equalCaseInsensitive(extension,"source");
        }

        virtual ReadResult readNode(const std::string& file, const Options* opt) const
        {
            OSG_INFO<<"VPBReaderWriter::readNode()"<<std::endl;

            std::string ext = osgDB::getFileExtension(file);
            if (!acceptsExtension(ext)) return ReadResult::FILE_NOT_HANDLED;

            std::string fileName = osgDB::findDataFile( file, opt );
            if (fileName.empty()) return ReadResult::FILE_NOT_FOUND;

            // code for setting up the database path so that internally referenced file are searched for on relative paths.
            osg::ref_ptr<Options> local_opt = opt ? static_cast<Options*>(opt->clone(osg::CopyOp::SHALLOW_COPY)) : new Options;
            local_opt->setDatabasePath(osgDB::getFilePath(fileName));

            osgDB::ifstream fin(fileName.c_str(), std::ios::in);
            if (fin)
            {
                std::string str;
                fin >> str;

                fin.seekg(0);
                if (str=="#Ascii")
                {
                    local_opt->setPluginStringData( "fileType", "Ascii" );
                    return readNode_new(fin, local_opt.get());
                }
                else
                {
                    return readNode_old(fin, local_opt.get());
                }
            }
            return ReadResult::ERROR_IN_READING_FILE;
        }

        virtual ReadResult readNode_new(std::istream& fin, const Options* options) const
        {
            OSG_INFO<<"readNode_new()"<<std::endl;

            osgDB::ReaderWriter* rw = osgDB::Registry::instance()->getReaderWriterForExtension("osg2");
            if (!rw) return ReadResult::FILE_NOT_HANDLED;

            OSG_INFO<<"   found ReaderWriter, readNode_new()"<<std::endl;

            return rw->readNode( fin, options );
        }

        virtual ReadResult readNode_old(std::istream& fin, const Options* options) const
        {
            OSG_INFO<<"readNode_old()"<<std::endl;
            fin.imbue(std::locale::classic());

            osgDB::Input fr;
            fr.attach(&fin);
            fr.setOptions(options);
            
            typedef std::vector<osg::Node*> NodeList;
            NodeList nodeList;

            // load all nodes in file, placing them in a group.
            while(!fr.eof())
            {
                osg::Node *node = fr.readNode();
                if (node) nodeList.push_back(node);
                else fr.advanceOverCurrentFieldOrBlock();
            }

            if  (nodeList.empty())
            {
                return ReadResult("No data loaded");
            }
            else if (nodeList.size()==1)
            {
                return nodeList.front();
            }
            else
            {
                osg::Group* group = new osg::Group;
                group->setName("import group");
                for(NodeList::iterator itr=nodeList.begin();
                    itr!=nodeList.end();
                    ++itr)
                {
                    group->addChild(*itr);
                }
                return group;
            }
        }

        Options* prepareWriting( WriteResult& result, const std::string& fileName, const Options* options ) const
        {
            std::string ext = osgDB::getFileExtension( fileName );
            if ( !acceptsExtension(ext) ) result = WriteResult::FILE_NOT_HANDLED;

            osg::ref_ptr<Options> local_opt = options ?
                static_cast<Options*>(options->clone(osg::CopyOp::SHALLOW_COPY)) : new Options;
            local_opt->getDatabasePathList().push_front(osgDB::getFilePath(fileName));
            if ( ext=="osgt" || ext=="source" || ext=="vpb" ) local_opt->setPluginStringData( "fileType", "Ascii" );
            if ( ext=="osgx" ) local_opt->setPluginStringData( "fileType", "XML" );
            
            return local_opt.release();
        }

        virtual WriteResult writeNode(const osg::Node& node,const std::string& fileName, const osgDB::ReaderWriter::Options* options) const
        {
            return writeNode_new(node, fileName, options);
        }

        virtual WriteResult writeNode_new(const osg::Node& node,const std::string& fileName, const osgDB::ReaderWriter::Options* options) const
        {
            WriteResult result = WriteResult::FILE_SAVED;
            osg::ref_ptr<Options> local_opt = prepareWriting( result, fileName, options );
            if ( !result.success() ) return result;

            osgDB::ofstream fout( fileName.c_str(), std::ios::out );
            if ( !fout ) return WriteResult::ERROR_IN_WRITING_FILE;

            osgDB::ReaderWriter* rw = osgDB::Registry::instance()->getReaderWriterForExtension("osg2");
            if (!rw) return WriteResult::FILE_NOT_HANDLED;

            result = rw->writeNode( node, fout, local_opt.get() );

            return result;
        }

        virtual WriteResult writeNode_old(const osg::Node& node,const std::string& fileName, const osgDB::ReaderWriter::Options* options) const
        {
            std::string ext = osgDB::getFileExtension(fileName);
            if (!acceptsExtension(ext)) return WriteResult::FILE_NOT_HANDLED;

            osgDB::Output fout(fileName.c_str());
            if (fout)
            {
                fout.setOptions(options);

                fout.imbue(std::locale::classic());

                // setPrecision(fout,options);

                fout.writeObject(node);
                fout.close();
                return WriteResult::FILE_SAVED;
            }
            return WriteResult("Unable to open file for output");
        }

};

// now register with Registry to instantiate the above
// reader/writer.
REGISTER_OSGPLUGIN(vpb, VPBReaderWriter)

