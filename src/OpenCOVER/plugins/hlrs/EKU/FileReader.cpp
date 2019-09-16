 /* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "FileReader.hpp"

/*
FileReaderWriter::FileReaderWriter()
{
    scene=osgDB::readNodeFile( "/home/AD.EKUPD.COM/matthias.epple/data/osgt/test.osgt" );

    if (!scene.valid())
      {
          osg::notify( osg::FATAL ) << "Unable to load data file. Exiting." << std::endl;
          //return( 1 );
      }

    // Find the node who's name is "Flat".
        fnn=new FindNamedNode( "Flat" );
        scene->accept( *fnn );
        if (fnn->getNode() != NULL)
        {
          std::string test = fnn->getName();
          auto position=  fnn->getNode()->getBound().center();


            // We found the node. Get the ShadeModel attribute
            //   from its StateSet and set it to SMOOTH shading.
            osg::StateSet* ss = fnn.getNode()->getOrCreateStateSet();
            osg::ShadeModel* sm = dynamic_cast<osg::ShadeModel*>(
                    ss->getAttribute(
                            osg::StateAttribute::SHADEMODEL ) );
            sm->setMode( osg::ShadeModel::SMOOTH );
        }
}
*/
