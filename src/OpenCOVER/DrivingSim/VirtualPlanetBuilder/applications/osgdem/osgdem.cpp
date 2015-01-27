/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
 *
 * This application is open source and may be redistributed and/or modified
 * freely and without restriction, both in commericial and non commericial applications,
 * as long as this copyright notice is maintained.
 * 
 * This application is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <vpb/Commandline>
#include <vpb/DataSet>
#include <vpb/DatabaseBuilder>
#include <vpb/System>
#include <vpb/Version>
#include <vpb/FileUtils>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

#include <iostream>
#include <vpb/VPBRoad>

int main(int argc, char **argv)
{
    osg::Timer_t startTick = osg::Timer::instance()->tick();

    osg::ArgumentParser arguments(&argc, argv);

    // set up the usage document, in case we need to print out how to use this program.
    arguments.getApplicationUsage()->setApplicationName(arguments.getApplicationName());
    arguments.getApplicationUsage()->setDescription(arguments.getApplicationName() + " application is utility tools which can be used to generate paged geospatial terrain databases.");
    arguments.getApplicationUsage()->setCommandLineUsage(arguments.getApplicationName() + " [options] filename ...");
    arguments.getApplicationUsage()->addCommandLineOption("-h or --help", "Display this information");
    arguments.getApplicationUsage()->addCommandLineOption("--version", "Display version information");
    arguments.getApplicationUsage()->addCommandLineOption("--cache <filename>", "Read the cache file to use a look up for locally cached files.");

    vpb::Commandline commandline;

    commandline.getUsage(*arguments.getApplicationUsage());

    if (arguments.read("--version"))
    {
        std::cout << "VirtualPlanetBuilder/osgdem version " << vpbGetVersion() << std::endl;
        return 0;
    }

    if (arguments.read("--version-number"))
    {
        std::cout << vpbGetVersion() << std::endl;
        return 0;
    }

    std::string runPath;
    if (arguments.read("--run-path", runPath))
    {
        vpb::chdir(runPath.c_str());
    }

    // if user requests help write it out to cout.
    if (arguments.read("-h") || arguments.read("--help"))
    {
        arguments.getApplicationUsage()->write(std::cout, osg::ApplicationUsage::COMMAND_LINE_OPTION);
        return 1;
    }

    // if user requests list of supported formats write it out to cout.
    if (arguments.read("--formats"))
    {

        std::cout << "Supported formats:" << std::endl;
        const vpb::System::SupportedExtensions &extensions = vpb::System::instance()->getSupportExtensions();
        for (vpb::System::SupportedExtensions::const_iterator itr = extensions.begin();
             itr != extensions.end();
             ++itr)
        {
            std::cout << "  " << itr->first << " :";
            bool first = true;
            if (itr->second.acceptedTypeMask & vpb::Source::IMAGE)
            {
                std::cout << " imagery";
                first = false;
            }
            if (itr->second.acceptedTypeMask & vpb::Source::HEIGHT_FIELD)
            {
                if (!first)
                    std::cout << ",";
                std::cout << " dem";
                first = false;
            }

            if (itr->second.acceptedTypeMask & vpb::Source::MODEL)
            {
                if (!first)
                    std::cout << ",";
                std::cout << " model";
                first = false;
            }

            if (itr->second.acceptedTypeMask & vpb::Source::SHAPEFILE)
            {
                if (!first)
                    std::cout << ",";
                std::cout << " shapefile";
                first = false;
            }
            std::cout << " : " << itr->second.description << std::endl;
        }

        return 1;
    }

    vpb::System::instance()->readArguments(arguments);

    std::string taskFileName;
    osg::ref_ptr<vpb::Task> taskFile;
    while (arguments.read("--task", taskFileName))
    {
        if (!taskFileName.empty())
        {
            taskFile = new vpb::Task(taskFileName);

            taskFile->read();

            taskFile->setStatus(vpb::Task::RUNNING);
            taskFile->setProperty("pid", vpb::getProcessID());
            taskFile->write();
        }
    }

    osg::ref_ptr<osgTerrain::TerrainTile> terrain = 0;

    //std::cout<<"PID="<<getpid()<<std::endl;

    std::string sourceName;
    while (arguments.read("-s", sourceName))
    {
        std::string fileName = osgDB::findDataFile(sourceName);
        if (fileName.empty())
        {

            osg::notify(osg::NOTICE) << "Error: osgdem running on \"" << vpb::getLocalHostName() << "\", could not find source file \"" << sourceName << "\"" << std::endl;
            char str[2048];
            if (vpb::getCurrentWorkingDirectory(str, sizeof(str)))
            {
                osg::notify(osg::NOTICE) << "       current working directory at time of error = " << str << std::endl;
            }
            osg::setNotifyLevel(osg::DEBUG_INFO);

            osg::notify(osg::NOTICE) << "       now setting NotifyLevel to DEBUG, and re-running find:" << std::endl;
            osg::notify(osg::NOTICE) << std::endl;
            fileName = osgDB::findDataFile(sourceName);
            if (!fileName.empty())
            {
                osg::notify(osg::NOTICE) << std::endl << "Second attempt at finding source file successful!" << std::endl << std::endl;
            }
            else
            {
                osg::notify(osg::NOTICE) << std::endl << "Second attempt at finding source file also failed." << std::endl << std::endl;
            }

            osg::setNotifyLevel(osg::NOTICE);

            return 1;
        }

        osg::ref_ptr<osg::Node> node = osgDB::readNodeFile(fileName);
        if (node.valid())
        {
            osgTerrain::TerrainTile *loaded_terrain = dynamic_cast<osgTerrain::TerrainTile *>(node.get());
            if (loaded_terrain)
            {
                terrain = loaded_terrain;
            }
            else
            {
                osg::notify(osg::NOTICE) << "Error: source file \"" << sourceName << "\" not suitable terrain data." << std::endl;
                return 1;
            }
        }
        else
        {
            osg::notify(osg::NOTICE) << "Error: unable to load source file \"" << sourceName << "\"" << std::endl;
            osg::notify(osg::NOTICE) << "       the file was found as \"" << fileName << "\"" << std::endl;
            osg::notify(osg::NOTICE) << "       now setting NotifyLevel to DEBUG, and re-running load:" << std::endl;
            osg::notify(osg::NOTICE) << std::endl;

            osg::setNotifyLevel(osg::DEBUG_INFO);

            osg::ref_ptr<osg::Node> node = osgDB::readNodeFile(fileName);
            if (node.valid())
            {
                osg::notify(osg::NOTICE) << std::endl;
                osg::notify(osg::NOTICE) << "Second attempt to load source file \"" << sourceName << "\" successful!" << std::endl << std::endl;
            }
            else
            {
                osg::notify(osg::NOTICE) << std::endl;
                osg::notify(osg::NOTICE) << "Second attempt to load source file \"" << sourceName << "\" also failed." << std::endl << std::endl;
            }

            osg::setNotifyLevel(osg::NOTICE);

            return 1;
        }
    }

    if (!terrain)
        terrain = new osgTerrain::TerrainTile;

    std::string terrainOutputName;
    while (arguments.read("--so", terrainOutputName))
    {
    }

    bool report = false;
    while (arguments.read("--report"))
    {
        report = true;
    }

    int result = commandline.read(std::cout, arguments, terrain.get());
    if (result)
        return result;

    // any options left unread are converted into errors to write out later.
    arguments.reportRemainingOptionsAsUnrecognized();

    // report any errors if they have occured when parsing the program aguments.
    if (arguments.errors())
    {
        arguments.writeErrorMessages(std::cout);
        return 1;
    }

    std::string xodrName = vpb::System::instance()->getXodrName();
    VPBRoad *road = NULL;
    if (!xodrName.empty())
    {
        //sleep(20);
        VPBRoad *road = new VPBRoad(xodrName);
    }

    if (!terrainOutputName.empty())
    {
        if (terrain.valid())
        {
            osgDB::writeNodeFile(*terrain, terrainOutputName);

            // make sure the OS writes changes to disk
            vpb::sync();
        }
        else
        {
            osg::notify(osg::NOTICE) << "Error: unable to create terrain output \"" << terrainOutputName << "\"" << std::endl;
        }
        return 1;
    }

    double duration = 0.0;

    // generate the database
    if (terrain.valid())
    {
        try
        {

            vpb::DatabaseBuilder *db = dynamic_cast<vpb::DatabaseBuilder *>(terrain->getTerrainTechnique());
            vpb::BuildOptions *bo = db ? db->getBuildOptions() : 0;

            if (bo)
            {
                osg::setNotifyLevel(osg::NotifySeverity(bo->getNotifyLevel()));
            }
            osg::ref_ptr<vpb::DataSet> dataset = new vpb::DataSet;

            if (bo && !(bo->getLogFileName().empty()))
            {
                dataset->setBuildLog(new vpb::BuildLog(bo->getLogFileName()));
            }

            if (taskFile.valid())
            {
                dataset->setTask(taskFile.get());
            }

            dataset->addTerrain(terrain.get());

            // make sure the OS writes changes to disk
            vpb::sync();

            // check to make sure that the build itself is ready to run and configured correctly.
            std::string buildProblems = dataset->checkBuildValidity();
            if (buildProblems.empty())
            {
                result = dataset->run();

                if (dataset->getBuildLog() && report)
                {
                    dataset->getBuildLog()->report(std::cout);
                }

                duration = osg::Timer::instance()->delta_s(startTick, osg::Timer::instance()->tick());

                dataset->log(osg::NOTICE, "Elapsed time = %f", duration);

                if (taskFile.valid())
                {
                    taskFile->setStatus(vpb::Task::COMPLETED);
                }
            }
            else
            {
                dataset->log(osg::NOTICE, "Build configuration invalid : %s", buildProblems.c_str());
                if (taskFile.valid())
                {
                    taskFile->setStatus(vpb::Task::FAILED);
                }
            }
        }
        catch (std::string str)
        {
            printf("Caught exception : %s\n", str.c_str());

            if (taskFile.valid())
            {
                taskFile->setStatus(vpb::Task::FAILED);
            }

            result = 1;
        }
        catch (...)
        {
            printf("Caught exception.\n");

            if (taskFile.valid())
            {
                taskFile->setStatus(vpb::Task::FAILED);
            }

            result = 1;
        }
    }

    if (duration == 0)
        duration = osg::Timer::instance()->delta_s(startTick, osg::Timer::instance()->tick());

    if (taskFile.valid())
    {
        taskFile->setProperty("duration", duration);
        taskFile->write();
    }

    // make sure the OS writes changes to disk
    vpb::sync();
    delete road;

    return result;
}
