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
#include <vpb/TaskManager>
#include <vpb/System>
#include <vpb/FileUtils>
#include <vpb/DatabaseBuilder>
#include <vpb/Version>

#include <osg/Timer>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>

#include <iostream>

#include <signal.h>

int main(int argc, char **argv)
{
    osg::ref_ptr<vpb::TaskManager> taskManager = vpb::System::instance()->getTaskManager();

#ifndef _WIN32
    taskManager->setSignalAction(SIGHUP, vpb::TaskManager::COMPLETE_RUNNING_TASKS_THEN_EXIT);
    taskManager->setSignalAction(SIGQUIT, vpb::TaskManager::TERMINATE_RUNNING_TASKS_THEN_EXIT);
    taskManager->setSignalAction(SIGKILL, vpb::TaskManager::TERMINATE_RUNNING_TASKS_THEN_EXIT);
    taskManager->setSignalAction(SIGUSR1, vpb::TaskManager::RESET_MACHINE_POOL);
    taskManager->setSignalAction(SIGUSR2, vpb::TaskManager::UPDATE_MACHINE_POOL);
#endif
    taskManager->setSignalAction(SIGABRT, vpb::TaskManager::TERMINATE_RUNNING_TASKS_THEN_EXIT);
    taskManager->setSignalAction(SIGINT, vpb::TaskManager::TERMINATE_RUNNING_TASKS_THEN_EXIT);
    taskManager->setSignalAction(SIGTERM, vpb::TaskManager::TERMINATE_RUNNING_TASKS_THEN_EXIT);

    osg::Timer_t startTick = osg::Timer::instance()->tick();

    osg::ArgumentParser arguments(&argc, argv);

    // set up the usage document, in case we need to print out how to use this program.
    arguments.getApplicationUsage()->setApplicationName(arguments.getApplicationName());
    arguments.getApplicationUsage()->setDescription(arguments.getApplicationName() + " application is utility tools which can be used to generate paged geospatial terrain databases.");
    arguments.getApplicationUsage()->setCommandLineUsage(arguments.getApplicationName() + " [options] filename ...");
    arguments.getApplicationUsage()->addCommandLineOption("--version", "Display version information");
    arguments.getApplicationUsage()->addCommandLineOption("--cache <filename>", "Read the cache file to use a look up for locally cached files.");
    arguments.getApplicationUsage()->addCommandLineOption("-h or --help", "Display this information");

    if (arguments.read("--version"))
    {
        std::cout << "VirtualPlanetBuilder/vpbmaster version " << vpbGetVersion() << std::endl;
        return 0;
    }

    if (arguments.read("--version-number"))
    {
        std::cout << vpbGetVersion() << std::endl;
        return 0;
    }

    // if user requests help write it out to cout.
    if (arguments.read("-h") || arguments.read("--help"))
    {
        arguments.getApplicationUsage()->write(std::cout, osg::ApplicationUsage::COMMAND_LINE_OPTION);
        return 1;
    }

    int result = 0;

    try
    {
        std::string runPath;
        if (arguments.read("--run-path", runPath))
        {
            vpb::chdir(runPath.c_str());
            taskManager->setRunPath(runPath);
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

        taskManager->read(arguments);

        bool buildWithoutSlaves = false;
        while (arguments.read("--build"))
        {
            buildWithoutSlaves = true;
        }

        std::string tasksOutputFileName;
        while (arguments.read("--to", tasksOutputFileName))
            ;

        // any options left unread are converted into errors to write out later.
        arguments.reportRemainingOptionsAsUnrecognized();

        // report any errors if they have occured when parsing the program aguments.
        if (arguments.errors())
        {
            arguments.writeErrorMessages(std::cout);
            taskManager->exit(SIGTERM);
            return 1;
        }

        if (!tasksOutputFileName.empty())
        {
            std::string sourceFileName = taskManager->getBuildName() + std::string("_master.source");
            taskManager->setSourceFileName(sourceFileName);
            taskManager->generateTasksFromSource();

            taskManager->writeSource(tasksOutputFileName);
            taskManager->writeTasks(tasksOutputFileName, true);
            taskManager->exit(SIGTERM);
            return 1;
        }

        std::string buildProblems = taskManager->checkBuildValidity();
        if (buildProblems.empty())
        {
            if (buildWithoutSlaves)
            {
                taskManager->buildWithoutSlaves();
            }
            else
            {
                if (!taskManager->hasTasks())
                {
                    std::string sourceFileName = taskManager->getBuildName() + std::string("_master.source");
                    tasksOutputFileName = taskManager->getBuildName() + std::string("_master.tasks");

                    taskManager->setSourceFileName(sourceFileName);
                    if (!taskManager->generateTasksFromSource())
                    {
                        // nothing to do.
                        taskManager->exit(SIGTERM);
                        return 1;
                    }

                    taskManager->writeSource(sourceFileName);
                    taskManager->writeTasks(tasksOutputFileName, true);

                    taskManager->log(osg::NOTICE, "Generated tasks file = %s", tasksOutputFileName.c_str());

                    vpb::DatabaseBuilder *db = dynamic_cast<vpb::DatabaseBuilder *>(taskManager->getSource()->getTerrainTechnique());
                    vpb::BuildOptions *buildOptions = (db && db->getBuildOptions()) ? db->getBuildOptions() : 0;

                    if (buildOptions)
                    {
                        std::stringstream sstr;
                        sstr << buildOptions->getDirectory() << buildOptions->getDestinationTileBaseName() << buildOptions->getDestinationTileExtension() << "." << buildOptions->getRevisionNumber() << ".source";

                        taskManager->writeSource(sstr.str());

                        taskManager->log(osg::NOTICE, "Revsion source = %s", sstr.str().c_str());
                    }
                }

                // make sure the OS writes changes to disk
                vpb::sync();

                if (taskManager->hasMachines())
                {
                    if (!taskManager->run())
                    {
                        result = 1;
                    }
                }
                else
                {
                    taskManager->log(osg::NOTICE, "Cannot run build without machines assigned, please pass in a machines definition file via --machines <file>.");
                }
            }
        }
        else
        {
            taskManager->log(osg::NOTICE, "Build configuration invalid : %s", buildProblems.c_str());
            result = 1;
        }

        double duration = osg::Timer::instance()->delta_s(startTick, osg::Timer::instance()->tick());
        taskManager->log(osg::NOTICE, "Total elapsed time = %f", duration);
    }
    catch (std::string str)
    {
        taskManager->log(osg::NOTICE, "Caught exception : %s", str.c_str());
        result = 1;
    }
    catch (...)
    {
        taskManager->log(osg::NOTICE, "Caught exception.");
        result = 1;
    }

    // make sure the OS writes changes to disk
    vpb::sync();
    taskManager->log(osg::NOTICE, "Run Complete.");
    taskManager->exit(SIGTERM);
    return result;
}
