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

#include <vpb/TaskManager>
#include <vpb/Commandline>
#include <vpb/DatabaseBuilder>
#include <vpb/System>
#include <vpb/FileUtils>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osg/Math>

#include <osgDB/Input>
#include <osgDB/Output>
#include <osgDB/FileUtils>

#include <iostream>

#include <signal.h>

using namespace vpb;

TaskManager::TaskManager()
{
    _done = false;
    _buildName = "build";

    char str[2048];
    _runPath = vpb::getCurrentWorkingDirectory( str, sizeof(str));

    _defaultSignalAction = COMPLETE_RUNNING_TASKS_THEN_EXIT;
}

TaskManager::~TaskManager()
{
    log(osg::INFO,"TaskManager::~TaskManager()");
}

void TaskManager::setBuildLog(BuildLog* bl)
{
    Logger::setBuildLog(bl);

    if (getMachinePool()) getMachinePool()->setBuildLog(bl);
}


void TaskManager::setRunPath(const std::string& runPath)
{
    _runPath = runPath;
    chdir(_runPath.c_str());

    log(osg::NOTICE,"setRunPath = %s",_runPath.c_str());
}

MachinePool* TaskManager::getMachinePool()
{
    return System::instance()->getMachinePool();
}

const MachinePool* TaskManager::getMachinePool() const
{
    return System::instance()->getMachinePool();
}

void TaskManager::readPatchSetUp(const std::string& patchFile)
{
    bool firstTimeBuild = false;
    std::string originalSourceFile;
    std::string path = osgDB::getFilePath(patchFile);
    std::string extension = osgDB::getFileExtension(patchFile);
    std::string filename = osgDB::getSimpleFileName(patchFile);
    std::string basename = osgDB::getNameLessExtension(filename);
    if (extension=="source")
    {
        originalSourceFile = patchFile;

        filename = osgDB::getNameLessExtension(filename);
        extension = osgDB::getFileExtension(filename);
        bool checkSourceFileDirectly = true;

        if (!extension.empty())
        {
            unsigned int i;
            for(i=0; i<extension.size(); ++i)
            {
                char c = extension[i];
                if (c<'0' || c>'9') break;
            }

            bool isNumeric = i==extension.size();
            //int revisionNum = -1;
            if (isNumeric)
            {
                //revisionNum = atoi(extension.c_str());
                filename = osgDB::getNameLessExtension(filename);
                extension = osgDB::getFileExtension(filename);
                basename = filename;
                checkSourceFileDirectly = false;
            }
        }

        if (checkSourceFileDirectly)
        {
            osg::ref_ptr<osgTerrain::TerrainTile> terrainTile = readSourceFile(patchFile);
            vpb::DatabaseBuilder* db = dynamic_cast<vpb::DatabaseBuilder*>(terrainTile->getTerrainTechnique());
            vpb::BuildOptions* bo = db ? db->getBuildOptions() : 0;
            if (bo)
            {
                path = bo->getDirectory();
                filename = bo->getDestinationTileBaseName();
                basename = filename;
                extension =  bo->getDestinationTileExtension();

                osg::notify(osg::NOTICE)<<"   path "<<path<<std::endl;
                osg::notify(osg::NOTICE)<<"   filename "<<filename<<std::endl;
                osg::notify(osg::NOTICE)<<"   extension "<<extension<<std::endl;

                std::string rootTile = path + basename + extension;

                // check to see if database has already been built.
                osgDB::FileType type = osgDB::fileType(rootTile);
                if (type!=osgDB::REGULAR_FILE)
                {
                    firstTimeBuild = true;
                }
            }
            else
            {
                throw std::string("Error: No BuildOptions found in source file.");
            }
        }
    }

    typedef std::map<int, std::string> SourceMap;
    SourceMap sourceMap;

    osgDB::DirectoryContents directoryContents = osgDB::getDirectoryContents(path);
    for(osgDB::DirectoryContents::iterator itr = directoryContents.begin();
        itr != directoryContents.end();
        ++itr)
    {
        std::string file = *itr;

        if (file.size()>=basename.size() && file.compare(0, basename.size(), basename)==0)
        {
            if (osgDB::getFileExtension(file)=="source")
            {
                std::string nameLessSourceExtension = osgDB::getNameLessExtension(file);
                std::string revisionExtension = osgDB::getFileExtension(nameLessSourceExtension);
                if (!revisionExtension.empty())
                {
                    unsigned int i;
                    for(i=0; i<revisionExtension.size(); ++i)
                    {
                        char c = revisionExtension[i];
                        if (c<'0' || c>'9') break;
                    }

                    bool isNumeric = i==revisionExtension.size();
                    int revisionNum = -1;
                    if (isNumeric)
                    {
                        revisionNum = atoi(revisionExtension.c_str());
                        sourceMap[revisionNum] = file;
                    }
                    else
                    {
                        sourceMap[0] = file;
                    }
                }
            }
        }
    }

    if (sourceMap.empty())
    {
        if (!originalSourceFile.empty())
        {
            if (firstTimeBuild)
            {
                readSource(originalSourceFile);
            }
            else
            {
                if (!readPreviousSource(originalSourceFile))
                {
                    throw std::string("Error: Unable to read source file ") + originalSourceFile;
                }

                vpb::DatabaseBuilder* db = dynamic_cast<vpb::DatabaseBuilder*>(_previousTerrainTile->getTerrainTechnique());
                vpb::BuildOptions* bo = db ? db->getBuildOptions() : 0;
                unsigned int lastRevisionNum = bo ? bo->getRevisionNumber() : 0;
                unsigned int newRevisionNum = lastRevisionNum+1;

                readSource(originalSourceFile);

                getBuildOptions()->setRevisionNumber(newRevisionNum);
            }

        }
        else
        {
            throw std::string("Error: No source files found to base database patching on.");
        }
    }
    else
    {
        int lastRevisionNum = sourceMap.rbegin()->first;
        std::string previousSourceFile = osgDB::concatPaths(path, sourceMap.rbegin()->second);

        int newRevisionNum = lastRevisionNum+1;

        readPreviousSource(previousSourceFile);
        readSource(previousSourceFile);

        getBuildOptions()->setRevisionNumber(newRevisionNum);
    }
}


int TaskManager::read(osg::ArgumentParser& arguments)
{
    std::string patchFile;
    while (arguments.read("--patch",patchFile))
    {
        osg::notify(osg::NOTICE)<<"--patch "<<patchFile<<std::endl;
        readPatchSetUp(patchFile);
    }

    std::string logFileName;
    while (arguments.read("--master-log",logFileName))
    {
        BuildLog* bl = new BuildLog(logFileName);
        setBuildLog(bl);
        pushOperationLog(bl);
    }

    std::string sourceName;
    while (arguments.read("-s",sourceName))
    {
        readSource(sourceName);
    }

    if (!_terrainTile) _terrainTile = new osgTerrain::TerrainTile;

    std::string terrainOutputName;
    while (arguments.read("--so",terrainOutputName)) {}


    Commandline commandlineParser;


    int result = commandlineParser.read(std::cout, arguments, _terrainTile.get());
    if (result) return result;

    while (arguments.read("--build-name",_buildName)) {}

    if (!terrainOutputName.empty())
    {
        if (_terrainTile.valid())
        {
            osgDB::writeNodeFile(*_terrainTile, terrainOutputName);

            // make sure the changes are written to disk.
            vpb::sync();
        }
        else
        {
            log(osg::NOTICE,"Error: unable to create terrain output \"%s\"",terrainOutputName.c_str());
        }
    }

    std::string taskSetFileName;
    while (arguments.read("--tasks",taskSetFileName)) {}

    if (!taskSetFileName.empty())
    {
        readTasks(taskSetFileName);
    }

    while (arguments.read("--modified"))
    {
        setOutOfDateTasksToPending();
    }

    return 0;
}

void TaskManager::setSource(osgTerrain::TerrainTile* terrainTile)
{
    _terrainTile = terrainTile;
}

osgTerrain::TerrainTile* TaskManager::getSource()
{
    return _terrainTile.get();
}

void TaskManager::setPreviousSource(osgTerrain::TerrainTile* terrainTile)
{
    _previousTerrainTile = terrainTile;
}

osgTerrain::TerrainTile* TaskManager::getPreviousSource()
{
    return _previousTerrainTile.get();
}

void TaskManager::nextTaskSet()
{
    // don't need to add a new task set if last task set is still empty.
    if (!_taskSetList.empty() && _taskSetList.back().empty()) return;

    _taskSetList.push_back(TaskSet());
}

void TaskManager::addTask(Task* task)
{
    if (!task) return;

    if (_taskSetList.empty()) _taskSetList.push_back(TaskSet());
    _taskSetList.back().push_back(task);

}

void TaskManager::addTask(const std::string& taskFileName, const std::string& application, const std::string& sourceFile,
                          const std::string& fileListBaseName)
{
    osg::ref_ptr<Task> taskFile = new Task(taskFileName);

    if (taskFile->valid())
    {
        taskFile->setProperty("application",application);
        taskFile->setProperty("source",sourceFile);
        taskFile->setProperty("fileListBaseName",fileListBaseName);

        taskFile->write();

        addTask(taskFile.get());
    }
}

std::string TaskManager::createUniqueTaskFileName(const std::string application)
{
    return "taskfile.task";
}

void TaskManager::buildWithoutSlaves()
{

    if (_terrainTile.valid())
    {
        try
        {
            osg::ref_ptr<vpb::DataSet> dataset = new vpb::DataSet;

            vpb::DatabaseBuilder* db = dynamic_cast<vpb::DatabaseBuilder*>(_terrainTile->getTerrainTechnique());
            vpb::BuildOptions* bo = db ? db->getBuildOptions() : 0;

            if (bo && !(bo->getLogFileName().empty()))
            {
                dataset->setBuildLog(new vpb::BuildLog);
            }

            if (_taskFile.valid())
            {
                dataset->setTask(_taskFile.get());
            }

            dataset->addTerrain(_terrainTile.get());

            int result = dataset->run();

            log(osg::NOTICE,"dataset->run() completed, return value %d",result);

            if (dataset->getBuildLog())
            {
                dataset->getBuildLog()->report(std::cout);
            }

        }
        catch(...)
        {
            printf("Caught exception.\n");
        }

    }
}

bool TaskManager::generateTasksFromSource()
{
    if (!_terrainTile) return false;
    try
    {

        osg::ref_ptr<vpb::DataSet> dataset = new vpb::DataSet;

        vpb::DatabaseBuilder* db = dynamic_cast<vpb::DatabaseBuilder*>(_terrainTile->getTerrainTechnique());
        vpb::BuildOptions* bo = db ? db->getBuildOptions() : 0;

        if (getBuildLog())
        {
            dataset->setBuildLog(getBuildLog());
        }
        else if (bo && !(bo->getLogFileName().empty()))
        {
            dataset->setBuildLog(new vpb::BuildLog(bo->getLogFileName()));
        }


        if (_taskFile.valid())
        {
            dataset->setTask(_taskFile.get());
        }

        if (_previousTerrainTile.valid())
        {
            unsigned int numberAlteredSources = dataset->addPatchedTerrain(_previousTerrainTile.get(), _terrainTile.get());
            if (numberAlteredSources==0)
            {
                dataset->log(osg::NOTICE,"No new, modified or removed sources in patch, so no need to patch database.");
                return false;
            }
        }
        else
        {
            dataset->addTerrain(_terrainTile.get());
        }

        if (dataset->requiresReprojection())
        {
            dataset->log(osg::NOTICE,"Error: vpbmaster can not run without all source data being in the correct destination coordinates system, please reproject them.");
            return false;
        }

        dataset->generateTasks(this);

        // update the current build options with the distination extents.
        bo->setDestinationExtents(dataset->getDestinationExtents());
        bo->setDistributedBuildSecondarySplitLevel(dataset->getDistributedBuildSecondarySplitLevel());
        bo->setDistributedBuildSplitLevel(dataset->getDistributedBuildSplitLevel());

        if (dataset->getBuildLog())
        {
            dataset->getBuildLog()->report(std::cout);
        }
    }
    catch(...)
    {
        printf("Caught exception.\n");
        return false;
    }
    return true;
}

bool TaskManager::run()
{
    log(osg::NOTICE,"Begining run");

    if (getBuildOptions() && getBuildOptions()->getAbortRunOnError())
    {
        getMachinePool()->setTaskFailureOperation(MachinePool::COMPLETE_RUNNING_TASKS_THEN_EXIT);
    }

    getMachinePool()->setTaskManager(this);

    std::string revisionsFileName;
    if (getBuildOptions())
    {
        revisionsFileName = getBuildOptions()->getDirectory() +
                            getBuildOptions()->getDestinationTileBaseName() +
                            getBuildOptions()->getDestinationTileExtension() + std::string(".revisions");


        osg::ref_ptr<osg::Object> object = osgDB::readObjectFile(revisionsFileName);
        osg::ref_ptr<osgDB::DatabaseRevisions> dr = dynamic_cast<osgDB::DatabaseRevisions*>(object.get());

        if (!dr)
        {
            dr = new osgDB::DatabaseRevisions;
            dr->setName(revisionsFileName);
        }

        setDatabaseRevisions(dr.get());
    }


    for(TaskSetList::iterator tsItr = _taskSetList.begin();
        tsItr != _taskSetList.end() && !done();
        )
    {
        for(TaskSet::iterator itr = tsItr->begin();
            itr != tsItr->end() && !done();
            ++itr)
        {
            Task* task = itr->get();
            Task::Status status = task->getStatus();
            switch(status)
            {
                case(Task::RUNNING):
                {
                    // do we check to see if this process is still running?
                    // do we kill this process?
                    log(osg::NOTICE,"Task claims still to be running: %s",task->getFileName().c_str());
                    break;
                }
                case(Task::COMPLETED):
                {
                    // task already completed so we can ignore it.
                    log(osg::NOTICE,"Task claims to have been completed: %s",task->getFileName().c_str());
                    break;
                }
                case(Task::FAILED):
                {
                    // run the task
                    log(osg::NOTICE,"Task previously failed attempting re-run: %s",task->getFileName().c_str());
                    getMachinePool()->run(task);
                    break;
                }
                case(Task::PENDING):
                {
                    // run the task
                    log(osg::NOTICE,"scheduling task : %s",task->getFileName().c_str());
                    getMachinePool()->run(task);
                    break;
                }
            }

        }

        // now need to wait till all dispatched tasks are complete.
        getMachinePool()->waitForCompletion();

        // tally up the tasks to see how we've done on this TasksSet
        unsigned int tasksPending = 0;
        unsigned int tasksRunning = 0;
        unsigned int tasksCompleted = 0;
        unsigned int tasksFailed = 0;
        for(TaskSet::iterator itr = tsItr->begin();
            itr != tsItr->end();
            ++itr)
        {
            Task* task = itr->get();
            Task::Status status = task->getStatus();
            switch(status)
            {
                case(Task::RUNNING):
                {
                    ++tasksRunning;
                    break;
                }
                case(Task::COMPLETED):
                {
                    ++tasksCompleted;
                    break;
                }
                case(Task::FAILED):
                {
                    ++tasksFailed;
                    break;
                }
                case(Task::PENDING):
                {
                    ++tasksPending;
                    break;
                }
            }
        }
        log(osg::NOTICE,"End of TaskSet: tasksPending=%d taskCompleted=%d taskRunning=%d tasksFailed=%d",tasksPending,tasksCompleted,tasksRunning,tasksFailed);

        // if (tasksFailed != 0) break;

        if (getBuildOptions() && getBuildOptions()->getAbortRunOnError() && tasksFailed>0)
        {
            log(osg::NOTICE,"Task failed aborting.");
            break;
        }


        if (getMachinePool()->getNumThreadsNotDone()==0)
        {
            while(getMachinePool()->getNumThreadsRunning()>0)
            {
                log(osg::INFO,"TaskManager::run() - Waiting for threads to exit.");
                OpenThreads::Thread::YieldCurrentThread();
            }

            break;
        }


        if (tasksPending!=0 || tasksFailed!=0 || tasksRunning!=0)
        {
            log(osg::NOTICE,"Continuing with existing TaskSet.");
        }
        else
        {
            ++tsItr;
        }

    }

    // tally up the tasks to see how we've done overall
    unsigned int tasksPending = 0;
    unsigned int tasksRunning = 0;
    unsigned int tasksCompleted = 0;
    unsigned int tasksFailed = 0;
    for(TaskSetList::iterator tsItr = _taskSetList.begin();
        tsItr != _taskSetList.end();
        ++tsItr)
    {
        for(TaskSet::iterator itr = tsItr->begin();
            itr != tsItr->end();
            ++itr)
        {
            Task* task = itr->get();
            Task::Status status = task->getStatus();
            switch(status)
            {
                case(Task::RUNNING):
                {
                    ++tasksPending;
                    break;
                }
                case(Task::COMPLETED):
                {
                    ++tasksCompleted;
                    break;
                }
                case(Task::FAILED):
                {
                    ++tasksFailed;
                    break;
                }
                case(Task::PENDING):
                {
                    ++tasksPending;
                    break;
                }
            }
        }
    }
    log(osg::NOTICE,"End of run: tasksPending=%d taskCompleted=%d taskRunning=%d tasksFailed=%d",tasksPending,tasksCompleted,tasksRunning,tasksFailed);

    getMachinePool()->reportTimingStats();

    if (tasksFailed==0)
    {
        if (tasksPending==0) log(osg::NOTICE,"Finished run successfully.");
        else log(osg::NOTICE,"Finished run, but did not complete %d tasks.",tasksPending);
    }
    else log(osg::NOTICE,"Finished run, but failed on %d  tasks.",tasksFailed);

    return tasksFailed==0 && tasksPending==0;
}


bool TaskManager::writeSource(const std::string& filename)
{
    if (_terrainTile.valid())
    {
        _sourceFileName = filename;


        std::string path = osgDB::getFilePath(filename);
        if (!path.empty())
        {
            osgDB::FileType type = osgDB::fileType(path);
            if (type==osgDB::REGULAR_FILE)
            {
                throw std::string("Error: TaskManager::writeSource(")+filename+std::string("), path has already been assigned to a regular file.");
            }
            else if (type==osgDB::FILE_NOT_FOUND)
            {
                vpb::mkpath(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
            }
        }

        osgDB::writeNodeFile(*_terrainTile, _sourceFileName);

        // make sure the OS writes the file to disk
        vpb::sync();

        return true;
    }
    else
    {
        return false;
    }
}

osgTerrain::TerrainTile* TaskManager::readSourceFile(const std::string& filename)
{
    osg::ref_ptr<osg::Node> node = osgDB::readNodeFile(filename);
    if (node.valid())
    {
        osg::ref_ptr<osgTerrain::TerrainTile> loaded_terrain = dynamic_cast<osgTerrain::TerrainTile*>(node.get());

        // make sure loaded_terrain is the only one with a reference so we can release it safely on return
        node = 0;

        if (loaded_terrain.valid())
        {
            return loaded_terrain.release();
        }
        else
        {
            log(osg::WARN,"Error: source file \"%s\" not suitable terrain data.",filename.c_str());
        }
    }
    else
    {
        log(osg::WARN,"Error: unable to load source file \"%s\" not suitable terrain data.",filename.c_str());
    }

    return 0;
}

bool TaskManager::readSource(const std::string& filename)
{
    osgTerrain::TerrainTile* loaded_terrain = readSourceFile(filename);
    if (loaded_terrain)
    {
        _sourceFileName = filename;
        _terrainTile = loaded_terrain;
        return true;
    }
    else
    {
        return false;
    }
}

bool TaskManager::readPreviousSource(const std::string& filename)
{
    osgTerrain::TerrainTile* loaded_terrain = readSourceFile(filename);
    if (loaded_terrain)
    {
        _previousSourceFileName = filename;
        _previousTerrainTile = loaded_terrain;
        return true;
    }
    else
    {
        return false;
    }
}


void TaskManager::clearTaskSetList()
{
    _taskSetList.clear();
}

Task* TaskManager::readTask(osgDB::Input& fr, bool& itrAdvanced)
{
    if (fr.matchSequence("exec {"))
    {
        int local_entry = fr[0].getNoNestedBrackets();

        fr += 2;

        std::string application;

        while (!fr.eof() && fr[0].getNoNestedBrackets()>local_entry)
        {
            if (fr[0].getStr())
            {
                if (application.empty())
                {
                    // first entry is the application
                    application = fr[0].getStr();
                }
                else
                {
                    // subsequent entries and appended to arguments
                    application += std::string(" ") + std::string(fr[0].getStr());
                }
            }
            ++fr;
        }

        ++fr;

        itrAdvanced = true;

        if (!application.empty())
        {
            osg::ref_ptr<Task> task = new Task(createUniqueTaskFileName(application));

            if (task->valid())
            {
                task->setProperty("application",application);

                task->write();

                return task.release();
            }
        }

    }

    std::string filename;
    if (fr.read("taskfile",filename))
    {
        itrAdvanced = true;

        osg::ref_ptr<Task> task = new Task(filename);
        task->read();

        return task.release();
    }

    return 0;
}

bool TaskManager::readTasks(const std::string& filename)
{
    log(osg::NOTICE,"Reading tasks from file...");

    std::string foundFile = osgDB::findDataFile(filename);
    if (foundFile.empty())
    {
        log(osg::WARN,"Error: could not find task file '%s'",filename.c_str());
        return false;
    }

    _tasksFileName = filename;

    osgDB::ifstream fin(foundFile.c_str());

    if (fin)
    {
        osgDB::Input fr;
        fr.attach(&fin);

        while(!fr.eof())
        {
            bool itrAdvanced = false;

            std::string readFilename;
            if (fr.read("file",readFilename))
            {
                nextTaskSet();
                readTasks(readFilename);
                ++itrAdvanced;
            }

            Task* task = readTask(fr, itrAdvanced);
            if (task)
            {
                nextTaskSet();
                addTask(task);
            }

            if (fr.matchSequence("Tasks {"))
            {
                nextTaskSet();

                int local_entry = fr[0].getNoNestedBrackets();

                fr += 2;

                while (!fr.eof() && fr[0].getNoNestedBrackets()>local_entry)
                {
                    bool localAdvanced = false;

                    Task* task = readTask(fr, localAdvanced);
                    if (task)
                    {
                        addTask(task);
                    }

                    if (!localAdvanced) ++fr;
                }

                ++fr;

                itrAdvanced = true;

            }

            if (!itrAdvanced) ++fr;
        }
    }

    log(osg::NOTICE,"Task file read");
    return false;
}

bool TaskManager::writeTask(osgDB::Output& fout, const Task* task, bool asFileNames) const
{
    if (asFileNames)
    {
        fout.indent()<<"taskfile "<<task->getFileName()<<std::endl;
    }
    else
    {
        std::string application;
        std::string arguments;
        if (task->getProperty("application",application))
        {
            fout.indent()<<"exec { "<<application<<" }"<<std::endl;
        }
    }
    return true;
}

bool TaskManager::writeTasks(const std::string& filename, bool asFileNames)
{
    _tasksFileName = filename;

    osgDB::Output fout(filename.c_str());

    for(TaskSetList::const_iterator tsItr = _taskSetList.begin();
        tsItr != _taskSetList.end();
        ++tsItr)
    {
        const TaskSet& taskSet = *tsItr;

        if (taskSet.size()==1)
        {
            writeTask(fout,taskSet.front().get(), asFileNames);
        }
        else if (taskSet.size()>1)
        {
            fout.indent()<<"Tasks {"<<std::endl;
            fout.moveIn();

            for(TaskSet::const_iterator itr = taskSet.begin();
                itr != taskSet.end();
                ++itr)
            {
                writeTask(fout,itr->get(), asFileNames);
            }

            fout.moveOut();
            fout.indent()<<"}"<<std::endl;
        }
    }


    return true;
}

BuildOptions* TaskManager::getBuildOptions()
{
    vpb::DatabaseBuilder* db = _terrainTile.valid() ? dynamic_cast<vpb::DatabaseBuilder*>(_terrainTile->getTerrainTechnique()) : 0;
    return db ? db->getBuildOptions() : 0;
}

void TaskManager::setOutOfDateTasksToPending()
{
    typedef std::map<std::string, Date> FileNameDateMap;
    FileNameDateMap filenameDateMap;

    for(TaskSetList::iterator tsItr = _taskSetList.begin();
        tsItr != _taskSetList.end();
        ++tsItr)
    {
        TaskSet& taskSet = *tsItr;
        for(TaskSet::iterator itr = taskSet.begin();
            itr != taskSet.end();
            ++itr)
        {
            Task* task = itr->get();
            task->read();

            if (task->getStatus()==Task::COMPLETED)
            {
                std::string sourceFile;
                Date buildDate;
                if (task->getProperty("source", sourceFile) &&
                    task->getDate("date",buildDate))
                {
                    Date sourceFileLastModified;

                    FileNameDateMap::iterator fndItr = filenameDateMap.find(sourceFile);
                    if (fndItr != filenameDateMap.end())
                    {
                        sourceFileLastModified = fndItr->second;
                    }
                    else
                    {
                        if (sourceFileLastModified.setWithDateOfLastModification(sourceFile))
                        {
                            if (sourceFileLastModified < buildDate)
                            {
                                osg::ref_ptr<osg::Node> loadedModel = osgDB::readNodeFile(sourceFile);
                                osgTerrain::TerrainTile* terrainTile = dynamic_cast<osgTerrain::TerrainTile*>(loadedModel.get());
                                if (terrainTile)
                                {
                                    System::instance()->getDateOfLastModification(terrainTile, sourceFileLastModified);
                                }
                            }

                            filenameDateMap[sourceFile] = sourceFileLastModified;
                        }
                    }

                    if (sourceFileLastModified > buildDate)
                    {
                        task->setStatus(Task::PENDING);
                    }
                }
            }
        }
    }
}


void TaskManager::setDone(bool done)
{
    _done = done;

    if (_done) getMachinePool()->release();
}

void TaskManager::exit(int sig)
{
    handleSignal(sig);
    setDone(true);
}

void TaskManager::handleSignal(int sig)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_signalHandleMutex);

    switch(getSignalAction(sig))
    {
        case(IGNORE_SIGNAL):
        {
            log(osg::NOTICE,"Ignoring signal %d.",sig);
            break;
        }
        case(DO_NOT_HANDLE):
        {
            log(osg::NOTICE,"DO_NOT_HANDLE signal %d.",sig);
            break;
        }
        case(COMPLETE_RUNNING_TASKS_THEN_EXIT):
        {
            log(osg::NOTICE,"Recieved signal %d, doing COMPLETE_RUNNING_TASKS_THEN_EXIT.",sig);

            _done = true;
            getMachinePool()->removeAllOperations();
            getMachinePool()->release();

            break;
        }
        case(TERMINATE_RUNNING_TASKS_THEN_EXIT):
        {
            log(osg::NOTICE,"Recieved signal %d, doing TERMINATE_RUNNING_TASKS_THEN_EXIT.",sig);

            _done = true;

            getMachinePool()->removeAllOperations();
            getMachinePool()->signal(sig);

            getMachinePool()->cancelThreads();
            getMachinePool()->release();

            break;
        }
        case(RESET_MACHINE_POOL):
        {
            log(osg::NOTICE,"Recieved signal %d, doing RESET_MACHINE_POOL.",sig);
            getMachinePool()->release();
            getMachinePool()->resetMachinePool();
            break;
        }
        case(UPDATE_MACHINE_POOL):
        {
            log(osg::NOTICE,"Recieved signal %d, doing UPDATE_MACHINE_POOL.",sig);
            getMachinePool()->release();
            getMachinePool()->updateMachinePool();
            break;
        }
    }
}

void TaskManager::setSignalAction(int sig, SignalAction action)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex>  lock(_signalHandleMutex);

    if (action==DO_NOT_HANDLE)
    {
        if (_signalActionMap.count(sig)!=0)
        {
            // remove signal handler for signal.
            signal(sig, 0);
        }

        _signalActionMap.erase(sig);
    }
    else
    {
        if (_signalActionMap.count(sig)==0)
        {
            // need to register signal handler for signal
            signal(sig, TaskManager::signalHandler);
        }

        _signalActionMap[sig] = action;
    }
}

TaskManager::SignalAction TaskManager::getSignalAction(int sig) const
{
    SignalActionMap::const_iterator itr = _signalActionMap.find(sig);
    if (itr==_signalActionMap.end()) return _defaultSignalAction;
    return itr->second;
}

void TaskManager::signalHandler(int sig)
{
    System::instance()->getTaskManager()->handleSignal(sig);
}

std::string TaskManager::checkBuildValidity()
{
    BuildOptions* buildOptions = getBuildOptions();
    if (!buildOptions) return std::string("No BuildOptions supplied in source file");

    bool isTerrain = buildOptions->getGeometryType()==BuildOptions::TERRAIN;
    bool containsOptionalLayers = !(buildOptions->getOptionalLayerSet().empty());

    if (containsOptionalLayers && !isTerrain) return std::string("Can not mix optional layers with POLYGONAL and HEIGHTFIELD builds, must use --terrain to enable optional layer support.");

    if (_previousTerrainTile.valid() && _terrainTile.valid())
    {
        vpb::DatabaseBuilder* db = dynamic_cast<vpb::DatabaseBuilder*>(_previousTerrainTile->getTerrainTechnique());
        vpb::BuildOptions* previous_bo = db ? db->getBuildOptions() : 0;

        db = dynamic_cast<vpb::DatabaseBuilder*>(_terrainTile->getTerrainTechnique());
        vpb::BuildOptions* current_bo = db ? db->getBuildOptions() : 0;

        if (previous_bo && current_bo)
        {
            if (!previous_bo->compatible(*current_bo))
            {
                return std::string("Previous build options not compatible with new build options, cannot patch database.");
            }
        }
    }

    return std::string();
}

void TaskManager::setDatabaseRevisions(osgDB::DatabaseRevisions* db)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_databaseRevisionsMutex);
    _databaseRevisions = db;
}

osgDB::DatabaseRevisions* TaskManager::getDatabaseRevisions()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_databaseRevisionsMutex);
    return _databaseRevisions.get();
}

void TaskManager::addRevisionFileList(const std::string& filename)
{
    log(osg::INFO,"addRevisionFileList(%s)",filename.c_str());

    osg::ref_ptr<osg::Object> object = osgDB::readObjectFile(filename);
    osg::ref_ptr<osgDB::FileList> fileList = dynamic_cast<osgDB::FileList*>(object.get());
    if (!fileList)
    {
        log(osg::INFO,"   failed to load file list %s",filename.c_str());
        return;
    }

    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_databaseRevisionsMutex);

    bool writeChangesImmediately = true;

    if (!filename.empty())
    {
        std::string ext = osgDB::getLowerCaseFileExtension(filename);
        std::string fileListName = osgDB::getNameLessExtension(filename);
        std::string revisionName = osgDB::getLowerCaseFileExtension(fileListName);
        if (!revisionName.empty())
        {

            osg::ref_ptr<osgDB::DatabaseRevision> dbRevision;

            // look for a suitable database revision to append to
            for(osgDB::DatabaseRevisions::DatabaseRevisionList::iterator itr = _databaseRevisions->getDatabaseRevisionList().begin();
                itr != _databaseRevisions->getDatabaseRevisionList().end() && !dbRevision;
                ++itr)
            {
                if ((*itr)->getName()==revisionName)
                {
                    log(osg::INFO,"   reusing exsiting DatabaseRevision structure");
                    dbRevision = *itr;
                }
            }

            if (!dbRevision)
            {
                log(osg::INFO,"   create new DatabaseRevision structure %s", revisionName.c_str());

                dbRevision = new osgDB::DatabaseRevision;
                dbRevision->setName(revisionName);
                dbRevision->setDatabasePath(getBuildOptions()->getDirectory());

                _databaseRevisions->addRevision(dbRevision.get());
           }

            std::stringstream sstr;
            sstr << getBuildOptions()->getDirectory()
                 << getBuildOptions()->getDestinationTileBaseName()
                 << getBuildOptions()->getDestinationTileExtension()
                 << "."<<getBuildOptions()->getRevisionNumber();

            std::string fileListBaseName =sstr.str();

            if (ext=="added")
            {
                if (!dbRevision->getFilesAdded())
                {
                    dbRevision->setFilesAdded(new osgDB::FileList);
                    dbRevision->getFilesAdded()->setName(fileListBaseName+".added");
                    if (writeChangesImmediately) osgDB::writeObjectFile(*_databaseRevisions, _databaseRevisions->getName());
                }
                dbRevision->getFilesAdded()->append(fileList.get());

                if (writeChangesImmediately) osgDB::writeObjectFile(*(dbRevision->getFilesAdded()),dbRevision->getFilesAdded()->getName());
            }
            else if (ext=="removed")
            {
                if (!dbRevision->getFilesRemoved())
                {
                    dbRevision->setFilesRemoved(new osgDB::FileList);
                    dbRevision->getFilesRemoved()->setName(fileListBaseName+".removed");
                    if (writeChangesImmediately) osgDB::writeObjectFile(*_databaseRevisions, _databaseRevisions->getName());
                }
                dbRevision->getFilesRemoved()->append(fileList.get());

                if (writeChangesImmediately) osgDB::writeObjectFile(*(dbRevision->getFilesRemoved()),dbRevision->getFilesRemoved()->getName());
            }
            else if (ext=="modified")
            {
                if (!dbRevision->getFilesModified())
                {
                    dbRevision->setFilesModified(new osgDB::FileList);
                    dbRevision->getFilesModified()->setName(fileListBaseName+".modified");
                    if (writeChangesImmediately) osgDB::writeObjectFile(*_databaseRevisions, _databaseRevisions->getName());
                }
                dbRevision->getFilesModified()->append(fileList.get());

                if (writeChangesImmediately) osgDB::writeObjectFile(*(dbRevision->getFilesModified()),dbRevision->getFilesModified()->getName());
            }
        }
    }
}

