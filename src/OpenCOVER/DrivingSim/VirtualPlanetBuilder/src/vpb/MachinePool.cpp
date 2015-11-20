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

#include <vpb/MachinePool>
#include <vpb/Task>
#include <vpb/TaskManager>
#include <vpb/System>

#include <osg/GraphicsThread>
#include <osg/Timer>

#include <osgDB/Input>
#include <osgDB/Output>
#include <osgDB/FileUtils>

#include <signal.h>

#include <iostream>

using namespace vpb;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  MachineOperation
//
MachineOperation::MachineOperation(Task* task):
    osg::Operation(task->getFileName(), false),
    _task(task)
{
    _task->setStatus(Task::PENDING);
    _task->write();
}

void MachineOperation::operator () (osg::Object* object)
{
    Machine* machine = dynamic_cast<Machine*>(object);
    if (machine)
    {
        std::string application;
        if (_task->getProperty("application",application))
        {
            osg::Timer_t startTick = osg::Timer::instance()->tick();

            _task->setProperty("hostname",machine->getHostName());
            _task->setStatus(Task::RUNNING);
            _task->setWithCurrentDate("date");
            _task->write();
            
            // machine->log(osg::NOTICE,"machine=%s running task=%s",machine->getHostName().c_str(),_task->getFileName().c_str());

            machine->startedTask(_task.get());

            int result = machine->exec(application);
            
            machine->endedTask(_task.get());

            // read any updates to the task written to file by the application.
            _task->read();
            
            double duration;
            if (!_task->getProperty("duration",duration))
            {
                duration = osg::Timer::instance()->delta_s(startTick, osg::Timer::instance()->tick());
            }

            if (result==0)
            {
                // success
                _task->setStatus(Task::COMPLETED);
                _task->write();

                // need to update taskmanger with any new file lists
                if (machine->getMachinePool() && machine->getMachinePool()->getTaskManager())
                {
                    std::string fileListBaseName;
                    if (_task->getProperty("fileListBaseName",fileListBaseName))
                    {
                        machine->getMachinePool()->getTaskManager()->addRevisionFileList(fileListBaseName+".added");
                        machine->getMachinePool()->getTaskManager()->addRevisionFileList(fileListBaseName+".removed");
                        machine->getMachinePool()->getTaskManager()->addRevisionFileList(fileListBaseName+".modified");
                    }
                }
            }
            else
            {
                // failure
                _task->setStatus(Task::FAILED);
                _task->write();
                
                // tell the machine about this task failure.
                machine->taskFailed(_task.get(), result);
            }
            
            // machine->log(osg::NOTICE,"machine=%s completed task=%s in %f seconds, result=%d",machine->getHostName().c_str(),_task->getFileName().c_str(),duration,result);
        }

    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  BlockOperation
//
BlockOperation::BlockOperation():
    osg::Operation("Block", false)
{
}

void BlockOperation::release()
{
    Block::release();
}

void BlockOperation::operator () (osg::Object* object)
{
    Block::release();
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Machine
//
Machine::Machine():
    _machinePool(0)
{
}

Machine::Machine(const Machine& m, const osg::CopyOp& copyop):
    osg::Object(m, copyop),
    _machinePool(m._machinePool),
    _hostname(m._hostname),
    _commandPrefix(m._commandPrefix),
    _commandPostfix(m._commandPostfix)
{
}

Machine::Machine(const std::string& hostname,const std::string& cacheDirectory, const std::string& commandPrefix, const std::string& commandPostfix, int numThreads):
    _machinePool(0),
    _hostname(hostname),
    _cacheDirectory(cacheDirectory),
    _commandPrefix(commandPrefix),
    _commandPostfix(commandPostfix)
{
    if (numThreads<0)
    {
        // autodetect
        numThreads = 1;
    }
    
    for(int i=0; i<numThreads; ++i)
    {
        osg::OperationThread* thread = new osg::OperationThread;
        thread->setParent(this);
        _threads.push_back(thread);
    }
}

Machine::~Machine()
{
    log(osg::INFO,"Machine::~Machine()");
}

int Machine::exec(const std::string& application)
{
    bool runningRemotely = getHostName()!=getLocalHostName() && getHostName()!="localhost";

    std::string executionString;

    if (!getCommandPrefix().empty())
    {
        executionString = getCommandPrefix() + std::string(" ") + application;
    }
    else if (runningRemotely)
    {
        executionString = std::string("ssh ") +
                          getHostName() +
                          std::string(" \"") +
                          application +
                          std::string("\"");
    }
    else
    {
        executionString = application;
    }

    if (!getCommandPostfix().empty())
    {
        executionString += std::string(" ") + getCommandPostfix();
    }

    log(osg::INFO,"%s : running %s",getHostName().c_str(),executionString.c_str());

    return system(executionString.c_str());
}

void Machine::startThreads()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_threadsMutex);

    log(osg::INFO,"Machine::startThreads() hostname=%s, threads=%d",_hostname.c_str(),_threads.size());
    for(Threads::iterator itr = _threads.begin();
        itr != _threads.end();
        ++itr)
    {
        log(osg::INFO,"  Started thread");
    
        (*itr)->setDone(false);
        (*itr)->startThread();
    }
}

void Machine::cancelThreads()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_threadsMutex);

    log(osg::NOTICE,"Machine::cancelThreads() hostname=%s, threads=%d",_hostname.c_str(),_threads.size());
    for(Threads::iterator itr = _threads.begin();
        itr != _threads.end();
        ++itr)
    {
        log(osg::NOTICE,"  Cancel thread");
        (*itr)->cancel();

        // assign a new thread as OpenThreads doesn't currently allow cancelled threads to be restarted.
        osg::OperationThread* thread = new osg::OperationThread;
        thread->setParent(this);
        thread->setOperationQueue(_machinePool->getOperationQueue());

        (*itr) = thread;
    }
    log(osg::NOTICE,"Completed Machine::cancelThreads() hostname=%s, threads=%d",_hostname.c_str(),_threads.size());

}

void Machine::setOperationQueue(osg::OperationQueue* queue)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_threadsMutex);

    for(Threads::iterator itr = _threads.begin();
        itr != _threads.end();
        ++itr)
    {
        (*itr)->setOperationQueue(queue);
    }
}

unsigned int Machine::getNumThreadsActive() const
{
#if 1
    return _runningTasks.size();
#else
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_threadsMutex);
    
    unsigned int numThreadsActive = 0;
    for(Threads::const_iterator itr = _threads.begin();
        itr != _threads.end();
        ++itr)
    {
        if ((*itr)->getCurrentOperation().valid() || 
           !(*itr)->getOperationQueue()->empty())
        {
            ++numThreadsActive;
        }
    }
    return numThreadsActive;
#endif
}

unsigned int Machine::getNumThreadsRunning() const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_threadsMutex);
    
    unsigned int numThreadsRunning = 0;
    for(Threads::const_iterator itr = _threads.begin();
        itr != _threads.end();
        ++itr)
    {
        if ((*itr)->isRunning())
        {
            ++numThreadsRunning;
        }
    }
    return numThreadsRunning;
}

unsigned int Machine::getNumThreadsNotDone() const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_threadsMutex);
    
    unsigned int numThreadsNotDone = 0;
    for(Threads::const_iterator itr = _threads.begin();
        itr != _threads.end();
        ++itr)
    {
        if (!(*itr)->getDone())
        {
            ++numThreadsNotDone;
        }
    }
    return numThreadsNotDone;
}

void Machine::startedTask(Task* task)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_runningTasksMutex);
    _runningTasks[task] = osg::Timer::instance()->time_s();

    log(osg::NOTICE,"machine=%s running task=%s",getHostName().c_str(),task->getFileName().c_str());
}

void Machine::endedTask(Task* task)
{
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_runningTasksMutex);

        double duration = 0.0;

        RunningTasks::iterator itr = _runningTasks.find(task);
        if (itr != _runningTasks.end())
        {
            duration = osg::Timer::instance()->time_s() - itr->second;

            _runningTasks.erase(itr);

            std::string taskType;
            task->getProperty("type",taskType);

            _taskStatsMap[taskType].logTime(duration);
        }

        log(osg::NOTICE,"machine=%s completed task=%s in %.1f seconds",getHostName().c_str(),task->getFileName().c_str(),duration);
    }
    
    if (_machinePool) _machinePool->reportTimingStatus();
    
}

void Machine::taskFailed(Task* task, int result)
{
    log(osg::INFO,"%s : taskFailed(%d)", getHostName().c_str(), result);
    if (_machinePool)
    {
        switch(_machinePool->getTaskFailureOperation())
        {
            case(MachinePool::IGNORE_FAILED_TASK):
            {
                log(osg::INFO,"   IGNORE");
                break;
            }
            case(MachinePool::BLACKLIST_MACHINE_AND_RESUBMIT_TASK):
            {
                log(osg::NOTICE,"\nWarning: Task %s has failed, blacklisting machine %s and resubmitting task.\n",task->getFileName().c_str(),getHostName().c_str());
                setDone(true);
                //setOperationQueue(0);
                _machinePool->run(task);
                _machinePool->release();
                break;
            }
            case(MachinePool::COMPLETE_RUNNING_TASKS_THEN_EXIT):
            {
                log(osg::NOTICE,"\nWarning: Task %s on machine %s has failed, completing running tasks then exiting.\n",task->getFileName().c_str(),getHostName().c_str());
                _machinePool->setTaskFailureOperation(MachinePool::IGNORE_FAILED_TASK);
                System::instance()->getTaskManager()->setDone(true);
                _machinePool->removeAllOperations();
                _machinePool->release();
                break;
            }
            case(MachinePool::TERMINATE_RUNNING_TASKS_THEN_EXIT):
            {
                log(osg::NOTICE,"\nWarning: Task %s on machine %s has failed, terminating running tasks then exiting.\n",task->getFileName().c_str(),getHostName().c_str());
                _machinePool->setTaskFailureOperation(MachinePool::IGNORE_FAILED_TASK);
                System::instance()->getTaskManager()->setDone(true);
                _machinePool->removeAllOperations();
                _machinePool->signal(SIGTERM);
                _machinePool->release();
                break;
            }
        }
    }
}


void Machine::signal(int signal)
{
    log(osg::NOTICE,"Machine::signal(%d)",signal);

    RunningTasks tasks;
    {
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_runningTasksMutex);
        tasks = _runningTasks;
    }

    for(RunningTasks::iterator itr = tasks.begin();
        itr != tasks.end();
        ++itr)
    {
        Task* task = itr->first;
        task->read();
        std::string pid;
        if (task->getProperty("pid", pid))
        {
            std::stringstream signalcommand;
            signalcommand << "kill -" << signal<<" "<<pid;
            exec(signalcommand.str());
        }
    }
}

void Machine::setDone(bool done)
{
    for(Threads::const_iterator itr = _threads.begin();
        itr != _threads.end();
        ++itr)
    {
        (*itr)->setDone(done);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  MachinePool
//

MachinePool::MachinePool():
    _done(false),
    _taskFailureOperation(IGNORE_FAILED_TASK),
    _taskManager(0)
{
    //_taskFailureOperation = IGNORE_FAILED_TASK;
    _taskFailureOperation = BLACKLIST_MACHINE_AND_RESUBMIT_TASK;
    //_taskFailureOperation = COMPLETE_RUNNING_TASKS_THEN_EXIT;
    //_taskFailureOperation = TERMINATE_RUNNING_TASKS_THEN_EXIT;
            
    _operationQueue = new osg::OperationQueue;
    _blockOp = new BlockOperation;
}

MachinePool::~MachinePool()
{
    log(osg::INFO,"MachinePool::~MachinePool()");
}

void MachinePool::setBuildLog(BuildLog* bl)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    Logger::setBuildLog(bl);

    for(Machines::const_iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        (*itr)->setBuildLog(bl);
    }
}


void MachinePool::addMachine(const std::string& hostname, const std::string& cacheDirectory, const std::string& commandPrefix, const std::string& commandPostfix, int numThreads)
{
    log(osg::INFO,"addMachine(");
    log(osg::INFO,"     hostname = %s",hostname.c_str());
    log(osg::INFO,"     cacheDirectory = %s",cacheDirectory.c_str());
    log(osg::INFO,"     commandPrefix = %s",commandPrefix.c_str());
    log(osg::INFO,"     commandPostfix = %s",commandPostfix.c_str());
    log(osg::INFO,"     numThreads = %d)",numThreads);

    addMachine(new Machine(hostname, cacheDirectory, commandPrefix, commandPostfix, numThreads));
}

void MachinePool::addMachine(Machine* machine)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    machine->_machinePool = this;
    machine->setBuildLog(getBuildLog());
    machine->setOperationQueue(_operationQueue.get());
    machine->startThreads();
        
    _machines.push_back(machine);
}

Machine* MachinePool::getMachine(const std::string& hostname)
{
    for(Machines::iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        if ((*itr)->getHostName()==hostname) return itr->get();
    }
    return 0;
}


void MachinePool::run(Task* task)
{
    log(osg::INFO, "Adding Task to MachinePool::OperationQueue %s",task->getFileName().c_str());
    _operationQueue->add(new MachineOperation(task));
}

void MachinePool::waitForCompletion()
{
    _blockOp->reset();

    _operationQueue->add(_blockOp.get());

    // wait till block is complete i.e. the operation queue has been cleared up to the block
    _blockOp->block();

    if (!done()) OpenThreads::Thread::YieldCurrentThread();

    // there can still be operations running though so need to double check.
    while(getNumThreadsActive()>0 && !done())
    {
        // log(osg::INFO, "MachinePool::waitForCompletion : Waiting for threads to complete = %d",getNumThreadsActive());
#if 1
        OpenThreads::Thread::microSleep(1000000);
#else
        OpenThreads::Thread::YieldCurrentThread();
#endif
    }

    log(osg::INFO, "MachinePool::waitForCompletion : done %d",done());
    log(osg::INFO, "                               : getNumThreadsActive() %d",int(getNumThreadsActive()));
    log(osg::INFO, "                               : empty %d",int(_operationQueue->empty()));
}

unsigned int MachinePool::getNumThreads() const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    unsigned int numThreads = 0;
    for(Machines::const_iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        numThreads += (*itr)->getNumThreads();
    }
    return numThreads;
}

unsigned int MachinePool::getNumThreadsActive() const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    unsigned int numThreadsActive = 0;
    for(Machines::const_iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        numThreadsActive += (*itr)->getNumThreadsActive();
    }
    return numThreadsActive;
}

unsigned int MachinePool::getNumThreadsRunning() const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    unsigned int numThreadsRunning = 0;
    for(Machines::const_iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        numThreadsRunning += (*itr)->getNumThreadsRunning();
    }
    return numThreadsRunning;
}

unsigned int MachinePool::getNumThreadsNotDone() const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    unsigned int numThreadsNotDone = 0;
    for(Machines::const_iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        numThreadsNotDone += (*itr)->getNumThreadsNotDone();
    }
    return numThreadsNotDone;
}

void MachinePool::clear()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    _machines.clear();
}

bool MachinePool::read(const std::string& filename)
{
    std::string foundFile = osgDB::findDataFile(filename);
    if (foundFile.empty())
    {
        log(osg::WARN, "Error: could not find machine specification file '%s'",filename.c_str());
        return false;
    }
    

    osgDB::ifstream fin(foundFile.c_str());
    
    if (fin)
    {
    
        _machinePoolFileName = foundFile;

        {
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);
            _machines.clear();
        }
        
        osgDB::Input fr;
        fr.attach(&fin);
        
        while(!fr.eof())
        {
            bool itrAdvanced = false;

            std::string readFilename;
            if (fr.read("file",readFilename))
            {
                read(readFilename);
                ++itrAdvanced;
            }
        
            if (fr.matchSequence("Machine {"))
            {
                int local_entry = fr[0].getNoNestedBrackets();

                fr += 2;

                std::string hostname;
                std::string cacheDirectory;
                std::string prefix;
                std::string postfix;
                int numThreads=-1;

                while (!fr.eof() && fr[0].getNoNestedBrackets()>local_entry)
                {
                    bool localAdvanced = false;

                    if (fr.read("hostname",hostname)) localAdvanced = true;
                    if (fr.read("cache",cacheDirectory)) localAdvanced = true;
                    if (fr.read("prefix",prefix)) localAdvanced = true;
                    if (fr.read("postfix",postfix)) localAdvanced = true;
                    if (fr.read("threads",numThreads)) localAdvanced = true;
                    if (fr.read("processes",numThreads)) localAdvanced = true;

                    if (!localAdvanced) ++fr;
                }

                addMachine(hostname,cacheDirectory,prefix,postfix,numThreads);

                ++fr;

                itrAdvanced = true;
            }

            if (!itrAdvanced) ++fr;
        }
    }
    
    return true;
}

bool MachinePool::write(const std::string& filename) const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    osgDB::Output fout(filename.c_str());

    for(Machines::const_iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        const Machine* machine = itr->get();
    
        if (itr != _machines.begin()) fout.indent()<<std::endl;

        fout.indent()<<"Machine {"<<std::endl;
        fout.moveIn();
        
        if (!machine->getHostName().empty()) fout.indent()<<"hostname "<<machine->getHostName()<<std::endl;
        if (!machine->getCacheDirectory().empty()) fout.indent()<<"cache "<<machine->getCacheDirectory()<<std::endl;
        if (!machine->getCommandPrefix().empty()) fout.indent()<<"prefix "<<machine->getCommandPrefix()<<std::endl;
        if (!machine->getCommandPostfix().empty()) fout.indent()<<"postfix "<<machine->getCommandPostfix()<<std::endl;
        if (machine->getNumThreads()>0) fout.indent()<<"processes "<<machine->getNumThreads()<<std::endl;
        
        fout.moveOut();
        fout.indent()<<"}"<<std::endl;
    }
    
    return true;
}

bool MachinePool::setUpOnLocalHost()
{
    log(osg::NOTICE,"Setting up MachinePool to use all %i cores on this machine.",OpenThreads::GetNumberOfProcessors());
    addMachine(vpb::getLocalHostName(),osgDB::getFilePath(vpb::getCacheFileName()),std::string(),std::string(),OpenThreads::GetNumberOfProcessors());
    return true;
}

void MachinePool::removeAllOperations()
{
    _operationQueue->removeAllOperations();
}

void MachinePool::signal(int signal)
{
    log(osg::NOTICE,"MachinePool::signal(%d)",signal);
    
    for(Machines::iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        (*itr)->signal(signal);
    }
}

void MachinePool::setDone(bool done)
{
    log(osg::NOTICE,"MachinePool::setDone(%d)",(int)done);

    _done = done;

    if (_done) removeAllOperations();

    for(Machines::iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        (*itr)->setDone(done);
    }
}

void MachinePool::release()
{
    if (_blockOp.valid()) _blockOp->release();
}

void MachinePool::startThreads()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    for(Machines::iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        (*itr)->startThreads();
    }
}

void MachinePool::cancelThreads()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_machinesMutex);

    for(Machines::iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        (*itr)->cancelThreads();
    }
}


void MachinePool::resetMachinePool()
{
    log(osg::NOTICE,"MachinePool::resetMachinePool()");
    
    // remove all pending tasks
    removeAllOperations();
    
    // stopped threads.
    
    setDone(true);
    
    cancelThreads();
    
    setDone(false);

    // restart any stopped threads.
    startThreads();
}

void MachinePool::updateMachinePool()
{
    log(osg::NOTICE,"MachinePool::updateMachinePool()");

    log(osg::NOTICE,"MachinePool::resetMachinePool()");
    
    // remove all pending tasks
    removeAllOperations();
    
    // stopped threads.
    
    setDone(true);
    
    cancelThreads();
    
    read(_machinePoolFileName);
    
    setDone(false);

    // restart any stopped threads.
    startThreads();
}

void MachinePool::reportTimingStatus()
{
    unsigned int numTasksPending = _operationQueue->getNumOperationsInQueue();

    unsigned int numTasksCompleted = 0;
    double totalComputeTime = 0.0;
    unsigned int numCores = 0;
    
    for(Machines::iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        Machine* machine = itr->get();
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(machine->getRunningTasksMutex());

        TaskStatsMap& taskStatsMap = machine->getTaskStatsMap();
        for(TaskStatsMap::iterator titr = taskStatsMap.begin();
            titr != taskStatsMap.end();
            ++titr)
        {
            TaskStats& stats = titr->second;
            
            numTasksCompleted += stats.numTasks();
            totalComputeTime += stats.totalTime();
        }
        numCores += getNumThreads();
    }
        
    double averageTaskTime = (numTasksCompleted!=0) ? (totalComputeTime/ double(numTasksCompleted)) : 0;

    double currentTime = osg::Timer::instance()->time_s();
    double estimatedTimeOfLastCompletion = currentTime;
    unsigned int numTasksRunning = 0;
    for(Machines::iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        Machine* machine = itr->get();
        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(machine->getRunningTasksMutex());
        
        Machine::RunningTasks& runningTasks = machine->getRunningTasks();
        numTasksRunning += runningTasks.size();
        for(Machine::RunningTasks::iterator ritr = runningTasks.begin();
            ritr != runningTasks.end();
            ++ritr)
        {
            double startTime = ritr->second;
            double elapsedTime = currentTime - startTime;
            double estimatedEndTime = (elapsedTime < averageTaskTime) ? startTime + averageTaskTime : currentTime+1.0;
            if (estimatedTimeOfLastCompletion < estimatedEndTime) estimatedTimeOfLastCompletion = estimatedEndTime;
        }
    }
    
    double numTaskPendingAcrossAllCores = (numTasksPending>0) ? ceil(double(numTasksPending) / double(numCores)) : 0;
    estimatedTimeOfLastCompletion += numTaskPendingAcrossAllCores*averageTaskTime;
    double estimateTimeToCompletion = estimatedTimeOfLastCompletion-currentTime;

    double daysToCompletion = floor(estimateTimeToCompletion / (24.0*60.0*60.0) );
    double secondsRemainder = estimateTimeToCompletion - daysToCompletion*24.0*60.0*60.0;
    double hoursToCompletion = floor(secondsRemainder / (60.0*60.0));
    secondsRemainder = secondsRemainder - hoursToCompletion*60.0*60.0;
    
    double minutesToCompletion = floor(secondsRemainder / 60.0);
    secondsRemainder = secondsRemainder - minutesToCompletion*60.0;

    if (daysToCompletion>0.0)
    {
        log(osg::NOTICE,"Number of tasks completed %i, running %i, pending %i. Estimated time to completion %d days, %d hours, %d minutes, %2.1f percent done.",
            numTasksCompleted, numTasksRunning, numTasksPending, 
            int(daysToCompletion), int(hoursToCompletion), int(minutesToCompletion),
            100.0*currentTime/estimatedTimeOfLastCompletion);
    }
    else if (hoursToCompletion>0.0)
    {
        log(osg::NOTICE,"Number of tasks completed %i, running %i, pending %i. Estimated time to completion %d hours, %d minutes, %2.1f percent done.",
            numTasksCompleted, numTasksRunning, numTasksPending, 
            int(hoursToCompletion), int(minutesToCompletion),
            100.0*currentTime/estimatedTimeOfLastCompletion);
    }
    else if (minutesToCompletion>0.0)
    {
        log(osg::NOTICE,"Number of tasks completed %i, running %i, pending %i. Estimated time to completion %d minutes, %d seconds, %2.1f percent done.",
            numTasksCompleted, numTasksRunning, numTasksPending, 
            int(minutesToCompletion),int(secondsRemainder),
            100.0*currentTime/estimatedTimeOfLastCompletion);
    }
    else
    {
        log(osg::NOTICE,"Number of tasks completed %i, running %i, pending %i. Estimated time to completion %d seconds, %2.1f percent done.",
            numTasksCompleted, numTasksRunning, numTasksPending, 
            int(secondsRemainder),
            100.0*currentTime/estimatedTimeOfLastCompletion);
    }

}

void MachinePool::reportTimingStats()
{
    log(osg::NOTICE,"MachinePool::reportTimingStats()");
    for(Machines::iterator itr = _machines.begin();
        itr != _machines.end();
        ++itr)
    {
        Machine* machine = itr->get();
        TaskStatsMap& taskStatsMap = machine->getTaskStatsMap();
        log(osg::NOTICE,"    Machine : %s",machine->getHostName().c_str());
        for(TaskStatsMap::iterator titr = taskStatsMap.begin();
            titr != taskStatsMap.end();
            ++titr)
        {
            TaskStats& stats = titr->second;
            log(osg::NOTICE,"        Task::type='%s'\tminTime=%f\tmaxTime=%f\taverageTime=%f\ttotalComputeTime=%f\tnumTasks=%d",
                titr->first.c_str(),
                stats.minTime(),
                stats.maxTime(),
                stats.averageTime(),
                stats.totalTime(),
                stats.numTasks());
        }
    }
}
