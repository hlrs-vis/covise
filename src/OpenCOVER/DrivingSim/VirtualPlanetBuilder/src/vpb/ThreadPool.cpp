/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include <vpb/ThreadPool>

using namespace vpb;

ThreadPool::ThreadPool(unsigned int numThreads, bool requiresGraphicsContext)
    : _numThreads(numThreads)
    , _requiresGraphicsContext(requiresGraphicsContext)
{
    init();
}

ThreadPool::ThreadPool(const ThreadPool &tp, const osg::CopyOp &copyop)
    : _numThreads(tp._numThreads)
    , _requiresGraphicsContext(tp._requiresGraphicsContext)
{
    init();
}

ThreadPool::~ThreadPool()
{
    stopThreads();
}

void ThreadPool::init()
{
    _numRunningOperations = 0;
    _done = false;

    _operationQueue = new osg::OperationQueue;
    _blockOp = new BlockOperation;

    osg::GraphicsContext *sharedContext = 0;

    _maxNumberOfOperationsInQueue = 64;

    for (unsigned int i = 0; i < _numThreads; ++i)
    {
        osg::ref_ptr<osg::OperationThread> thread;
        osg::ref_ptr<osg::GraphicsContext> gc;

        if (_requiresGraphicsContext)
        {
            osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
            traits->readDISPLAY();
            traits->x = 0;
            traits->y = 0;
            traits->width = 1;
            traits->height = 1;
            traits->windowDecoration = false;
            traits->doubleBuffer = false;
            traits->sharedContext = sharedContext;
            traits->pbuffer = true;

            gc = osg::GraphicsContext::createGraphicsContext(traits.get());

            if (!gc)
            {
                traits->pbuffer = false;
                gc = osg::GraphicsContext::createGraphicsContext(traits.get());
            }

            if (gc.valid())
            {
                gc->realize();

                if (!sharedContext)
                    sharedContext = gc.get();

                gc->createGraphicsThread();

                thread = gc->getGraphicsThread();
            }
        }
        else
        {

            thread = new osg::OperationThread;
            thread->setParent(this);
        }

        if (thread.valid())
        {
            thread->setOperationQueue(_operationQueue.get());

            _threads.push_back(ThreadContextPair(thread, gc));
        }
    }
}

void ThreadPool::startThreads()
{
    //int numProcessors = OpenThreads::GetNumberOfProcessors();
    int processNum = 0;
    _done = false;
    for (Threads::iterator itr = _threads.begin();
         itr != _threads.end();
         ++itr, ++processNum)
    {
        osg::OperationThread *thread = itr->first.get();
        if (!thread->isRunning())
        {
            //thread->setProcessorAffinity(processNum % numProcessors);
            thread->startThread();
        }
    }
}

void ThreadPool::stopThreads()
{
    _done = true;

    for (Threads::iterator itr = _threads.begin();
         itr != _threads.end();
         ++itr)
    {
        osg::OperationThread *thread = itr->first.get();
        if (thread->isRunning())
        {
            thread->setDone(true);
        }
    }
}

void ThreadPool::run(osg::Operation *op)
{
    if (_done)
    {
        log(osg::NOTICE, "ThreadPool::run() Attempt to run BuilderOperation after ThreadPool has been suspended.");
        return;
    }

    while (_operationQueue->getNumOperationsInQueue() >= _maxNumberOfOperationsInQueue)
    {
        log(osg::INFO, "ThreadPool::run() Waiting for operation queue to clear.");

        // Wait for half a seocnd for the queue to clear.
        OpenThreads::Thread::microSleep(500000);
    }

    _operationQueue->add(op);
}

unsigned int ThreadPool::getNumOperationsRunning() const
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    return _numRunningOperations;
}

void ThreadPool::waitForCompletion()
{
    _blockOp->reset();

    _operationQueue->add(_blockOp.get());

    // wait till block is complete i.e. the operation queue has been cleared up to the block
    _blockOp->block();

    // there can still be operations running though so need to double check.
    while (getNumOperationsRunning() > 0 && !done())
    {
        // log(osg::INFO, "MachinePool::waitForCompletion : Waiting for threads to complete = %d",getNumOperationsRunning());
        OpenThreads::Thread::YieldCurrentThread();
    }
}

void ThreadPool::runningOperation(BuildOperation *op)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    ++_numRunningOperations;
}

void ThreadPool::completedOperation(BuildOperation *op)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    --_numRunningOperations;
}
