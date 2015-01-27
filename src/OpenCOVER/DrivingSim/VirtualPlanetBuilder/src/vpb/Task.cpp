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

#include <vpb/Task>
#include <vpb/System>

#include <iostream>
#include <sstream>
#include <string>

using namespace vpb;

Task::Task(const std::string &filename)
    : PropertyFile(filename)
    , _argc(0)
    , _argv(0)
{
}

Task::~Task()
{
}

void Task::init(osg::ArgumentParser &arguments)
{
    std::string application;
    for (int i = 0; i < arguments.argc(); ++i)
    {
        if (i > 0)
            application += " ";
        application += arguments[i];
    }

    setProperty("application", application);

    osg::notify(osg::NOTICE) << "Task::init() Application " << application << std::endl;

    setProperty("pid", getProcessID());

    setProperty("hostname", getLocalHostName());
}

void Task::setStatus(Status status)
{
    std::string statusString;
    switch (status)
    {
    case (RUNNING):
        statusString = "running";
        break;
    case (COMPLETED):
        statusString = "completed";
        break;
    case (FAILED):
        statusString = "failed";
        break;
    default:
        statusString = "pending";
        break;
    }
    setProperty("status", statusString);
}

Task::Status Task::getStatus() const
{
    std::string status;
    getProperty("status", status);
    if (status == "running")
        return RUNNING;
    if (status == "failed")
        return FAILED;
    if (status == "completed")
        return COMPLETED;
    return PENDING;
}

void Task::setDate(const std::string &property, const Date &date)
{
    std::string dateString = date.getDateString();
    setProperty(property, dateString);
}

void Task::setWithCurrentDate(const std::string &property)
{
    Date date;
    date.setWithCurrentDate();
    std::string dateString = date.getDateString();
    setProperty(property, dateString);
}

bool Task::getDate(const std::string &property, Date &date) const
{
    std::string value;
    if (getProperty(property, value))
    {
        return date.setWithDateString(value);
    }
    else
    {
        return false;
    }
}

void TaskOperation::operator()(osg::Object *)
{
    _task->read();
    _task->report(std::cout);
}

void SleepOperation::operator()(osg::Object *)
{
    OpenThreads::Thread::microSleep(_microSeconds);
}
