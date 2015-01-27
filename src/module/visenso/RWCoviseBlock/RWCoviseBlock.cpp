/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RWCoviseBlock.h"
#include "OutputObjectFactory.h"
#include "OutputObject.h"
#include "LinesObject.h"
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

RWCoviseBlock::RWCoviseBlock(int argc, char *argv[])
    : coModule(argc, argv, "Write COVISE-data blockwise", true)
{
    outFileParam_ = addFileBrowserParam("file_path", "Out_file_path");
    outFileParam_->setValue(".", "*");

    readFinishedParam_ = addBooleanParam("readFinished", "sync_param");
    ///readFinishedParam_->setImmediate(1);

    gridInPort_ = addInputPort("geometry_in", "Polygons|Lines", "geometry");
    dataInPort_ = addInputPort("data_in", "Float|Vec3", "data");
    gridInPort_->setRequired(0);
    dataInPort_->setRequired(0);
}

int RWCoviseBlock::compute(const char * /*port*/)
{

    std::cerr << "RWCoviseBlock::compute() called" << endl;

    readFinishedParam_->setValue(0);

    const coDistributedObject *d = gridInPort_->getCurrentObject();
    if (!d)
    {
        sendError(" did not receive object at grid input port");
        return STOP_PIPELINE;
    }

#if defined(WIN32) || defined(WIN64)
    OutputObject *oGrid = OutputObjectFactory::create(d);
    fprintf(stderr, "DDD");
    const char *fName = outFileParam_->getValue();
    int fd;
    if (_sopen_s(&fd, fName, O_BINARY | O_WRONLY | O_CREAT, _SH_DENYNO, _S_IWRITE))
    {
        fprintf(stderr, "Error opening %s", fName);
        return -1;
    }
    fprintf(stderr, "EEE");
#else //WIN32 || WIN64
    mode_t my_umask, filemode;

    my_umask = umask(0777);
    umask(my_umask);

    filemode = 0777 & (~my_umask);

    OutputObject *oGrid = OutputObjectFactory::create(d);

    const char *fName = outFileParam_->getValue();
    int fd = ::open(fName, O_WRONLY | O_CREAT, filemode | O_TRUNC);
#endif //WIN32 || WIN64
    if (fd <= 0)
    {
        std::cerr << "RWCoviseBlock::compute() could not open file" << endl;
        return -1;
    }
    oGrid->process(fd);

    const coDistributedObject *dData = dataInPort_->getCurrentObject();
    if (!d)
    {
        sendError(" did not receive object at grid inpu port");
        return STOP_PIPELINE;
    }

    OutputObject *oData = OutputObjectFactory::create(dData);
    oData->process(fd);
#if defined(WIN32) || defined(WIN64)
    _close(fd);
#else //WIN32 || WIN64
// fsync(fd);
#endif //WIN32 || WIN64

    readFinishedParam_->setValue(1);

    return 0;
}

MODULE_MAIN(Unsupported, RWCoviseBlock)
