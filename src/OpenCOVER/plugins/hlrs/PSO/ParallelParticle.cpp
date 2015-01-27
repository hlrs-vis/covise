/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Author: Florian Seybold, Professor Gerhard Venter
 * University of Stellenbosch and High-Performance Computing Center Stuttgart
 * Copyright: 2007, 2008
*/

#include "ParallelParticle.h"

using namespace pso;

std::queue<Job *> ParallelParticle::jobqueue;
std::set<int> ParallelParticle::procset;
unsigned int ParallelParticle::procs = 0;

ParallelParticle::ParallelParticle()
    : Particle::Particle()
{
}

ParallelParticle::~ParallelParticle()
{
}

void ParallelParticle::init(double (*setresponse)(double *), int setnvar, double *setlowerbound, double *setupperbound, bool *setinteger, double setdt)
{
    Particle::init(setresponse, setnvar, setlowerbound, setupperbound, setinteger, setdt);

    while (!jobqueue.empty())
        jobqueue.pop();
    procset.clear();
    procs = 0;
}

void ParallelParticle::addXJob()
{
    jobqueue.push(new Job(x, particle));
}

void ParallelParticle::addBetaJob()
{
    jobqueue.push(new Job(xbeta1, (100 + particle)));
    jobqueue.push(new Job(xbeta2, (200 + particle)));
}

void ParallelParticle::addProcessor(int proc)
{
    procset.insert(proc);
    ++procs;
}

void ParallelParticle::sendJob()
{
    MPI_Send(jobqueue.front()->x, nvar, MPI_DOUBLE, *procset.begin(), jobqueue.front()->key, MPI_COMM_WORLD);

    free(jobqueue.front());
    jobqueue.pop();
    procset.erase(procset.begin());
}

void ParallelParticle::receiveJob()
{
    double recval;
    MPI_Status status;

    MPI_Recv(&recval, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    int parnum = status.MPI_TAG % 100;
    int kind = status.MPI_TAG / 100;

    switch (kind)
    {
    case 0:
        par[parnum]->setVal(recval);
        break;
    case 1:
        par[parnum]->setValBeta1(recval);
        break;
    case 2:
        par[parnum]->setValBeta2(recval);
        break;
    }

    procset.insert(status.MPI_SOURCE);
}

bool ParallelParticle::checkJobQueueEmpty()
{
    return jobqueue.empty();
}

bool ParallelParticle::checkProcSetFull()
{
    return procset.size() == procs;
}

bool ParallelParticle::checkProcSetEmpty()
{
    return procset.empty();
}

void ParallelParticle::processJobQueue()
{
    while (!jobqueue.empty() || procset.size() < procs)
    {
        if (!procset.empty() && !jobqueue.empty())
        {
            ParallelParticle::sendJob();
        }

        if (procset.empty() || jobqueue.empty())
        {
            ParallelParticle::receiveJob();
        }

        usleep(10);
    }
}

void pso::slave(double (*response)(double *), int nvar)
{
    double x[nvar];
    int key = 1;
    MPI_Status status;

    double val;

    while (key != 1000)
    {
        MPI_Recv(x, nvar, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        key = status.MPI_TAG;
        if (key == 1000)
            break;

        val = (*response)(x);

        MPI_Send(&val, 1, MPI_DOUBLE, 0, key, MPI_COMM_WORLD);

        //std::cerr << "Hello! Slave has finished!" << std::endl;
    }
}

void pso::slaveExternCall(std::string command, int nvar)
{
    const char dvarfilename[] = "dvar.vef";
    const char respfilename[] = "resp.vef";

    double x[nvar];
    int key = 1;
    MPI_Status status;

    double val;

    std::stringstream dirstream;
    std::string dir;
    std::ofstream dvarfile;
    std::ifstream respfile;
    std::string rmcommand;

    while (key != 1000)
    {
        MPI_Recv(x, nvar, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        key = status.MPI_TAG;
        if (key == 1000)
            break;

        dirstream.clear();
        dir.clear();
        dirstream << std::setfill('0') << std::setw(3) << key;
        dirstream >> dir;

        mkdir(dir.c_str(), S_IRWXU);

        chdir(dir.c_str());

        dvarfile.open(dvarfilename);
        for (int i = 0; i < nvar; ++i)
            dvarfile << std::scientific << x[i] << std::endl;
        dvarfile.close();

        system(command.c_str());

        respfile.open(respfilename);
        respfile >> val;
        respfile.close();

        chdir("..");

        rmcommand = "rm -rf " + dir;
        system(rmcommand.c_str());

        MPI_Send(&val, 1, MPI_DOUBLE, 0, key, MPI_COMM_WORLD);
    }
}
