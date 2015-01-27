/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PSO_ParallelParticle_H
#define PSO_ParallelParticle_H

/*
 * Author: Florian Seybold, Professor Gerhard Venter
 * University of Stellenbosch and High-Performance Computing Center Stuttgart
 * Copyright: 2007, 2008
*/

#include "Particle.h"
#include <mpi.h>

/// Namespace containing the Particle class as well as the Job-class, the slave-functions and various test functions
namespace pso
{

/// Class representing a function evaluation job. The jobs are sent to slaves processors in a parallel computing environment. A job consits of the point where the response function is to be evaluated and a identifying key.
class Job
{
public:
    /// Array of input variables of the response function. (Position in the design space.)
    double *x;
    /// Number (id/key) of the job.
    int key;

    /// Constructor.
    /**
		\param x Array of input variables.
		\param key Id of job.
	**/
    Job(double *x, int key)
    {
        this->x = x;
        this->key = key;
    }
};

class ParallelParticle : public Particle
{
public:
    /// Constructor of the ParallelParticle class. Randomly initializes the position and velocity vector of the particle.
    ParallelParticle();

    /// Virtual destructor.
    virtual ~ParallelParticle();

    /// Static function initializing the Particle class. This function should be called before any ParallelParticle objects in a parallel computing environment are instantiated. Calls Particle::init().
    /**
		\param setresponse Function pointer to the response function.
		\param setnvar Dimension of the optimization problem (number of input parameters of the response function).
		\param setlowerbound Array defining the lower bound of the search area. Same dimension as the optimization problem.
		\param setupperbound Array defining the upper bound of the search area. Same dimension as the optimization problem.
		\param setinteger Array defining whether one search dimension is continuous or discrete (using integer). Same dimension as the optimization problem.
		\param setdt Time step size. Normally dt=1.
	**/
    static void init(double (*setresponse)(double *), int setnvar, double *setlowerbound, double *setupperbound, bool *setinteger, double setdt);

    /// Function adding the particle's present position to the response value computation job queue.
    void addXJob();

    /// Function adding the present positions of the two additional points to the response value computation job queue.
    void addBetaJob();

    /// Static function which adds an processor to the set of available processors.
    /**
		\param proc Identifier of the processor.
	**/
    static void addProcessor(int proc);

    /// Static function which sends the job on top of the job queue to the first available processor in the processor set.
    static void sendJob();

    /// Static function which receives a processed job from a processor.
    static void receiveJob();

    /// Checks whether the job queue is empty.
    /**
		\return True, if the job queue is empty.
	**/
    static bool checkJobQueueEmpty();

    /// Checks whether the set of available processors is full.
    /**
		\return True, if the processor set is full.
	**/
    static bool checkProcSetFull();

    /// Checks whether the set of available processors is empty.
    /**
		\return True, if the processor set is empty.
	**/
    static bool checkProcSetEmpty();

    /// Static function which processes the job queue by sending its jobs to available processors, until the job queue is empty.
    static void processJobQueue();

protected:
    /// Job queue for parallel processing.
    static std::queue<Job *> jobqueue;
    /// Set of available processors for parallel processing.
    static std::set<int> procset;
    /// Number of available procs.
    static unsigned int procs;
};

/// Function for slaves that receives and processes parallel jobs from the master by calling the given response function.
/**
	\param response Response function.
	\param nvar Number of input parameters of the response function (Size of optimization problem).
**/
void slave(double (*response)(double *), int nvar);

/// Function for slaves that receives and processes parallel jobs from the master by calling the given system commando.
/**
	\param commando System commando that calls the application which evaluates the response function. The input variables are stored in 'dvar.vef', the response value is to be stored in 'resp.vef'.
	\param nvar Number of input parameters of the response function (Size of optimization problem).
**/
void slaveExternCall(std::string commando, int nvar);

/// Inline operator<< function which prints the particle's state to the given output stream.
/**
	\param os Output stream.
	\param par Particle object.
	\return The output stream.
**/
inline std::ostream &operator<<(std::ostream &os, const Particle &par)
{
    par.printState(os);

    return os;
}
}

#endif
