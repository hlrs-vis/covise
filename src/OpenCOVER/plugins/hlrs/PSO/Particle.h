/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PSO_Particle_H
#define PSO_Particle_H

/*
 * Author: Florian Seybold, Professor Gerhard Venter
 * University of Stellenbosch and High-Performance Computing Center Stuttgart
 * Copyright: 2007, 2008
*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <cmath>
#ifndef WIN32
#include <sys/time.h>
#include <unistd.h>
#endif
#include <queue>
#include <set>
#include <sys/stat.h>

#include "MersenneTwister.h"

/// Namespace containing the Particle class as well as the Job-class, the slave-functions and various test functions
namespace pso
{

/// Class representing a particle in the swarm of the Particle Swarm Optimization (PSO) algorithm. This class includes particle own variables and functions, which are instantiated for every new particle object. It also includes static variables and functions which are shared among the whole swarm.
class Particle
{

public:
    /// Constructor of the Particle class. Randomly initializes the position and velocity vector of the particle.
    Particle();

    /// Virtual destructor.
    virtual ~Particle()
    {
    }

    /// Function updating the velocity of the particle using the present best position values of the particle and of the swarm.
    void updateVelocity();

    /// Function updating the position of the particle using the present velocity.
    void updatePosition();

    /// Function updating the response value at the newly computated position by directly calling the response function.
    void updateVal();

    /// Function comparing the present value with the best value found so far and the global best value, and updating them and the corresponding position, if the present value is better.
    void updateBestVal();

    /// Returns the identification number of the particle. Every particle object has an unique identification number.
    /**
		\return The number of the particle.
	**/
    int getNumPar();

    /// Returns the last evaluated response value.
    /**
		\return Present response value.
	**/
    double getVal();

    /// Set the last evaluated response value.
    /**
		\param setVal Present response value.
	**/
    void setVal(double setVal);

    /// Returns the best value found by the particle so far.
    /**
		\return Best value found by the particle so far.
	**/
    double getBestVal();

    /// Returns a pointer to an array representing the present position.
    /**
 		\return Pointer to the present position vector array.
	**/
    double *getX();

    /// Returns a pointer to an array representing the present velocity.
    /**
 		\return Pointer to the present velocity vector array.
	**/
    double *getV();

    /// Returns a pointer to an array representing the position of the best value found so far by the particle.
    /**
 		\return Pointer to the position of the particle's best position found so far.
	**/
    double *getBestX();

    /// Function updating the two beta values each representing the position of the two additional points, where the response value is to be evaluated at, in order to conduct a one dimensional search.
    void updateBeta();

    /// Function approximating the response curve along the velocity vector by using the information gathered by evaluating two additional points. The function then applies the minimum of the approximated response curve to a position update by scaling the velocity part of the position update.
    void approximateX();

    /// Function approximating the response curve along the velocity vector by using the information gathered by evaluating two additional points. The function then applies the minimum of the approximated response curve by scaling the present velocity vector.
    void approximateV();

    /// Function updating the response value at the two additional points by directly calling the response function.
    void updateValBeta();

    /// Set the last evaluated response value at additional point 1.
    /**
		\param setVal Present response value at additional point 1.
	**/
    void setValBeta1(double setVal);

    /// Set the last evaluated response value at additional point 2.
    /**
		\param setVal Present response value at additional point 2.
	**/
    void setValBeta2(double setVal);

    /// Returns an array representing the present position in the design space of the first additional point.
    /**
 		\return Position of the first additional point.
	**/
    double *getPositionBeta1();

    /// Returns an array representing the present position in the design space of the second additional point.
    /**
 		\return Position of the second additional point.
	**/
    double *getPositionBeta2();

    /// Function printing the state values of the particle. This function is used by the operator<< of the pso namespace.
    void printState(std::ostream &os) const;

    /// Function opening a statistic file for the particle.The name of the file is "particle.[id number of particle]".
    /**
		\param psofs File stream which is used to open the file. Data is written to the file by streaming it to this file stream.
	**/
    void openParticleStateFile(std::ofstream &psofs);

    /// Static function initializing the Particle class. This function should be called before any Particle objects are instantiated.
    /**
		\param setresponse Function pointer to the response function.
		\param setnvar Dimension of the optimization problem (number of input parameters of the response function).
		\param setlowerbound Array defining the lower bound of the search area. Same dimension as the optimization problem.
		\param setupperbound Array defining the upper bound of the search area. Same dimension as the optimization problem.
		\param setinteger Array defining whether one search dimension is continuous or discrete (using integer). Same dimension as the optimization problem.
		\param setdt Time step size. Normally dt=1.
	**/
    static void init(double (*setresponse)(double *), int setnvar, double *setlowerbound, double *setupperbound, bool *setinteger, double setdt);

    static void initSwarm(double (*setresponse)(double *), int setnvar, double *setlowerbound, double *setupperbound, bool *setinteger, double setdt);

    /// Static function destroying the Particle class. Should be called after optimization defined by the init() function is finished.
    static void destroy();

    /// Static helper function which calls the named Particle class function of every instantiated Particle objects.
    /**
		\param func Pointer to Particle class function which is to be called on every Particle object.
	**/
    static void all(void (Particle::*func)());

    /// Static function that dynamically updates (decreases) the particles' inertia by using swarm statistics.
    static void updateInertia();

    /// Static function that implements the craziness operator. When called, by statistical analysis sorted out particles  are going crazy: There position and velocity is randomly reinitialized.
    static void goCrazy();

    /// Returns the global best value found so far by the swarm.
    /**
		\return Best value found so far by the swarm.
	**/
    static double getGBestVal();

    /// Returns the position of the global best value found so far by the swarm.
    /**
		\return Array representing the global best position.
	**/
    static double *getGBestX();

    /// Static function computing the mean of the present response values of the particles.
    static void computeValsMean();

    /// Static function computing the standard deviatioin of the present response values of the particles.
    static void computeValsDev();

    /// Static function that computes the mean of the response values of the best particles.
    static void computeSortValsMean();

    /// Static function that computes the standard deviation of the response values of the best particles.
    static void computeSortValsDev();

    /// Static function that computes the element mean vector of the particles' position vector elements.
    static void computeXMean();

    /// Static function that computes the element mean vector of the particles' velocity vector elements.
    static void computeXDev();

    /// Static function that sorts the particles by their present response value. The sort is done by an insertion sort algorithm.
    static void sortVals();

    /// Returns an array which contains pointers to the instantiated particle objects.
    static Particle **getPars();

    /// Returns an array which contains pointers to the instantiated particle objects and is sorted by the particles best values. Function sortVals() has to be called first in order to get the latest sort of the particles.
    static Particle **getBestPars();

protected:
    /// Number (id) of this particle, starting with zero.
    int particle;

    /// Response value.
    double val;
    /// Position of particle.
    double *x;
    /// Velocity of particle.
    double *v;

    /// Best response value the particle found so far.
    double bestval;
    /// Best position of particle found so far.
    double *bestx;

    /// Random value 1.
    double r1;
    /// Random value 2.
    double r2;

    /// Line search parameter beta1.
    double beta1;
    /// Position in design space of line search parameter beta1.
    double *xbeta1;
    /// Line search parameter beta2.
    double beta2;
    /// Position in design space of line search parameter beta2.
    double *xbeta2;

    /// Response value at position beta1.
    double valbeta1;
    /// Response value at position beta2.
    double valbeta2;

    /// Enforcing constrains like search region bounds or integer values
    virtual void enforceConstrains();

    /// Number of particles.
    static int numpars;

    /// Array of instantiated particles.
    static Particle **par;
    /// Array of particles sorted by their response values.
    static Particle **bestpar;

    /// Function pointer to response function.
    static double (*response)(double *);

    /// Number of design variables (Size of optimization problem).
    static int nvar;

    /// Inertia.
    static double w;
    /// self-confidence factor.
    static double c1;
    /// swarm trust factor.
    static double c2;

    /// Lower bound of the search region.
    static double *lowerbound;
    /// Upper bound of the search region.
    static double *upperbound;

    /// Integer flags.
    static bool *integer;

    /// Global best response value found so far.
    static double gbestval;
    /// Global best position found so far.
    static double *gbestx;

    /// Time step size.
    static double dt;

    /// Mean of response values.
    static double meanval;
    /// Stand deviation of response values.
    static double devval;

    /// Mean of sorted response values.
    static double meansortval;
    /// Stand deviation of sorted response values.
    static double devsortval;

    /// Mean of position vector elements.
    static double *meanx;
    /// Standard deviation of position vector elements.
    static double *devx;

    /// Maximum number of particles.
    static const int MAXPARS;

    /// Maximum inertia.
    static const double MAXINERTIA;
    /// Minimum intertia.
    static const double MININERTIA;
    /// Fraction of particles (used by function computeSortValsMean() and computeSortValsDev() ).
    static const double FRACPARS;
    /// Fraction of inertia (used by function updateInertia() ).
    static const double FRACINERTIA;
    /// Covariance threshold of inertia.
    static const double COVTHRESHOLDINERTIA;
    /// Covariance threshold of craziness operator.
    static const double COVTHRESHOLDCRAZY;

    /// Defines the beta1 bound of the extend of the one dimensional search.
    static const double BOUNDBETA1;
    /// Defines the beta2 bound of the extend of the one dimensional search.
    static const double BOUNDBETA2;

    /// Mersenne Twister random number generator
    static MTRand *mtrand;
    /// Static function which initializes the random number generator.
    static void initRand();
    /// Static function which destroys the random number generator.
    static void destroyRand();
    /// Static function that returns a random number between zero and one.
    /**
		\return Random number: [0,1].
	**/
    static double rand();
};

/// Number of design variables of the griewank test function (griewank5_nvar = 5)
extern int griewank5_nvar;
/// Lower bounds of the search space of the griewank test function.
extern double griewank5_lbound[5];
/// Upper bounds of the search space of the griewank test function.
extern double griewank5_ubound[5];
/// Kinds of the variables: continuous or integer.
extern bool griewank5_integer[5];
/// The griewank test function.
/**
	\param x Array of input parameters (Position of a particle).
	\return Response value.
**/
double griewank5(double *x);

/// Number of design variables of the griewank test function (griewank3_nvar = 3)
extern int griewank3_nvar;
/// Lower bounds of the search space of the griewank test function.
extern double griewank3_lbound[3];
/// Upper bounds of the search space of the griewank test function.
extern double griewank3_ubound[3];
/// Kinds of the variables: continuous or integer.
extern bool griewank3_integer[3];
/// The griewank test function.
/**
	\param x Array of input parameters (Position of a particle).
	\return Response value.
**/
double griewank3(double *x);

/// Number of design variables of the griewank test function (griewank2_nvar = 2)
extern int griewank2_nvar;
/// Lower bounds of the search space of the griewank test function.
extern double griewank2_lbound[2];
/// Upper bounds of the search space of the griewank test function.
extern double griewank2_ubound[2];
/// Kinds of the variables: continuous or integer.
extern bool griewank2_integer[2];
/// The griewank test function.
/**
	\param x Array of input parameters (Position of a particle).
	\return Response value.
**/
double griewank2(double *x);

/// Number of design variables of the rosenbrock test function (rosenbrock3_nvar = 3).
extern int rosenbrock3_nvar;
/// Lower bounds of the search space of the rosenbrock test function.
extern double rosenbrock3_lbound[3];
/// Upper bounds of the search space of the rosenbrock test function.
extern double rosenbrock3_ubound[3];
/// Kinds of the variables: continuous or integer.
extern bool rosenbrock3_integer[3];
/// The rosenbrock test function.
/**
	\param x Array of input parameters (Position of a particle).
	\return Response value.
**/
double rosenbrock3(double *x);

/// Number of design variables of the rosenbrock test function (rosenbrock2_nvar = 2).
extern int rosenbrock2_nvar;
/// Lower bounds of the search space of the rosenbrock test function.
extern double rosenbrock2_lbound[2];
/// Upper bounds of the search space of the rosenbrock test function.
extern double rosenbrock2_ubound[2];
/// Kinds of the variables: continuous or integer.
extern bool rosenbrock2_integer[2];
/// The rosenbrock test function.
/**
	\param x Array of input parameters (Position of a particle).
	\return Response value.
**/
double rosenbrock2(double *x);
}

#endif
