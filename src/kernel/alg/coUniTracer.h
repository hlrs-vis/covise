/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS coUniTracer
//
//  An object of this class may be used to work out
//  a streamline in a uniform grid.
//
//  Initial version: 24.06.2004 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef CO_UNI_TRACER_H
#define CO_UNI_TRACER_H

#include <vector>
#include <util/coExport.h>

namespace covise
{

// coUniState is an auxiliary structure
// representing a "state" during particle integration.
// It consists of point, velocity and integration time
struct coUniState
{
    float point_[3];
    float velocity_[3];
    float time_;
};

class ALGEXPORT coUniTracer
{
public:
    /// constructor specifying an input uniform grid
    /// and an input velocity field
    coUniTracer(float xmin, float xmax, float ymin, float ymax,
                float zmin, float zmax, int nx, int ny, int nz,
                const float *u, const float *v, const float *w);
    /// desrtuctor
    virtual ~coUniTracer();
    /// solve is a driver using a RK-stepper, after
    /// compkletion you get the solution in parameter solution
    /// yini is an array of three float with the initial point,
    /// solution is the output streamline
    /// eps_rel is the relative error used by the stepper
    /// length_factor is the streamline maximal length relative
    /// to the model characteristic dimension
    /// and vel_factor is used to specified when the integration
    /// is interrupted for too low speeds (a speed is too low when
    /// smaller than this factor multiplied by the highest speed
    /// in the model).
    void solve(const float *yini, std::vector<coUniState> &solution, float eps_rel,
               float max_length = 2.0, float vel_factor = 0.005, int ts = 1) const;

protected:
private:
    // interpolate returns true when a point y is in the the uniform grid
    // and writes the interpolated speed in dydx
    bool interpolate(const float *y, float *dydx) const;
    // getInitialH returns a "reasonable" value for the integration
    // step given a point and a speed (previously calculated with interpolate)
    float getInitialH(const float *y, const float *dydx) const;
    // stop returns true when the integration is to be interrupted because
    // the speed is too low
    bool stop(float factor, const std::vector<coUniState> &solu) const;
    // rkqs is a stepper using rkck. See Tracer module for documentation
    bool rkqs(float *y, const float *dydx, float *x,
              float h, float eps, float eps_abs,
              const float *yscal,
              float *hdid, float *hnext) const;

    // rkck performs a Runge-Kutta step. See Tracer module for documentation
    bool rkck(const float *y, const float *dydx,
              float x, float h, float yout[], float yerr[]) const;

    // uniform grid dimensions
    float xmin_;
    float ymin_;
    float zmin_;
    float xmax_;
    float ymax_;
    float zmax_;
    // model characteristic length
    float model_length_;
    // number of node divisions
    int nx_;
    int ny_;
    int nz_;
    // grid cell dimensions
    float dx_;
    float dy_;
    float dz_;
    // velocity field arrays
    const float *u_;
    const float *v_;
    const float *w_;
    // highest velocity
    float max_vel_;
    // time direction
};
}
#endif
