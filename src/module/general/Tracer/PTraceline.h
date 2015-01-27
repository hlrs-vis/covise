/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS PTraceline
//
//  Base class for classes related with trajectory integration
//
//  Initial version: 2001-12-07 sl
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:
#ifndef _P_TRACELINE_H_
#define _P_TRACELINE_H_

#include "PTask.h"

/**
 * From this class we derive all PTask classes that generate a trajectory
 *
 */
class PTraceline : public PTask
{
public:
    /** Get array for x-coordinates of the integrated line.
       * @return             Pointer to first array element.
       */
    float *x_c()
    {
        return ((p_c_[0].size() > 0) ? &p_c_[0][0] : NULL);
    }
    /** Get array for y-coordinates of the integrated line.
       * @return             Pointer to first array element.
       */
    float *y_c()
    {
        return ((p_c_[1].size() > 0) ? &p_c_[1][0] : NULL);
    }
    /** Get array for z-coordinates of the integrated line.
       * @return             Pointer to first array element.
       */
    float *z_c()
    {
        return ((p_c_[2].size() > 0) ? &p_c_[2][0] : NULL);
    }
    /** Get array for the output magnitude mapped onto the integrated line.
       * @return             Pointer to first array element.
       */
    float *m_c()
    {
        return ((m_c_.size() > 0) ? &m_c_[0] : NULL);
    }
    virtual float *m_c_interpolate(vector<const coDistributedObject *> &sfield0,
                                   vector<const coDistributedObject *> &sfield1);
    /** Get array for the time mapped onto the integrated line.
       * @return             Pointer to first array element.
       */
    float *t_c()
    {
        return ((t_c_.size() > 0) ? &t_c_[0] : NULL);
    }
    /** Get array for the X velocity.
       * @return             Pointer to first array element.
       */
    float *u_c()
    {
        return ((u_c_.size() > 0) ? &u_c_[0] : NULL);
    }
    /** Get array for the Y velocity.
       * @return             Pointer to first array element.
       */
    float *v_c()
    {
        return ((v_c_.size() > 0) ? &v_c_[0] : NULL);
    }
    /** Get array for the Z velocity.
       * @return             Pointer to first array element.
       */
    float *w_c()
    {
        return ((w_c_.size() > 0) ? &w_c_[0] : NULL);
    }
    /** Get length of the point coordinate and output magnitude arrays.
       * @return             Array length.
       */
    int num_points()
    {
        return (int)p_c_[0].size();
    }
    /** Constructor
       * @param x_ini X coordinate of initial point
       * @param y_ini Y coordinate of initial point
       * @param z_ini Z coordinate of initial point
       * @param grids0 list of grids
       * @param vels0 list of velocity objects
       * @param ts time direction
       */
    PTraceline(float x_ini, float y_ini, float z_ini,
               std::vector<const coDistributedObject *> &grids0, std::vector<const coDistributedObject *> &vels0, int ts = 1);
    /// Destructor.
    ~PTraceline()
    {
    }

protected:
    // release time
    virtual float OutputTime(float integrationTime) const;
    float release_time_;
    // 3 lists for the X, Y and Z coordinates of the integrated line
    std::vector<float> p_c_[3];
    std::vector<float> m_c_; // list for evaluated magnitude
    std::vector<float> t_c_; // list of integrated time
    std::vector<float> u_c_; // list for X velocity component
    std::vector<float> v_c_; // list for Y velocity component
    std::vector<float> w_c_; // list for Z velocity component
    int ts_; // time direction
    // Internal class which keeps information of the actual cell and grid
    struct gridAndCell
    {
        int grid_back_;
        int cell_back_[3];
        int grid_;
        int cell_[3];
        void notFound();
        void backUp();
        void restore();
        void setGrid(int i);
        int whichGrid();
        gridAndCell();
        gridAndCell(const gridAndCell &cp);
        gridAndCell &operator=(const gridAndCell &cp);
    } previousCell_0_;
    // Find the velocity givena a point, a grid and perhaps also a cell (if *cell>=0)
    status derivsForAGrid(const float *y, float *ydot,
                          const coDistributedObject *grid, const coDistributedObject *velo, int *cell,
                          int search_level);
    // add an integrated point to the result lists
    void addPoint(float time, float *posi, float *velo, int kount, int number);
    static const float TINY; // constant used by the integrator
    // length of the part of a line which lies out of the domain
    float getOutLength(const float *y, int kount_out_domain);

private:
};
#endif
