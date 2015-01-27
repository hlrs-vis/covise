/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "PStreamline.h"
#include <math.h>

extern float minimal_velocity;

// returns status (do not set it)
PTask::status
PStreamline::derivs(float x, // time
                    const float *y, // position
                    float *ydot, // velocity
                    int allGrids, // if !=0 looks in all grids if necessary
                    int search_level)
// active if emergency_ == GOT_OUT_OF_DOMAIN
{
    (void)x;
    status ret = FINISHED_DOMAIN;
    int i;

    // get "suggestion" for initial grid...
    int iniGrid = previousCell_0_.whichGrid();

    if (iniGrid < 0 || iniGrid >= grids0_->size()) // no hint available
    {
        previousCell_0_.notFound();
    } // ...and use it
    else
    {
        ret = derivsForAGrid(y, ydot, grids0_->operator[](iniGrid),
                             vels0_->operator[](iniGrid),
                             previousCell_0_.cell_, search_level);
        if (emergency_ == OK)
        {
            return ret;
        }
    }

    // if there was no useful suggestion or (it was not found in the suggested
    // grid and emergency != OK) -> look for other grids, but in the latter
    // case, do not use the ydot output values...
    float int_ydot[3];
    if (previousCell_0_.grid_ < 0 || (ret == FINISHED_DOMAIN && emergency_ == GOT_OUT_OF_DOMAIN && allGrids))
    {
        // fprintf(stderr,"derivs: x: %f, y: %f %f %f\n",x,*y,*(y+1),*(y+2));
        for (i = 0; i < grids0_->size(); ++i)
        {
            if (i == iniGrid)
                continue;
            // fprintf(stderr,"   derivs: Checking %d, (%d)\n",i,previousCell_0_.grid_);
            previousCell_0_.cell_[0] = -1;
            previousCell_0_.cell_[1] = -1;
            previousCell_0_.cell_[2] = -1;
            ret = derivsForAGrid(y, (previousCell_0_.grid_ < 0) ? ydot : int_ydot,
                                 grids0_->operator[](i),
                                 vels0_->operator[](i),
                                 previousCell_0_.cell_, search_level);
            if (ret == SERVICED)
            {
                if (previousCell_0_.grid_ >= 0)
                {
                    ydot[0] = int_ydot[0];
                    ydot[1] = int_ydot[1];
                    ydot[2] = int_ydot[2];
                }
                previousCell_0_.setGrid(i);
                break;
            }
        }
    }
    return ret;
}

extern float divide_cell;
extern float max_out_of_cell;
extern int search_level_polygons;

// driver for rkqs stepper
void
PStreamline::Solve(float eps, // relative error
                   float eps_abs) // absolute error
{
    // see documentation in Numerical Recipes...
    int i;
    int kount;
    int nok, nbad;
    int kount_in_domain = 0;
    int kount_out_domain = 0;
    //   int kount_in_ok=0;
    float x, hnext, hdid, h;
    float yscal[3], y[3], dydx[3];
    float length = 0.0, aux1, aux2;
    status problems;

    x = 0.0;
    nok = nbad = kount = 0;

    for (i = 0; i < 3; i++)
        y[i] = ini_point_[i];
    problems = derivs(x, y, dydx, 0, search_level_polygons);
    if (problems == FINISHED_DOMAIN)
    {
        set_status(FINISHED_DOMAIN);
        return;
    }
    else
    {
        previousCell_0_.backUp();
        addPoint(x, y, dydx, kount, number_);
        hintRegister_.push_back(previousCell_0_);
        ++kount;
    }
    h = ts_ * suggestInitialH(previousCell_0_.cell_,
                              grids0_->operator[](previousCell_0_.grid_),
                              vels0_->operator[](previousCell_0_.grid_));
    if (h == 0.0)
    {
        set_status(FINISHED_STEP);
        return;
    }
    h *= divide_cell;

    while (kount < max_points_)
    {
        // scale for relative error calculation
        for (i = 0; i < 3; i++)
            yscal[i] = fabs(y[i]) + fabs(dydx[i] * h) + TINY;

        // calculate next point
        float x_old, y_old[3];
        x_old = x;
        y_old[0] = y[0];
        y_old[1] = y[1];
        y_old[2] = y[2];

        // reduce h if necessary in case we want values at some
        // particular points.
        float h_old = h;
        h_Reduce(&h);
        problems = rkqs(y, dydx, 3, &x, h, eps, eps_abs, yscal, &hdid, &hnext);
        // cerr<<"x: "<<x<<", h: "<<h<<", hdid: "<<hdid<<", hnext: "<<hnext<<endl;
        ammendNextTime(hdid, h, h_old);

        // test intermediate calculations (this is what problems informs about)
        if (emergency_ == OK && problems == FINISHED_DOMAIN)
        {
            // emergency_ = GOT_OUT_OF_DOMAIN;
            kount_in_domain = 0;
            kount_out_domain = 0;
            // restore state of integration (x, y, dydx)?
            // No, rkqs has not modified it in this case.
            // But do restore previousCell_0_... !!!!!!!!!!!!!!!!!!
            previousCell_0_.restore();
            // Change h control
            float h_ctrl;
            h_ctrl = ts_ * suggestInitialH(previousCell_0_.cell_,
                                           grids0_->operator[](previousCell_0_.grid_),
                                           vels0_->operator[](previousCell_0_.grid_));
            if (h_ctrl == 0.0)
            {
                set_status(FINISHED_STEP);
                return;
            }
            if (fabs(h) < fabs(h_ctrl))
            {
                emergency_ = GOT_OUT_OF_DOMAIN;
            }
            h = h_ctrl;
            h *= divide_cell;
            continue;
        }
        previousCell_0_.backUp();

        problems = derivs(x, y, dydx, 1, -1);

        // test final point
        if (emergency_ == OK && problems == FINISHED_DOMAIN)
        {
            // emergency_ = GOT_OUT_OF_DOMAIN;
            kount_in_domain = 0;
            kount_out_domain = 0;
            // restore state of integration (x, y, dydx)?
            // No, rkqs has not modified it in this case.
            // But do restore previousCell_0_...
            x = x_old;
            y[0] = y_old[0];
            y[1] = y_old[1];
            y[2] = y_old[2];
            previousCell_0_.restore();
            // Change h control
            float h_ctrl;
            h_ctrl = ts_ * suggestInitialH(previousCell_0_.cell_,
                                           grids0_->operator[](previousCell_0_.grid_),
                                           vels0_->operator[](previousCell_0_.grid_));
            if (h_ctrl == 0.0)
            {
                set_status(FINISHED_STEP);
                return;
            }
            if (fabs(h) < fabs(h_ctrl))
            {
                emergency_ = GOT_OUT_OF_DOMAIN;
            }
            h = h_ctrl;
            h *= divide_cell;
            continue;
        }
        else if (emergency_ == GOT_OUT_OF_DOMAIN && problems == FINISHED_DOMAIN)
        {
            // the last call to derivs had a 1 as last parameter
            // and we may have overwritten previousCell_0_.cell_
            previousCell_0_.restore();
            if (getOutLength(y, kount_out_domain)
                > max_out_of_cell * suggestInitialH(previousCell_0_.cell_,
                                                    grids0_->operator[](previousCell_0_.grid_),
                                                    NULL))
            {
                int crop = kount_out_domain;
                if (crop < 0)
                    crop = 0;
                p_c_[0].resize(p_c_[0].size() - crop);
                p_c_[1].resize(p_c_[1].size() - crop);
                p_c_[2].resize(p_c_[2].size() - crop);
                m_c_.resize(m_c_.size() - crop);
                t_c_.resize(t_c_.size() - crop);
                u_c_.resize(u_c_.size() - crop);
                v_c_.resize(v_c_.size() - crop);
                w_c_.resize(w_c_.size() - crop);
                hintRegister_.resize(hintRegister_.size() - kount_out_domain);
                set_status(FINISHED_DOMAIN);
                break;
            }
            // continue counting points which are out of domain
            ++kount_out_domain;
        }
        else if (emergency_ == GOT_OUT_OF_DOMAIN && problems == SERVICED)
        {
            // have we landed in a new grid?
            // no, the same
            if (previousCell_0_.grid_ == previousCell_0_.grid_back_)
            {
                ++kount_in_domain;
                if (kount_in_domain > 4 / divide_cell)
                {
                    // seems that the alarm was not correct, so return to normality
                    // kount_in_ok=0;
                    emergency_ = OK;
                }
            }
            else
            {
                // found new grid!!!
                // kount_in_ok=0;
                emergency_ = OK;
                // keep the new info in safety!!!
                previousCell_0_.backUp();
                // this might be taken as a good starting point
                // for the new grid, but we may have penetrated it
                // too much. It may be better to check the intermediate
                // results.
                status ctrl;
                for (i = 0; i < 4; ++i)
                {
                    if (intermediate_[i].status_ == FINISHED_DOMAIN)
                    {
                        // well, we know this intermediate point was not in the
                        // previous grid, but this does not guarantee that
                        // it is in the actual grid. Let us check this.
                        ctrl = derivs(intermediate_[i].time_, &intermediate_[i].var_[0],
                                      &intermediate_[i].var_dot_[0], 0, -1);
                        if (ctrl == FINISHED_DOMAIN)
                        {
                            previousCell_0_.restore(); // no, it isn't
                        }
                        else
                        {
                            x = intermediate_[i].time_;
                            y[0] = intermediate_[i].var_[0];
                            y[1] = intermediate_[i].var_[1];
                            y[2] = intermediate_[i].var_[2];
                            dydx[0] = intermediate_[i].var_dot_[0];
                            dydx[1] = intermediate_[i].var_dot_[1];
                            dydx[2] = intermediate_[i].var_dot_[2];
                            previousCell_0_.backUp();
                            break;
                        }
                    }
                }
            }
        }

        // Write point to output
        addPoint(x, y, dydx, kount, number_);
        hintRegister_.push_back(previousCell_0_);
        if (kount)
        {
            for (i = 0, aux2 = 0.0; i < 3; i++)
            {
                aux1 = p_c_[i][kount] - p_c_[i][kount - 1];
                aux2 += aux1 * aux1;
            }
            length += sqrt(aux2);
        }
        kount++;

        if (interruptIntegration(dydx, kount, length, x))
        {
            break;
        }

#ifdef _DEBUG_
        fprintf(stderr, "odeint: eps: %f h: %f hdid: %f\n", eps, h, hdid);
#endif
        if (hdid == h)
            ++nok;
        else
            ++nbad;
        h = hnext;
    }
    if (kount >= max_points_)
    {
        set_status(FINISHED_POINTS);
    }
}

#undef TINY

PStreamline::PStreamline(real x_ini, // X coordinate of initial point
                         real y_ini, // Y coordinate of initial point
                         real z_ini, // Z coordinate of initial point
                         real max_length, // length limit
                         int max_points, // number of points limit
                         std::vector<const coDistributedObject *> &grids0, // list of grids
                         std::vector<const coDistributedObject *> &vels0, // list of grids
                         int ts, // time direction (+1, or -1)
                         int number)
    : // number of streamline
    PTraceline(x_ini, y_ini, z_ini, grids0, vels0, ts)
{
    max_length_ = max_length;
    max_points_ = max_points;
    number_ = number;
}

// check if there is a reason to interrupt integration
int
    // velocity
    PStreamline::interruptIntegration(const float *dydx,
                                      int kount, // number of points
                                      float length, // length
                                      float)
{
    // velocity test
    if (dydx[0] * dydx[0] + dydx[1] * dydx[1] + dydx[2] * dydx[2] < minimal_velocity * minimal_velocity)
    {
        set_status(FINISHED_VELOCITY);
        return 1;
    }

    // test number of points
    if (kount >= max_points_)
    {
        return 1;
    }

// test length
#ifdef _DEBUG_
    fprintf(stderr, "odeint: length: %f max_length_: %f\n", length, max_length_);
#endif
    if (length > max_length_)
    {
        set_status(FINISHED_LENGTH);
        return 1;
    }
    return 0;
}

// do nothing for a PStreamline
void
PStreamline::h_Reduce(float *h)
{
    (void)h;
}

// do nothing for a PStreamline
void
PStreamline::ammendNextTime(float, float, float)
{
}

void
PStreamline::interpolateField(const vector<const coDistributedObject *> &field,
                              vector<float> &interpolation)
{
    vector<float> l_interpolation;
    for (unsigned int point = 0; point < hintRegister_.size(); ++point)
    {
        float x = p_c_[0][point];
        float y = p_c_[1][point];
        float z = p_c_[2][point];

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // The interpolateFieldInGrid function makes use, after a series of subsequent calls to          //
        // other functions, of the testACell function defined in covise_unstrgrd which in turn            //
        // determines initially if the integration point is contained in a given cell before                       //
        // interpolating the values at the vertices of the cell.  Only the interpolation step is                //
        // necessary in this case since the whole streamline has been already calculated by the       //
        // time the PStreamline::m_c_interpolate function is called.  This is particularly important      //
        // in the case of polyhedral cells since the current in-polyhedron test is computationally       //
        // very expensive.  A modified version of the interpolateFieldInGrid function based on a       //
        // scattered data interpolation method is implemented next.  To use the original algorithm   //
        // it is only necessary to comment the newer mapFieldInGrid function and uncomment the //
        // older interpolateFieldInGrid function.                                                                                  //
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //       float value =
        //          interpolateFieldInGrid((*grids0_)[hintRegister_[point].grid_],
        //          field[hintRegister_[point].grid_],
        //          hintRegister_[point].cell_,
        //          x,y,z);

        float value = mapFieldInGrid((*grids0_)[hintRegister_[point].grid_],
                                     field[hintRegister_[point].grid_],
                                     hintRegister_[point].cell_,
                                     x, y, z);

        l_interpolation.push_back(value);
    }
    l_interpolation.swap(interpolation);
}

float *
PStreamline::m_c_interpolate(vector<const coDistributedObject *> &sfield,
                             vector<const coDistributedObject *> &)
{
    vector<float> interpolation;
    interpolateField(sfield, interpolation);
    assert(m_c_.size() == interpolation.size());
    unsigned int i;
    for (i = 0; i < interpolation.size(); ++i)
    {
        m_c_[i] = interpolation[i];
    }
    return m_c();
}
