/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#define COMP_S 0
#define COMP_S_COL0 1
#define COMP_S_COL1 2
#define COMP_S_COL2 3
#define COMP_M 4

#include "../Mz/Mz_lib.cpp"
#include <climits>
#include <cfloat>

void vortex_criteria_impl(UniSys *us,
                          Unstructured *unst_in, int compVelo,
                          Unstructured *unst_in_time_deriv, int compVeloTimeDeriv,
                          //Unstructured *unst_vorticity,
                          Unstructured *unst_scalar, int quantityNr,
                          int smoothing_range,
                          char *quantity_name)
{ // velocity gradient must already be computed inside unst_in for compVelo
    // unst_in_time_deriv: may be NULL, otherwise gradient must already be computed
    // quantity_name: must be NULL or large enough

    // "global" methods ---------------------------------------------------------

    // Mz strong, according to Haller ("objective definition of a vortex")
    if ((quantityNr == 8) || (quantityNr == 9))
    {

        if (quantityNr == 8)
        {
            us->info("Computing Mz");
            if (quantity_name)
                strcpy(quantity_name, "Mz");
        }
        else
        {
            us->info("Computing Mz strong hyperbolic (M is positive definite)");
            if (quantity_name)
                strcpy(quantity_name, "Mz strong");
        }

        if (!unst_in_time_deriv)
        {
            us->warning("Mz method: unsteady data would require temporal derivative input");
        }

        Unstructured *temp = NULL;
        {
            int components[256] = {
                9, // S
                3, // S col 0
                3, // S col 1
                3, // S col 2
                9 // M
            };
            temp = new Unstructured(unst_in, 5, components);
        }

        // compute S
        for (int n = 0; n < unst_in->nNodes; n++)
        {
            // get velocity gradient at node
            fmat3 fgrad;
            mat3 grad;
            unst_in->getMatrix3(n, compVelo, Unstructured::OP_GRADIENT, fgrad);
            fmat3tomat3(fgrad, grad);

            // compute S
            mat3 S;
            mat3symm(grad, S);

            // store S
            fmat3 Sf;
            mat3tofmat3(S, Sf);
            temp->setMatrix3(n, COMP_S, Sf);

            // store columns of S
            vec3 Scol0, Scol1, Scol2;
            mat3getcols(S, Scol0, Scol1, Scol2);
            temp->setVector3(n, COMP_S_COL0, Scol0);
            temp->setVector3(n, COMP_S_COL1, Scol1);
            temp->setVector3(n, COMP_S_COL2, Scol2);
        }

        // compute gradient of S
        us->moduleStatus("computing gradient", 75);
        temp->gradient(COMP_S_COL0, false, smoothing_range);
        temp->gradient(COMP_S_COL1, false, smoothing_range);
        temp->gradient(COMP_S_COL2, false, smoothing_range);

        // compute M
        for (int n = 0; n < unst_in->nNodes; n++)
        {

            // get velocity
            vec3 velo;
            unst_in->getVector3(n, compVelo, velo);

            // time derivative of S
            // #### TODO: unsteady: add d/dt S
            mat3 Sdot;
            {

                // get gradient of S tensor
                fmat3 gradScol0f, gradScol1f, gradScol2f;
                mat3 gradScol0, gradScol1, gradScol2;
                temp->getMatrix3(n, COMP_S_COL0, Unstructured::OP_GRADIENT, gradScol0f);
                temp->getMatrix3(n, COMP_S_COL1, Unstructured::OP_GRADIENT, gradScol1f);
                temp->getMatrix3(n, COMP_S_COL2, Unstructured::OP_GRADIENT, gradScol2f);
                fmat3tomat3(gradScol0f, gradScol0);
                fmat3tomat3(gradScol1f, gradScol1);
                fmat3tomat3(gradScol2f, gradScol2);
                mat3 gradSx, gradSy, gradSz;
                vec3 c0x, c0y, c0z, c1x, c1y, c1z, c2x, c2y, c2z;
                mat3getcols(gradScol0, c0x, c0y, c0z);
                mat3getcols(gradScol1, c1x, c1y, c1z);
                mat3getcols(gradScol2, c2x, c2y, c2z);
                mat3setcols(gradSx, c0x, c1x, c2x);
                mat3setcols(gradSy, c0y, c1y, c2y);
                mat3setcols(gradSz, c0z, c1z, c2z);

                // tensor / vector multiplication
                {
                    mat3scal(gradSx, velo[0], gradSx);
                    mat3scal(gradSy, velo[1], gradSy);
                    mat3scal(gradSz, velo[2], gradSz);

                    mat3copy(gradSx, Sdot);
                    mat3add(Sdot, gradSy, Sdot);
                    mat3add(Sdot, gradSz, Sdot);
                }

                if (unst_in_time_deriv)
                {

                    // get gradient of time derivative of velocity at node
                    fmat3 fgrad;
                    mat3 grad;
                    unst_in_time_deriv->getMatrix3(n, compVeloTimeDeriv, Unstructured::OP_GRADIENT, fgrad);
                    fmat3tomat3(fgrad, grad);

                    // compute d/dt S
                    mat3 Sddt;
                    mat3symm(grad, Sddt);

                    mat3add(Sdot, Sddt, Sdot);
                }
            }

            // get S
            fmat3 Sf;
            mat3 S;
            temp->getMatrix3(n, COMP_S, Sf);
            fmat3tomat3(Sf, S);

            // get velocity gradient at node
            fmat3 fgrad;
            mat3 grad;
            unst_in->getMatrix3(n, compVelo, Unstructured::OP_GRADIENT, fgrad);
            fmat3tomat3(fgrad, grad);

            // S * grad
            mat3 SmulGrad;
            mat3mul(S, grad, SmulGrad);

            // gradT * S
            mat3 gradT, gradTmulS;
            mat3trp(grad, gradT);
            mat3mul(gradT, S, gradTmulS);

            mat3 M;
            mat3copy(Sdot, M);
            mat3add(M, SmulGrad, M);
            mat3add(M, gradTmulS, M);

            fmat3 Mf;
            mat3tofmat3(M, Mf);
            temp->setMatrix3(n, COMP_M, Mf);
        }

        // compute Mz and its definiteness
        for (int n = 0; n < unst_in->nNodes; n++)
        {

            // get S
            fmat3 Sf;
            mat3 S;
            temp->getMatrix3(n, COMP_S, Sf);
            fmat3tomat3(Sf, S);

            // get M
            fmat3 Mf;
            mat3 M;
            temp->getMatrix3(n, COMP_M, Mf);
            fmat3tomat3(Mf, M);

            // evaluate Mz definiteness
            if (quantityNr == 8)
            {
                // "weak" hyperbolicity (Mz is positive definite)

                if (Mz_is_positive_definite(us, S, M))
                {
                    unst_scalar->setScalar(n, 1.0);
                }
                else
                {
                    unst_scalar->setScalar(n, 0.0);
                }
            }
            else
            {
                // strong hyperbolicity (M is positive definite)

                double Smag2 = mat3magFrobeniusSqr(S);
                double Mmag2 = mat3magFrobeniusSqr(M);

                // values > 0 indicate hyperbolic (non-vortex) regions
                double value = (4 * Smag2 * Smag2 - Mmag2) * Smag2 - mat3det(M);

                // store
                if (mat3det(M) <= 0)
                {
// not strong hyperbolic -> hyperbolic or elliptic -> set to default
#if 0
          unst_scalar->setScalar(n, -FLT_MAX); // ### -FLT_MAX is a HACK
#else
                    unst_scalar->setScalar(n, 0.0);
#endif
                }
                else
                {
#if 0
          unst_scalar->setScalar(n, value);
#else
                    if (value > 0.0)
                    {
                        unst_scalar->setScalar(n, value);
                    }
                    else
                    {
                        unst_scalar->setScalar(n, 0.0);
                    }
#endif
                }
            }
        }

        delete temp;
    }

    // divergence of acceleration
    else if (quantityNr == 10)
    {
        us->info("Computing divergence of acceleration");
        if (quantity_name)
            strcpy(quantity_name, "div accel");

// create temporary vector field for acceleration
#if 1 // TODO: replace by simple constructor, one day it is debugged
        float *data = new float[3 * unst_in->nNodes];
        DataDesc dd = DataDesc(0, Unstructured::TP_FLOAT, 3, data);
        Unstructured *accel = new Unstructured(unst_in, &dd);
#endif

        // compute acceleration (stationary) TODO
        unst_in->matVec(compVelo, Unstructured::OP_GRADIENT,
                        unst_in, compVelo, accel, 0);

        // compute accel gradient
        accel->gradient(0, false, smoothing_range);

        us->moduleStatus("computing gradient", 75);

        for (int n = 0; n < unst_in->nNodes; n++)
        {

            // get accel gradient
            fmat3 fgrad;
            mat3 grad;
            accel->getMatrix3(n, 0, Unstructured::OP_GRADIENT, fgrad);
            fmat3tomat3(fgrad, grad);

            // compute and set divergence
            unst_scalar->setScalar(n, mat3trace(grad));
        }

        delete accel;
#if 1
        delete[] data;
#endif
    }

    else
    {

        // "local" methods --------------------------------------------------------

        for (int n = 0; n < unst_in->nNodes; n++)
        {

            // get velocity
            vec3 velo;
            unst_in->getVector3(n, compVelo, velo);

            // get velocity gradient at node
            fmat3 fgrad;
            mat3 grad;
            unst_in->getMatrix3(n, compVelo, Unstructured::OP_GRADIENT, fgrad);
            fmat3tomat3(fgrad, grad);

            // compute vorticity
            vec3 curl;
            mat3omega(grad, curl);
            //unst_vorticity->setVector3(n, curl);

            // compute the scalar output value
            {
                double sclr = 0.0;

                // (normalized) helicity
                if (quantityNr == 1)
                {
                    if (n == 0)
                    {
                        us->info("Computing helicity");
                        if (quantity_name)
                            strcpy(quantity_name, "helicity");
                    }
                    sclr = vec3dot(velo, curl);
                    unst_scalar->setScalar(n, fabs(sclr));
                }

                // velo-normalized helicity
                else if (quantityNr == 2)
                {
                    if (n == 0)
                    {
                        us->info("Computing velocity-normalized helicity");
                        if (quantity_name)
                            strcpy(quantity_name, "velo-norm helicity");
                    }
                    double denom = sqrt(vec3dot(velo, velo));
                    if (denom == 0)
                        sclr = 0;
                    else
                    {
                        double numer = vec3dot(velo, curl);
                        sclr = numer / denom;
                    }
                    unst_scalar->setScalar(n, fabs(sclr));
                }

                // vorticity magnitude
                else if (quantityNr == 3)
                {
                    if (n == 0)
                    {
                        us->info("Computing vorticity magnitude");
                        if (quantity_name)
                            strcpy(quantity_name, "vorticity mag");
                    }
                    sclr = sqrt(vec3dot(curl, curl));
                    unst_scalar->setScalar(n, sclr);
                }

                // z component of vorticity
                else if (quantityNr == 4)
                {
                    if (n == 0)
                    {
                        us->info("Computing z-component of vorticity");
                        if (quantity_name)
                            strcpy(quantity_name, "z vorticity");
                    }
                    sclr = curl[2];
                    unst_scalar->setScalar(n, sclr);
                }

                // lambda 2
                else if (quantityNr == 5)
                {
                    if (n == 0)
                    {
                        us->info("Computing lambda 2");
                        if (quantity_name)
                            strcpy(quantity_name, "lambda 2");
                    }
                    mat3 s, s2, omega, omega2, m;
                    vec3 lambda;

                    mat3symm(grad, s); // symmetric part
                    mat3asymm(grad, omega); // antisymmetric part

                    mat3mul(s, s, s2);
                    mat3mul(omega, omega, omega2);

                    mat3add(s2, omega2, m);

                    // force m to be symmetric
                    mat3symm(m, m);

                    int nr = mat3eigenvalues(m, lambda);
                    if (nr != 3)
                    {
#if 0
            //printf("Complex eigenvalues!\n");
            if (lambda[2] > 1e-8) {
              printf("warning: fixing complex eigenvalues (i > 1e-8)\n");
              //exit(1); // ###
              lambda[2] = lambda[1]; // double eigenvalue
            }
            else lambda[2] = lambda[1]; // double eigenvalue
#else
                        printf("error with eigenvalues at node=%d! l1=%g, l2=%g, l3=%g\n",
                               n, lambda[0], lambda[1], lambda[2]);
                        vec3dump(velo, stdout);
                        mat3dump(grad, stdout);
                        mat3dump(m, stdout);
                        lambda[2] = lambda[1];

                        printf("second time grad[n+1]:\n");
                        {
                            fmat3 fgrad;
                            mat3 grad;
                            unst_in->getMatrix3(n + 1, compVelo, Unstructured::OP_GRADIENT, fgrad);
                            fmat3tomat3(fgrad, grad);
                            mat3dump(grad, stdout);
                        }
#endif
                    }
                    // sort eigenvalues
                    if (lambda[0] < lambda[1])
                        swap(lambda[0], lambda[1]);
                    if (lambda[0] < lambda[2])
                        swap(lambda[0], lambda[2]);
                    if (lambda[1] < lambda[2])
                        swap(lambda[1], lambda[2]);

                    sclr = lambda[1];

                    // Try eliminating (setting to 0) those values
                    // where Q is negative (Q positive is Hunt's vortex
                    // criterion).
                    //
                    //double q = -(m[0][0] + m[1][1] + m[2][2])/2.;
                    //if (q < 0) sclr = 0;

                    unst_scalar->setScalar(n, sclr);
                }

                // Q: (|Omega|^2 - |S|^2) / 2 > 0
                else if (quantityNr == 6)
                {
                    if (n == 0)
                    {
                        us->info("Computing Q");
                        if (quantity_name)
                            strcpy(quantity_name, "Q");
                    }
                    mat3 s, omega;

                    mat3symm(grad, s); // symmetric part
                    mat3asymm(grad, omega); // antisymmetric part

                    sclr = 0.5 * (mat3magFrobeniusSqr(omega) - mat3magFrobeniusSqr(s));
                    unst_scalar->setScalar(n, sclr);
                }

                // Delta: (Q/3)^3 + (det grad velo / 2)^2 > 0
                else if (quantityNr == 7)
                {
                    if (n == 0)
                    {
                        us->info("Computing Delta");
                        if (quantity_name)
                            strcpy(quantity_name, "Delta");
                    }
                    mat3 s, omega;

                    mat3symm(grad, s); // symmetric part
                    mat3asymm(grad, omega); // antisymmetric part

                    double q = 0.5 * (mat3magFrobeniusSqr(omega) - mat3magFrobeniusSqr(s));

                    double q3 = q * q * q;
                    double gradDet = mat3det(grad);
                    double gradDet2 = gradDet * gradDet;
                    sclr = q3 / 27 + gradDet2 / 4;
                    unst_scalar->setScalar(n, sclr);
                }

                // divergence
                else if (quantityNr == 11)
                {
                    if (n == 0)
                    {
                        us->info("Computing divergence");
                        if (quantity_name)
                            strcpy(quantity_name, "divergence");
                    }
                    unst_scalar->setScalar(n, mat3trace(grad));
                }
            }
        }
    }
}
