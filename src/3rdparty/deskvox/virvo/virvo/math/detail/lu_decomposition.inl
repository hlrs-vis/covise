#include <sstream>
#include <stdexcept>


namespace MATH_NAMESPACE
{


//-----------------------------------------------------------------------
/** LU decomposition.
  From  Numerical Recipes, p. 46.<BR>
  Given a matrix a[n][n], this routine replaces the matrix by the
  LU decomposition of a rowwise permutation of itself. n and a are
  input. a is output. indx[n] is an output vector that records
  the row permutation effected by the partial pivoting; d is output
  +-1 depending on wether the number of eow interchanges was even
  or odd, respectively.
  @author Daniel Weiskopf
  @see LUBackSubstitution
*/
inline void lu_decomposition(int index[4], float& d, matrix< 4, 4, float >& m)
{

    const float TINY = 1.0e-20f;
    const int N = 4;                                // Special case for n=4, could be any positive number
    int i, imax, j, k;
    float big, dum, sum, temp;
    float vv[N];                                    // Stores the implicit scaling of each row

    d = 1.0f;

    // Loop over rows to get the implicit scaling information
    for(i = 0; i < N; i++)
    {
        big = 0.0f;
        for (j = 0; j < N; j++)
        {
            if ((temp = (float) fabs(m(i, j))) > big)
                big = temp;
        }
        if (big == 0.0)
        {
            std::stringstream str;
            str << "Singular matrix in routine LUdcmp " << m(i, 0) << " "
                << m(i, 1) << " " << m(i, 2) << " " << m(i, 3);
            std::runtime_error( str.str() );
        }
        vv[i] = 1.0f / big;                           // Save the scaling
    }

    // Loop over columns for Crout's method
    for (j = 0; j < N; j++)
    {
        for (i = 0; i < j; i++)
        {
            sum = m(i, j);
            for (k = 0; k < i; k++)
                sum -= m(i, k) * m(k, j);
            m(i, j) = sum;
        }

        // Finds the pivot point
        big = 0.0f;
        imax = 0;
        for (i = j; i < N; i++)
        {
            sum = m(i, j);
            for (k = 0; k < j; k++)
                sum -= m(i, k) * m(k, j);
            m(i, j) = sum;
            if ((dum = vv[i] * (float) fabs(sum)) >= big)
            {
                big = dum;
                imax = i;
            }
        }

        // Do we need to interchange rows?
        if (j != imax)
        {
            for (k = 0; k < N; k++)
            {
                dum = m(imax, k);
                m(imax, k) = m(j, k);
                m(j, k) = dum;
            }
            d = -d;
            vv[imax] = vv[j];                           // Also interchange the scale factor
        }
        index[j] = imax;
        if (m(j, j) == 0.0)
            m(j, j) = TINY;
        if (j != N)
        {
            dum = 1/(m(j, j));
            for (i = j+1; i < N; i++)
                m(i, j) *= dum;
        }
    }

}


//----------------------------------------------------------------------------
/** LU backsubstitution.
  @author Daniel Weiskopf
  @see LUDecomposition
*/
inline void lu_backsubstitution(int index[4], float b[4], matrix< 4, 4, float >& m)
{
    int   i, j;
    int   ii = -1;
    float sum;

    // Special case for n=4, could be any positive number
    const int n = 4;

    for (i = 0; i < n; i++)
    {
        const int ip    = index[i];
        sum   = b[ip];
        b[ip] = b[i];
        if (ii >= 0)
            for (j = ii; j <= i-1; j++)
                sum -= m(i, j) * b[j];
        else if (sum)
            ii = i;
        b[i] = sum;
    }

    for (i = n-1; i >= 0; i--)
    {
        sum = b[i];
        for (j = i+1; j < n; j++)
            sum -= m(i, j) * b[j];
        b[i] = sum/m(i, i);
    }
}


//----------------------------------------------------------------------------
/** Invert a matrix (according to Numerical Recipes, p. 48)
  @author Daniel Weiskopf
*/
inline matrix< 4, 4, float > lu_inverse(matrix< 4, 4, float > const& m)
{

    float    d;
    int      index[4];
    float    col[4];
    int      i;
    int      j;

    matrix< 4, 4, float > tmp = m;
    matrix< 4, 4, float > result;

    lu_decomposition(index, d, tmp);
    for ( j = 0; j < 4; j++)
    {
        for ( i = 0; i < 4; i++)
            col[i] = 0.0f;
        col[j] = 1.0f;
        lu_backsubstitution(index, col, tmp);
        for ( i = 0; i < 4; i++)
            result(i, j) = col[i];
    }

    return result;

}


} // MATH_NAMESPACE


