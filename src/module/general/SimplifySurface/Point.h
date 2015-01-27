/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SS_POINT_H_
#define _SS_POINT_H_

#define ILL_DEFINED 1.0e8

#include "util/coviseCompat.h"
#include <string.h>
#include <math.h>
//#include <algorithm>

using std::min_element;
using std::max_element;

extern bool choldc(double **a, int n, double p[]);
extern void cholsl(double **a, int n, double p[], const double b[], double x[]);

// scalar product of two vectors
extern double ScalarProd(int dimension, const float *data0, const float *data1);
extern void vect_prod(float *normal, const float *e0, const float *e1);

extern double ScalarProd(int dimension, const double *data0, const double *data1);
extern void vect_prod(double *normal, const double *e0, const double *e1);

extern bool Normalise(float *normal);
extern bool Normalise(float *normal, int dim);

extern bool ignoreData;

// given three points (data?) e1 is a normal
// vector along data0 - data1 and e2 is orthogonal
// to e1 and coplanar with the three input points.
// Both e0 and e1 are normalised. If the operation
// is possible true is returned, otherwise false.
static bool
Base2(int dimension, double e1[], double e2[],
      const double *data0, const double *data1, const double *data2)
{
    int i;
    for (i = 0; i < dimension; ++i)
    {
        e1[i] = data1[i] - data0[i];
    }
    double elen = ScalarProd(dimension, e1, e1);
    elen = sqrt(elen);
    if (elen == 0.0)
    {
        return false;
    }
    elen = 1.0 / elen;
    for (i = 0; i < dimension; ++i)
    {
        e1[i] *= elen;
    }

    for (i = 0; i < dimension; ++i)
    {
        e2[i] = data2[i] - data0[i];
    }
    double scal = ScalarProd(dimension, e1, e2);
    for (i = 0; i < dimension; ++i)
    {
        e2[i] = data2[i] - data0[i] - scal * e1[i];
    }
    elen = ScalarProd(dimension, e2, e2);
    elen = sqrt(elen);
    if (elen == 0.0)
    {
        return false;
    }
    elen = 1.0 / elen;
    for (i = 0; i < dimension; ++i)
    {
        e2[i] *= elen;
    }
    return true;
}

// template struct of a symmetric matrix
template <int dimension>
struct SymMatrix
{
    // default constructor: all elements are null
    SymMatrix()
    {
        // number of independent matrix components
        int num = (dimension * (dimension + 1)) / 2;
        int coord;
        for (coord = 0; coord < num; ++coord)
        {
            _matrix[coord] = 0.0;
        }
    }
    void print(std::string pre)
    {
        std::cout << pre << "SymMatrix";
        int i, j;
        for (i = 0; i < dimension; ++i)
        {
            for (j = 0; j < dimension; ++j)
            {
                cout << " " << matrix(i, j);
            }
        }
        cout << endl;
    }
    // matrix addition
    SymMatrix &operator+=(const SymMatrix &mat)
    {
        // number of independent matrix components
        int num = (dimension * (dimension + 1)) / 2;
        int coord;
        for (coord = 0; coord < num; ++coord)
        {
            _matrix[coord] += mat._matrix[coord];
        }
        return *this;
    }
    // scalar multiplication
    SymMatrix &operator*=(float factor)
    {
        // number of independent matrix components
        int num = (dimension * (dimension + 1)) / 2;
        int coord;
        for (coord = 0; coord < num; ++coord)
        {
            _matrix[coord] *= factor;
        }
        return *this;
    }
    // independent matrix components
    double _matrix[(dimension * (dimension + 1)) / 2];
    // find a term in terms of row and column labels
    double &matrix(int i, int j)
    {
        return (i >= j ? _matrix[j + (i * (i + 1)) / 2] : _matrix[i + (j * (j + 1)) / 2]);
    }
    // find a term in terms of row and column labels: const version
    const double &matrix(int i, int j) const
    {
        return (i >= j ? _matrix[j + (i * (i + 1)) / 2] : _matrix[i + (j * (j + 1)) / 2]);
    }
    // print quadric info
    void Print() const
    {
        int i, j;
        for (i = 0; i < dimension; ++i)
        {
            for (j = 0; j < dimension; ++j)
            {
                cerr << matrix(i, j) << ' ';
            }
            cerr << endl;
        }
    }
    // identity matrix
    void MakeIdent()
    {
        int i;
        for (i = 0; i < (dimension * (dimension + 1)) / 2; ++i)
        {
            _matrix[i] = 0.0;
        }
        for (i = 0; i < dimension; ++i)
        {
            matrix(i, i) = 1.0;
        }
    }
    // performs operation -= e * e^t (tensor product of a vector)
    void MinusQuad(const double *e)
    {
        int i, j;
        for (i = 0; i < dimension; ++i)
        {
            for (j = 0; j <= i; ++j)
            {
                matrix(i, j) -= e[i] * e[j];
            }
        }
    }

    // solve A*result = b, true is returned if a solution was found
    bool Solve(double result[dimension], const double b[dimension]) const
    {
        double a[dimension][dimension];
        double *awrap[dimension];
        double p[dimension];
        int coord;
        double *base = &(a[0][0]);
        for (coord = 0; coord < dimension; ++coord, base += dimension)
        {
            awrap[coord] = base;
        }
        int i, j;
        for (i = 0; i < dimension; ++i)
        {
            for (j = i; j < dimension; ++j)
            {
                a[i][j] = matrix(i, j);
            }
        }
        if (!choldc(awrap, dimension, p))
        {
            return false;
        }
        float p_max = (float)(*(max_element(p, p + dimension)));
        float p_min = (float)(*(min_element(p, p + dimension)));
        if (p_min < 0.0 || p_min * ILL_DEFINED < p_max)
        {
            return false;
        }
        cholsl(awrap, dimension, p, b, result);
        return true;
    }
};

class Q;

// abstract class for a Point: abstraction of geometric point + data
class Point
{
public:
    // generate a Quadric for the triangle determined by this, p1 and p2
    virtual Q *NewQ(const Point *p1, const Point *p2) const = 0;
    // creates a copy of this point
    virtual Point *Copy() const = 0;
    // destructor
    virtual ~Point()
    {
    }
    // get an array woth geometric + non-geometric data
    virtual const float *data() const
    {
        return NULL;
    }
    virtual int data_dim() const
    {
        return -1;
    }
    // add a point
    virtual Point &operator+=(const Point &rhs) = 0;
    // multiply by a float
    virtual Point &operator*=(float factor) = 0;
    void print(std::string pre)
    {
        std::cout << pre << "Point " << data()[0] << " " << data()[1] << " " << data()[2] << " " << data()[3] << std::endl;
    }

protected:
private:
};

// template of derived classes from Point
// dimension refers here only to the dimensionality of the
// non-geometric part defining a point
template <int dimension>
class PointAndData : public Point
{
public:
    // x, y, z define the geometric part, the non-geometric
    // part is defined by dimension values of array values
    PointAndData(float x, float y, float z, const float *values)
    {
        _data[0] = x;
        _data[1] = y;
        _data[2] = z;

        //TODO
        //Anyone for implementing a compile-time switch for dimension or creating a specialization PointAndData<0>?
        for (ssize_t i = 0; i < dimension; ++i)
            _data[3 + i] = values[i];
        //memcpy(_data+3,values,dimension*sizeof(float));
    }
    // destructor
    virtual ~PointAndData()
    {
    }
    // clone this Point
    virtual Point *Copy() const
    {
        return new PointAndData(_data[0], _data[1], _data[2], _data + 3);
    }
    // generate a Quadric for the triangle determined by this, p1 and p2
    virtual Q *NewQ(const Point *p1, const Point *p2) const;
    // get an array woth geometric + non-geometric data
    virtual const float *data() const
    {
        return _data;
    }
    virtual int data_dim() const
    {
        return dimension;
    }
    // add a point
    virtual Point &operator+=(const Point &rhs)
    {
        int coord;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            _data[coord] += rhs.data()[coord];
        }
        return *this;
    }
    // multiply by a float
    virtual Point &operator*=(float factor)
    {
        int coord;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            _data[coord] *= factor;
        }
        return *this;
    }

protected:
private:
    float _data[dimension + 3];
};

// abstract class of a quadric (sum of square of the distances
// of a generic point to a set of triangles --planes--)
class Q
{
public:
    // clone this quadric
    virtual Q *Copy() const = 0;
    // add a new contribution
    virtual Q &operator+=(const Q &) = 0;
    // factor
    virtual Q &operator*=(double factor) = 0;
    // find optimal point minimasing the quadric
    virtual Point *Solve(float &cost) const = 0;
    // use to ensure boundary is not modified
    virtual void ReduceData() = 0;
    // destructor
    virtual ~Q()
    {
    }
    // print quadric info
    virtual void Print() const = 0;
    // evaluate error
    virtual float Eval(const Point *point) const = 0;
    virtual void print(std::string pre) = 0;
};

// template struct derived from Q
// dimesion refers to the dimensionality of the
// non-geometric part of the points
template <int dimension>
class Qs : public Q
{
public:
    // quadric definition:
    // D^2(v) := v^t * _AMatrix * v + 2* _BVector^t * v + _Cnumber
    int _evalDimension;
    SymMatrix<dimension + 3> _AMatrix;
    SymMatrix<dimension + 3> _AMatrixWithData;
    double _BVector[dimension + 3];
    double _BVectorWithData[dimension + 3];
    double _Cnumber;
    // default contructor: _AMatrix is already initialised as null matrix
    Qs()
        : _AMatrix()
        , _AMatrixWithData()
    {
        _evalDimension = ignoreData ? 0 : dimension;
        _Cnumber = 0.0;
        int coord;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            _BVector[coord] = 0.0;
            _BVectorWithData[coord] = 0.0;
        }
    }
    // copy constructor
    Qs(const Qs &rhs)
        : _AMatrix()
        , _AMatrixWithData()
    {
        _evalDimension = ignoreData ? 0 : dimension;
        _AMatrix += rhs._AMatrix;
        _AMatrixWithData += rhs._AMatrixWithData;
        int coord;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            _BVector[coord] = rhs._BVector[coord];
            _BVectorWithData[coord] = rhs._BVectorWithData[coord];
        }
        _Cnumber = rhs._Cnumber;
    }
    // create a Q given a triangle
    Qs(const float *data0, const float *data1, const float *data2)
    {
        _evalDimension = ignoreData ? 0 : dimension;
        if (ignoreData)
        {
            double dummy;
            init(data0, data1, data2, _AMatrixWithData, _BVectorWithData, dummy, true);
            init(data0, data1, data2, _AMatrix, _BVector, _Cnumber, false);
        }
        else
        {
            init(data0, data1, data2, _AMatrix, _BVector, _Cnumber, true);
        }
    }
    // create a Q given a triangle
    Qs(const double *data0, const double *data1, const double *data2)
    {
        _evalDimension = ignoreData ? 0 : dimension;
        if (ignoreData)
        {
            double dummy;
            init(data0, data1, data2, _AMatrixWithData, _BVectorWithData, dummy, true);
            init(data0, data1, data2, _AMatrix, _BVector, _Cnumber, false);
        }
        else
        {
            init(data0, data1, data2, _AMatrix, _BVector, _Cnumber, true);
        }
    }

    bool init(const double *data0, const double *data1, const double *data2, SymMatrix<dimension + 3> &AMatrix, double BVector[dimension + 3], double &Cnumber, bool useData)
    {
        double e1[dimension + 3];
        double e2[dimension + 3];
        double ddata0[dimension + 3];
        double ddata1[dimension + 3];
        double ddata2[dimension + 3];
        int coord;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            if ((useData) || (coord < 3))
            {
                ddata0[coord] = data0[coord];
                ddata1[coord] = data1[coord];
                ddata2[coord] = data2[coord];
            }
            else
            {
                ddata0[coord] = 0.0f;
                ddata1[coord] = 0.0f;
                ddata2[coord] = 0.0f;
            }
        }
        if (!Base2(dimension + 3, (double *)e1, (double *)e2,
                   (const double *)ddata0, (const double *)ddata1, (const double *)ddata2))
        {
            for (coord = 0; coord < dimension + 3; ++coord)
            {
                BVector[coord] = 0.0;
            }
            Cnumber = 0.0;
            return false;
        }
        AMatrix.MakeIdent();
        AMatrix.MinusQuad(e1);
        AMatrix.MinusQuad(e2);
        double scl1 = ScalarProd(dimension + 3, ddata0, e1);
        double scl2 = ScalarProd(dimension + 3, ddata0, e2);
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            BVector[coord] = scl1 * e1[coord] + scl2 * e2[coord] - ddata0[coord];
        }
        Cnumber = -(scl1 * scl1 + scl2 * scl2);
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            Cnumber += ddata0[coord] * ddata0[coord];
        }

        // get 2xtriangle area and use as factor
        double side1[dimension + 3];
        double side2[dimension + 3];
        double length1_2 = 0.0;
        double length2_2 = 0.0;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            side1[coord] = ddata1[coord] - ddata0[coord];
            side2[coord] = ddata2[coord] - ddata0[coord];
            length1_2 += side1[coord] * side1[coord];
            length2_2 += side2[coord] * side2[coord];
        }
        float scal12 = (float)ScalarProd(dimension + 3, side1, side2);
        float factor = (float)(length1_2 * length1_2 - scal12 * scal12);
        if (factor < 0.0)
        {
            factor = 0.0;
        }
        else
        {
            factor = sqrt(factor);
        }
        AMatrix *= factor;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            BVector[coord] *= factor;
        }
        Cnumber *= factor;
        return true;
    }

    bool init(const float *data0, const float *data1, const float *data2, SymMatrix<dimension + 3> &AMatrix, double BVector[dimension + 3], double &Cnumber, bool useData)
    {
        double e1[dimension + 3];
        double e2[dimension + 3];
        double ddata0[dimension + 3];
        double ddata1[dimension + 3];
        double ddata2[dimension + 3];
        int coord;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            if ((useData) || (coord < 3))
            {
                ddata0[coord] = data0[coord];
                ddata1[coord] = data1[coord];
                ddata2[coord] = data2[coord];
            }
            else
            {
                ddata0[coord] = 0.0f;
                ddata1[coord] = 0.0f;
                ddata2[coord] = 0.0f;
            }
        }
        if (!Base2(dimension + 3, (double *)e1, (double *)e2,
                   (const double *)ddata0, (const double *)ddata1, (const double *)ddata2))
        {
            for (coord = 0; coord < dimension + 3; ++coord)
            {
                BVector[coord] = 0.0;
            }
            Cnumber = 0.0;
            return false;
        }
        AMatrix.MakeIdent();
        AMatrix.MinusQuad(e1);
        AMatrix.MinusQuad(e2);
        double scl1 = ScalarProd(dimension + 3, ddata0, e1);
        double scl2 = ScalarProd(dimension + 3, ddata0, e2);
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            BVector[coord] = scl1 * e1[coord] + scl2 * e2[coord] - ddata0[coord];
        }
        Cnumber = -(scl1 * scl1 + scl2 * scl2);
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            Cnumber += ddata0[coord] * ddata0[coord];
        }

        // get 2xtriangle area and use as factor
        double side1[dimension + 3];
        double side2[dimension + 3];
        double length1_2 = 0.0;
        double length2_2 = 0.0;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            side1[coord] = ddata1[coord] - ddata0[coord];
            side2[coord] = ddata2[coord] - ddata0[coord];
            length1_2 += side1[coord] * side1[coord];
            length2_2 += side2[coord] * side2[coord];
        }
        float scal12 = (float)ScalarProd(dimension + 3, side1, side2);
        float factor = (float)(length1_2 * length1_2 - scal12 * scal12);
        if (factor < 0.0)
        {
            factor = 0.0;
        }
        else
        {
            factor = sqrt(factor);
        }
        AMatrix *= factor;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            BVector[coord] *= factor;
        }
        Cnumber *= factor;
        return true;
    }

    virtual void print(std::string pre)
    {
        std::cout << pre << "Q" << std::endl;
        _AMatrix.print(pre + " ");
        std::cout << pre << " BVector " << _BVector[0] << " " << _BVector[1] << " " << _BVector[2] << " " << _BVector[3] << std::endl;
        ;
        std::cout << pre << " CNumber " << _Cnumber << std::endl;
    }

    // use to ensure boundary is not modified
    virtual void ReduceData()
    {
        int i;
        for (i = 3; i < dimension + 3; ++i)
        {
            _AMatrix.matrix(i, i) = 0.0;
            _AMatrixWithData.matrix(i, i) = 0.0;
        }
    }
    // factor
    virtual Q &operator*=(double factor)
    {
        _AMatrix *= (float)factor;
        _AMatrixWithData *= (float)factor;
        int i;
        for (i = 0; i < dimension + 3; ++i)
        {
            _BVector[i] *= factor;
            _BVectorWithData[i] *= factor;
        }
        _Cnumber *= factor;
        return *this;
    }
    // add a new quadric contribution
    Q &operator+=(const Q &rhs)
    {
        // rhs.Print();
        Qs *qm = (Qs *)(&rhs);
        _AMatrix += qm->_AMatrix;
        _AMatrixWithData += qm->_AMatrixWithData;
        int coord;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            _BVector[coord] += qm->_BVector[coord];
            _BVectorWithData[coord] += qm->_BVectorWithData[coord];
        }
        _Cnumber += qm->_Cnumber;
        return *this;
    }
    // clone this quadric
    Q *Copy() const
    {
        Qs *ret = new Qs;
        ret->_AMatrix += _AMatrix;
        ret->_AMatrixWithData += _AMatrixWithData;
        int i;
        for (i = 0; i < dimension + 3; ++i)
        {
            ret->_BVector[i] = _BVector[i];
            ret->_BVectorWithData[i] = _BVectorWithData[i];
        }
        ret->_Cnumber = _Cnumber;
        // ret->Print();
        return ret;
    }

    // print qaudric info
    virtual void Print() const
    {
        cerr << " _AMatrix +++++++++++++++++++++++" << endl;
        _AMatrix.Print();
        cerr << " _BVector +++++++++++++++++++++++" << endl;
        int i;
        for (i = 0; i < dimension + 3; ++i)
        {
            cerr << _BVector[i] << ' ';
        }
        cerr << endl;
        cerr << " _Cnumber +++++++++++++++++++++++" << endl;
        cerr << _Cnumber << endl;
    }
    // find optimal point minimising the quadric function
    Point *Solve(float &cost) const
    {
        double result[dimension + 3];
        if (!_AMatrix.Solve(result, _BVector))
        {
            return NULL;
        }
        int coord;
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            result[coord] *= -1.0;
        }
        float resultf[dimension + 3];
        for (coord = 0; coord < dimension + 3; ++coord)
        {
            resultf[coord] = float(result[coord]);
        }
        float *dataForNewPoint = resultf + 3;
        if (ignoreData)
        {
            double resultWithData[dimension + 3];
            _AMatrixWithData.Solve(resultWithData, _BVectorWithData);
            for (coord = 0; coord < dimension + 3; ++coord)
            {
                resultWithData[coord] *= -1.0;
            }
            float resultfWithData[dimension + 3];
            for (coord = 0; coord < dimension + 3; ++coord)
            {
                resultfWithData[coord] = float(resultWithData[coord]);
            }
            dataForNewPoint = resultfWithData + 3;
        }
        PointAndData<dimension> *ret = new PointAndData<dimension>(resultf[0], resultf[1], resultf[2], dataForNewPoint);
        double costd = _Cnumber;
        for (coord = 0; coord < _evalDimension + 3; ++coord)
        {
            costd += _BVector[coord] * result[coord];
        }
        cost = float(costd);
        return ret;
    }
    // evaluate error
    virtual float Eval(const Point *point) const
    {
        float ret = 0.0;
        const float *data = point->data();
        int i, j;
        for (i = 0; i < _evalDimension + 3; ++i)
        {
            ret += (float)(_AMatrix.matrix(i, i) * data[i] * data[i]);
        }
        for (i = 1; i < _evalDimension + 3; ++i)
        {
            for (j = 0; j < i; ++j)
            {
                float add = (float)(_AMatrix.matrix(i, j) * data[i] * data[j]);
                ret += add;
                ret += add;
            }
        }
        for (i = 0; i < _evalDimension + 3; ++i)
        {
            float add = (float)(_BVector[i] * data[i]);
            ret += add;
            ret += add;
        }
        ret += (float)_Cnumber;
        return ret;
    }
};

// create a quadric for the triangle defined by this, p1 and p2
template <int dimension>
Q *PointAndData<dimension>::NewQ(const Point *p1, const Point *p2) const
{
    return (new Qs<dimension>(_data, p1->data(), p2->data()));
}

class Factory
{
public:
    static Q *NewQs(int no_data, const float *pt0, const float *pt1, const float *pt2);
    static Q *NewQs(int no_data, const double *pt0, const double *pt1, const double *pt2);
    static Point *NewPointAndData(int no_data, float x, float y, float z, const float *data);
};
#endif
