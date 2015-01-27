/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LocalCS_Data.h"
#include <algorithm>

#ifdef _STANDARD_C_PLUS_PLUS
#include <algo.h> // HACK to get "copy"
#endif
#include <do/coDoData.h>
LocalCS_Data::LocalCS_Data(const vector<float> &ref_system)
{
    assert(ref_system.size() == 9 || ref_system.size() == 4 || ref_system.size() == 0);
    switch (ref_system.size())
    {
    case 9:
        copy(ref_system.begin(), ref_system.end(), _ref_system);
        break;
    case 4:
    {
        int i;
        for (i = 0; i < 9; ++i)
        {
            _ref_system[i] = 0;
        }
        _ref_system[0] = ref_system[0];
        _ref_system[1] = ref_system[1];
        _ref_system[3] = ref_system[2];
        _ref_system[4] = ref_system[3];
        _ref_system[8] = 1.0;
    }
    break;
    case 0:
    {
        int i;
        for (i = 0; i < 9; ++i)
        {
            _ref_system[i] = 0;
        }
        _ref_system[0] = 1.0;
        _ref_system[4] = 1.0;
        _ref_system[8] = 1.0;
    }
    break;
    }
}

LocalCS_Data::~LocalCS_Data()
{
}

Data *
LocalCS_Data::Copy() const
{
    vector<float> ref;
    int i;
    for (i = 0; i < 9; ++i)
    {
        ref.push_back(_ref_system[i]);
    }
    LocalCS_Data *ret = new LocalCS_Data(ref);
    ret->_position = _position;
    ret->_node = _node;
    return ret;
}

static void
find_invariant_vector(const float rot[9],
                      float &x,
                      float &y,
                      float &z)
{
    float b11 = rot[0] - 1.0;
    float b12 = rot[1];
    float b13 = rot[2];
    float b21 = rot[3];
    float b22 = rot[4] - 1.0;
    float b23 = rot[5];
    float b31 = rot[6];
    float b32 = rot[7];
    float b33 = rot[8] - 1.0;

    float minors[9] = {
        b11 * b22 - b12 * b21,
        b11 * b23 - b13 * b21,
        b12 * b23 - b13 * b22,
        b11 * b32 - b12 * b31,
        b11 * b33 - b13 * b31,
        b12 * b33 - b13 * b32,
        b21 * b32 - b22 * b31,
        b21 * b33 - b23 * b31,
        b22 * b33 - b23 * b32
    };

    float *where = ::std::max_element(minors, minors + 9);

    float det = *where;
    if (det == 0.0)
    {
        det = 1.0;
    }
    switch (where - minors)
    {
    case 0:

        z = 1.0;

        x = (-b13 * b22 + b12 * b23) / det;
        y = (-b11 * b23 + b13 * b21) / det;

        break;

    case 1:

        y = 1.0;

        x = (-b12 * b23 + b13 * b22) / det;
        z = (-b11 * b22 + b12 * b21) / det;

        break;

    case 2:

        x = 1.0;

        y = (-b11 * b23 + b13 * b21) / det;
        z = (-b12 * b21 + b11 * b22) / det;

        break;

    case 3:

        z = 1.0;

        x = (-b13 * b32 + b12 * b33) / det;
        y = (-b11 * b33 + b13 * b31) / det;

        break;

    case 4:

        y = 1.0;

        x = (-b12 * b33 + b13 * b32) / det;
        z = (-b11 * b32 + b12 * b31) / det;

        break;

    case 5:

        x = 1.0;

        y = (-b11 * b33 + b13 * b31) / det;
        z = (-b12 * b31 + b11 * b32) / det;

        break;

    case 6:

        z = 1.0;

        x = (-b23 * b32 + b22 * b33) / det;
        y = (-b21 * b33 + b23 * b31) / det;

        break;

    case 7:

        y = 1.0;

        x = (-b22 * b33 + b23 * b32) / det;
        z = (-b21 * b32 + b22 * b31) / det;

        break;

    case 8:

        x = 1.0;

        y = (-b21 * b33 + b23 * b31) / det;
        z = (-b22 * b31 + b21 * b32) / det;

        break;

    default:
        break;
    }

    float vecnorm = sqrt(x * x + y * y + z * z);

    x /= vecnorm;
    y /= vecnorm;
    z /= vecnorm;
}

float
LocalCS_Data::quaternion::abs2(const quaternion &q1, const quaternion &q2)
{
    return ((q1._a - q2._a) * (q1._a - q2._a) + (q1._b - q2._b) * (q1._b - q2._b) + (q1._c - q2._c) * (q1._c - q2._c) + (q1._d - q2._d) * (q1._d - q2._d));
}

static void
find_vector_for_BOD(float x, float y, float z, float u, float v, float w,
                    float &r, float &s, float &t)
{
    r = y * w - z * v;
    s = z * u - x * w;
    t = x * v - y * u;
}

static void
find_orthogonal_vector(float x, float y, float z,
                       float &u, float &v, float &w)
{
    float absvals[3];
    absvals[0] = fabs(x);
    absvals[1] = fabs(y);
    absvals[2] = fabs(z);
    float *where = ::std::min_element(absvals, absvals + 3);
    switch (where - absvals)
    {
    case 0:
        u = 0.0;
        v = -z;
        w = y;
        break;
    case 1:
        u = -z;
        v = 0.0;
        w = x;
        break;
    case 2:
    default:
        u = -y;
        v = x;
        w = 0.0;
        break;
    }
    float norm = sqrt(u * u + v * v + w * w);
    u /= norm;
    v /= norm;
    w /= norm;
}

LocalCS_Data::quaternion::quaternion(const float rot[9], const quaternion *pq)
{
    if (rot[0] == 1.0 && rot[4] == 1.0 && rot[8] == 1.0)
    {
        _a = 1.0;
        _b = _c = _d = 0.0;
    }
    else
    {
        float cos_theta = (rot[0] + rot[4] + rot[8] - 1.0) * 0.5;
        float stuff = (cos_theta + 1.0) * 0.5;
        float cos_theta_sur_2 = sqrt(stuff);
        float sin_theta_sur_2 = sqrt(1 - stuff);

        float x;
        float y;
        float z;

        find_invariant_vector(rot, x, y, z);

        float u;
        float v;
        float w;

        find_orthogonal_vector(x, y, z, u, v, w);

        float r;
        float s;
        float t;

        find_vector_for_BOD(x, y, z, u, v, w, r, s, t);

        float ru = rot[0] * u + rot[1] * v + rot[2] * w;
        float rv = rot[3] * u + rot[4] * v + rot[5] * w;
        float rw = rot[6] * u + rot[7] * v + rot[8] * w;

        float angle_sign_determinator = r * ru + s * rv + t * rw;
        if (angle_sign_determinator > 0.0)
        {
            _a = cos_theta_sur_2;
            _b = x * sin_theta_sur_2;
            _c = y * sin_theta_sur_2;
            _d = z * sin_theta_sur_2;
        }
        else if (angle_sign_determinator < 0.0)
        {
            _a = cos_theta_sur_2;
            _b = -x * sin_theta_sur_2;
            _c = -y * sin_theta_sur_2;
            _d = -z * sin_theta_sur_2;
        }
        else if (u * ru + v * rv + w * rw < 0.0)
        {
            _a = 0.0;
            _b = x;
            _c = y;
            _d = z;
        }
        else
        {
            _a = 1.0;
            _b = _c = _d = 0.0;
        }
    }

    if (pq && abs2(*pq, *this) > abs2(*this, *pq))
    {
        _a *= -1.0;
        _b *= -1.0;
        _c *= -1.0;
        _d *= -1.0;
    }
}

LocalCS_Data::quaternion::quaternion(const quaternion &rhs)
    : _a(rhs._a)
    , _b(rhs._b)
    , _c(rhs._c)
    , _d(rhs._d)
{
}

LocalCS_Data::quaternion::~quaternion()
{
}

LocalCS_Data::quaternion &
    LocalCS_Data::quaternion::
    operator+=(const quaternion &rhs)
{
    _a += rhs._a;
    _b += rhs._b;
    _c += rhs._c;
    _d += rhs._d;
    return *this;
}

void
LocalCS_Data::quaternion::GetMatrix(vector<float> &mat) const
{
    float vecnorm = sqrt(_a * _a + _b * _b + _c * _c + _d * _d);
    if (vecnorm <= 0.0)
    {
        mat.push_back(1.0);
        mat.push_back(0.0);
        mat.push_back(0.0);
        mat.push_back(0.0);
        mat.push_back(1.0);
        mat.push_back(0.0);
        mat.push_back(0.0);
        mat.push_back(0.0);
        mat.push_back(1.0);
        return;
    }
    float a(_a), b(_b), c(_c), d(_d);
    vecnorm = 1.0 / vecnorm;
    a *= vecnorm;
    b *= vecnorm;
    c *= vecnorm;
    d *= vecnorm;
    mat.push_back(a * a + b * b - c * c - d * d);
    mat.push_back(-2 * a * d + 2 * b * c);
    mat.push_back(2 * a * c + 2 * b * d);
    mat.push_back(2 * a * d + 2 * b * c);
    mat.push_back(a * a - b * b + c * c - d * d);
    mat.push_back(-2 * a * b + 2 * a * d);
    mat.push_back(-2 * a * c + 2 * b * d);
    mat.push_back(2 * a * b + 2 * c * d);
    mat.push_back(a * a - b * b - c * c + d * d);
};

Data *
LocalCS_Data::Average(const vector<Data *> &other_data) const
{
    quaternion quat(_ref_system, NULL);
    quaternion accum(quat);
    int i;
    for (i = 1; i < other_data.size(); ++i)
    {
        accum += quaternion(((LocalCS_Data *)other_data[i])->_ref_system, &quat);
    }
    vector<float> mat;
    accum.GetMatrix(mat);
    return new LocalCS_Data(mat);
}

coDistributedObject *
LocalCS_Data::GetNoDummy(const char *name,
                         const vector<const Data *> &datalist) const
{
    coDoMat3 *ret = new coDoMat3(name, datalist.size());
    float *refs;
    ret->getAddress(&refs);
    int i;
    for (i = 0; i < datalist.size(); ++i)
    {
        const LocalCS_Data *ThisRef = (const LocalCS_Data *)datalist[i];
        std::copy(ThisRef->_ref_system, ThisRef->_ref_system + 9, refs);
        refs += 9;
    }
    return ret;
}

float
    LocalCS_Data::
    operator[](int i) const
{
    return _ref_system[i];
}
