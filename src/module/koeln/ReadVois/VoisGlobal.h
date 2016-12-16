#ifndef _VOIS_GLOBAL_H
#define _VOIS_GLOBAL_H

#include <vector>
#include <math.h>

#define  no_voi_property        -1
#define  target_property         1
#define  head_property           2
#define  risk_property           3
#define  interest_property       4

struct xy  {
    double    x;
    double    y;
};

struct xyz {

    //member variables
    double    x;
    double    y;
    double    z;
    int index;

    //constructors
    xyz(){
        x = 0.0;
        y = 0.0;
        z = 0.0;
        index = -1;
    }

    xyz(double vx, double vy, double vz){
        x = vx;
        y = vy;
        z = vz;
        index = -1;
    }

    xyz(double vx, double vy, double vz, int vindex){
        x = vx;
        y = vy;
        z = vz;
        index = vindex;
    }

    //friends
    friend std::ostream& operator<< (std::ostream &out, const xyz &v);

    //operators
    xyz operator+(const xyz& v) const {
        xyz retVal;
        retVal.x = x + v.x;
        retVal.y = y + v.y;
        retVal.z = z + v.z;
        return retVal;
    }

    xyz operator-(const xyz& v) const {
        xyz retVal;
        retVal.x = x - v.x;
        retVal.y = y - v.y;
        retVal.z = z - v.z;
        return retVal;
    }

    //multiplication by factor
    xyz operator*(double factor) const {
        xyz retVal;
        retVal.x = x * factor;
        retVal.y = y * factor;
        retVal.z = z * factor;
        retVal.index = index;
        return retVal;
    }

    //divide by factor
    xyz operator/(double factor) const {
        xyz retVal;
        if(factor < 0.00000001){
            retVal.x = x;
            retVal.y = y;
            retVal.z = z;
            retVal.index = index;
            return retVal;
        }

        retVal.x = x / factor;
        retVal.y = y / factor;
        retVal.z = z / factor;
        retVal.index = index;

        return retVal;
    }

    void homogen(){
        double epsilon = z;
        if(epsilon < 0.0) epsilon = -epsilon;

        if(epsilon < 0.000000001){
            z = 0.0;
        } else {
            x = x / z;
            y = y / z;
            z = 1.0f;
        }
    }

    void normalize(){
        double length = sqrt(x*x + y*y + z*z);
        x /= length;
        y /= length;
        z /= length;
    }

    double squaredLength(){
        return x*x + y*y + z*z;
    }

    double length(){
        return sqrt(squaredLength());
    }

};

//dot product
double dot(xyz v1, xyz v2){
    double retVal = (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z);
    return retVal;
}

//cross product
xyz cross(xyz v1, xyz v2){
    xyz retVal;
    retVal.x = v1.y * v2.z - v1.z * v2.y;
    retVal.y = v1.z * v2.x - v1.x * v2.z;
    retVal.z = v1.x * v2.y - v1.y * v2.x;
    return retVal;
}

std::ostream& operator<< (std::ostream &out, const xy &v)
{
    //out << std::setprecision(9) << std::fixed;
    out << "(" << v.x << "," << v.y << ")";
    return out;
}

//stream xyzs
std::ostream& operator<< (std::ostream &out, const xyz &v)
{
    //out << std::setprecision(9) << std::fixed;
    out << "(" << v.x << "," << v.y << "," << v.z << ")  index: " << v.index;
    return out;
}

struct contour_t  {
    int    no_slice;
    std::vector<xyz> points;      // for each point we do not pass the coordinates
                                  // but its pixel_index_x; pixel_index_Y;  slice_index

    std::vector<xyz>::iterator getNextPoint(std::vector<xyz>::iterator iter) {
        int distance = std::distance(points.begin(),iter);
        if(distance < 0 || distance > points.size()){
            std::cerr << "getNextPoint distance: " << distance << "  size: " << points.size() << std::endl;
            return points.end();
        }

        iter++;
        if(iter == points.end()) return points.begin();
        return iter;
    }
};

//stream contours
std::ostream& operator<< (std::ostream &out, const contour_t &c){
    out << "contour " << c.no_slice << "  #points:" << c.points.size() << std::endl;
    int i = 0;
    for(std::vector<xyz>::const_iterator iter = c.points.begin(); iter != c.points.end(); iter++){
        xyz v = *iter;
        out << "  point " << i << ": (" << v.x << "," << v.y << "," << v.z << ")  index: " << v.index << std::endl;
        i++;
    }
    return out;
}

struct voi_t {
    std::string  voi_name;
    int          voi_property;
    int          voi_first_slice;
    int          voi_last_slice;
    int          voi_color;
    int          voi_index;

    //for each slice on vector of xyz-coordinates
    std::vector<contour_t> contours;
    /*
    int       number_points;
    int       is_included_by;           // this value allways -1, when the volume is included(is_voi_included=true)
    bool      is_voi_included;          // if it is true, the voi is include in another one
    float     Weight_Factor;
    float     Volume;                   // volume of the vois in ml

    // extreme values
    double    max_x;
    double    min_x;
    double    max_y;
    double    min_y;
    double    max_z;
    double    min_z;

    // global variables
    QVector<int>     index_vector;
    QVector<float>   dose_vector;
    QVector<xyze>    voi_points;         // contain the coordinates for all vois in stereotactic coordinates
    QVector<slice_t> No_slice_points ;   // gives the index of all points in each slice for each voi
    */
};

std::vector<voi_t>::iterator getVoiWithIndex(std::vector<voi_t>& voiVector, int index){
    for(std::vector<voi_t>::iterator iter = voiVector.begin(); iter != voiVector.end(); ++iter){
        if(iter->voi_index == index) return iter;
    }
    return voiVector.end();
}

bool voiIsValid(int property){
    if(property == no_voi_property){
        return false;
    }
    return true;
}

struct edge_t {
    int vertex1;
    int vertex2;
};

struct triangle_t {
    int vertex1;
    int vertex2;
    int vertex3;
};

//stream triangles
std::ostream& operator<< (std::ostream &out, const triangle_t &t)
{
    out << "(" << t.vertex1 << "," << t.vertex2 << "," << t.vertex3 << ")";
    return out;
}


#endif //_VOIS_GLOBAL_H

