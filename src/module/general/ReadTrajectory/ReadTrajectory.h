
#ifndef _READ_OBJ_SIMPLE_H
#define _READ_OBJ_SIMPLE_H

#include <api/coModule.h>
#include <vector>
using namespace covise;

//The base class for all module programming is coModule. 
//An application is created by deriving a class of coModule. The constructor of the derived class creates the module layout with input ports, output ports and parameters. 
//Virtual functions are overloaded to implement the module's reactions on events.
class ReadTrajectory : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void quit();

    bool openFile();
    void readFile();

    //  member data
    const char *filename; // obj file name
    FILE *fp;

    coOutputPort *linePort; 
    coFileBrowserParam *objFileParam;

public:
    ReadTrajectory(int argc, char *argv[]);
    virtual ~ReadTrajectory();
};

class Point {
public:
    Point(float x_c, float y_c, float z_c, float time_c, int corner_list_index, int line_list_index);
    
    float getX() const;
    float getY() const;
    float getZ() const;
    float getTime() const;
    int   getCorner() const;
    int   getLine() const;


private:
    float x, y, z;
    float time;
    int   cornerIndex;
    int   lineIndex;
};

bool withinRadius(const Point& p1, const Point& p2, float radius);

vector<pair<size_t, size_t>> findPointsWithinRadius(const vector<Point>& points, float radius, float startTime, float endTime);
#endif
