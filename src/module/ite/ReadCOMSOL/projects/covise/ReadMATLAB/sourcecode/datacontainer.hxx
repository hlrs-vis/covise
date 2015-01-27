// ITE-Toolbox ReadMATLAB (C) Institute for Theory of Electrical Engineering
//
// Data container

class DataContainer
{
public:
    DataContainer(const unsigned long noPoints);
    ~DataContainer();
    bool SetType(const bool isVector);
    bool IsVector(void) const;
    bool IsSet(void) const;
    double* GetX(void);
    double* GetY(void);
    double* GetZ(void);
protected:
private:
    const unsigned long _noPoints;
    bool _isSet;
    bool _isVector;
    double* _x;
    double* _y;
    double* _z;
};
