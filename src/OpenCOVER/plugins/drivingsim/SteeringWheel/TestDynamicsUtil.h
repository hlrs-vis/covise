#ifndef __TestDynamicsUtil_H
#define __TestDynamicsUtil_H

class Vec2D
{
public:
	Vec2D (double x, double y);
	void set(double x, double y);
	double getX();
	double getY();
private:
	double x;
	double y;
};

class Vec3D
{
public:
	Vec3D (double x, double y, double z);
	void set(double x, double y, double z);
	double getX();
	double getY();
	double getZ();
private:
	double x;
	double y;
	double z;
	
};

#endif