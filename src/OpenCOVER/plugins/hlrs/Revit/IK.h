

#ifndef IK_H
#define IK_H

#include <vector>
#include <queue>
#include <osg/Vec3>
#include <osg/Matrix>
#include <osg/Quat>


// MatrixF Declaration	
#define VNAME		F
#define VTYPE		float

class MatrixF {
public:
	VTYPE* data;
	int rows, cols, len;

	// Constructors/Destructors		
	MatrixF();
	~MatrixF();
	MatrixF(const int r, const int c);

	// Member Functions
	VTYPE GetVal(int c, int r);
	VTYPE& operator () (const int c, const int r);
	MatrixF& operator= (const unsigned char c);
	MatrixF& operator= (const int c);
	MatrixF& operator= (const double c);
	MatrixF& operator= (const MatrixF& op);

	MatrixF& operator+= (const unsigned char c);
	MatrixF& operator+= (const int c);
	MatrixF& operator+= (const double c);
	MatrixF& operator+= (const MatrixF& op);

	MatrixF& operator-= (const unsigned char c);
	MatrixF& operator-= (const int c);
	MatrixF& operator-= (const double c);
	MatrixF& operator-= (const MatrixF& op);

	MatrixF& operator*= (const unsigned char c);
	MatrixF& operator*= (const int c);
	MatrixF& operator*= (const double c);
	MatrixF& operator*= (const MatrixF& op);

	MatrixF& operator/= (const unsigned char c);
	MatrixF& operator/= (const int c);
	MatrixF& operator/= (const double c);
	MatrixF& operator/= (const MatrixF& op);

	MatrixF& Multiply4x4(const MatrixF& op);
	MatrixF& Multiply(const MatrixF& op);
	MatrixF& Resize(const int x, const int y);
	MatrixF& ResizeSafe(const int x, const int y);
	MatrixF& InsertRow(const int r);
	MatrixF& InsertCol(const int c);
	MatrixF& Transpose(void);
	MatrixF& Identity(const int order);
	MatrixF& RotateX(const double ang);
	MatrixF& RotateY(const double ang);
	MatrixF& RotateZ(const double ang);
	MatrixF& Ortho(double sx, double sy, double n, double f);
	MatrixF& Translate(double tx, double ty, double tz);
	MatrixF& Basis(const osg::Vec3& c1, const osg::Vec3& c2, const osg::Vec3& c3);
	MatrixF& GaussJordan(MatrixF& b);
	MatrixF& ConjugateGradient(MatrixF& b);
	MatrixF& Submatrix(MatrixF& b, int mx, int my);
	MatrixF& MatrixVector5(MatrixF& x, int mrows, MatrixF& b);
	MatrixF& ConjugateGradient5(MatrixF& b, int mrows);
	double Dot(MatrixF& b);

	void Print(char* fname);

	int GetX();
	int GetY();
	int GetRows(void);
	int GetCols(void);
	int GetLength(void);
	VTYPE* GetData(void);
	void GetRowVec(int r, osg::Vec3& v);

	unsigned char* GetDataC(void) const { return NULL; }
	int* GetDataI(void)	const { return NULL; }
	float* GetDataF(void) const { return data; }

	float GetF(const int r, const int c);
};

#undef VNAME
#undef VTYPE



struct Joint {
	char		name[64];
	char		lev;
	uint32_t		clr;
	float		length;		// bone length, v''		
	osg::Vec3	pos;		// bone position, T
	osg::Vec3	dof;
	osg::Quat	orient;		// orientation angles, Ri (converted to ideal format, Mv = T Ri' v'', see Meredith & Maddock paper)
	osg::Matrix	Mworld;		// world transform (computed on-the-fly)
	int			parent;
	int			child;
	int			next;

	osg::Vec3	min_limit;
	osg::Vec3	max_limit;

	//-- debugging
	osg::Quat	origOri;		// BVH angles (debug purposes, will be removed in future)
	osg::Vec3	bonevec;	// BVH bone vec
};

class Camera3D;

class Joints {
public:
	Joints();


	void Clear();

	int AddJoint(char* name, float length, osg::Quat ori, int cx, int cy, int cz);
	void SetLimits(int j, osg::Vec3 lmin, osg::Vec3 lmax);
	void MoveJoint(int j, int axis_id, float da);
	void EvaluateJoints(std::vector<Joint>& joints, osg::Matrix& world);			// Evaluate cycle to joints
	void EvaluateJointsRecurse(std::vector<Joint>& joints, int curr_jnt, osg::Matrix tform);	// Update joint transforms				
	void StartIK();
	void InverseKinematics(osg::Vec3 goal, int maxiter = 100);
	void ComputeJacobian();
	void ApplyJacobianTranspose(float amt);
	void LimitQuaternion(osg::Quat& o, int axis, float limitang);

	Joint* FindJoint(std::string name);
	Joint* getJoint(int i) { return &m_Joints[i]; }
	std::vector<Joint>& getJoints() { return m_Joints; }
	int getNumJoints() { return (int)m_Joints.size(); }
	int getLastChild(int p);

private:

	std::vector<Joint>		m_Joints;				// joint state

	osg::Vec3				m_Goal;
	osg::Vec3				m_Effector;
	MatrixF					m_Jacobian;
};

#endif