

#include "IK.h"
#include <osg/Vec3>
#include <osg/Matrix>

#include <stack>
#include <vector>

#define JUNDEF		-1

Joints::Joints ()
{
}

// To Do:
// - angle constraints
// - ball joints
// - 6-DOF end effector

float circleDelta ( float a, float b )
{
	a = (float) fmod (a, 360 ); if ( a < 0 ) a += 360;

	float d = fabs( b - a);
	float r = d > 180 ? 360 - d : d;
	int sign = (b - a >= 0 && b - a <= 180) || (b - a <=-180 && b - a>= -360) ? 1 : -1; 
	return r * sign;
}


int Joints::getLastChild ( int p )
{
	int child = m_Joints[p].child;
	if ( child == JUNDEF ) return JUNDEF;
	while ( m_Joints[child].next != JUNDEF ) {
		child = m_Joints[child].next;
	}
	return child;
}
void Joints::Clear ()
{
	m_Joints.clear ();
}

void Joints::SetLimits ( int j, osg::Vec3 lmin, osg::Vec3 lmax )
{
	m_Joints[j].min_limit = lmin;
	m_Joints[j].max_limit = lmax;
}

int Joints::AddJoint ( char* name, float length, osg::Quat ori, int cx, int cy, int cz )
{
	Joint jnt;	
	int parent = m_Joints.size()-1;		// use last joint added as parent

	strncpy ( jnt.name, name, 64 );
  jnt.name[63] = '\0';

	jnt.parent = parent;	
	jnt.pos = osg::Vec3(0,0,0);
	jnt.length = length;
	jnt.origOri = ori;
	jnt.dof = osg::Vec3(cx,cy,cz);
	jnt.orient = ori;	
	jnt.min_limit.set(-180.f,-180.f,-180.f);
	jnt.max_limit.set( 180.f, 180.f, 180.f);
	jnt.clr = 0;	
	jnt.child = JUNDEF;
	jnt.next = JUNDEF;
	
	int curr = (int) m_Joints.size();

	if ( parent == JUNDEF ) {	
		// this is root
		jnt.lev = 0;		
	} else {
		Joint* pjnt = getJoint ( parent );
		jnt.lev = pjnt->lev + 1;
		int plastchild = getLastChild ( parent );	// last child of parent
		if ( plastchild != JUNDEF ) {
			getJoint(plastchild)->next = curr;			
		} else {
			pjnt->child = curr;			
		}		
	}

	m_Joints.push_back ( jnt );					// add joint

	return (int) m_Joints.size()-1;
}

void normalize(osg::Quat& q)
{

	double len2 = q.length2();
	if (len2 != 1.0)
	{
		q /= sqrt(len2);
	}
}

#define DEGtoRAD		(3.1415926535897932384626433832795028841971693993751/180.0)
#define RADtoDEG		(180.0/3.1415926535897932384626433832795028841971693993751)
inline double clamp(double value, double low, double high)
{
	return std::min(std::max(value, low), high);
}
inline bool fequal(double a, double b, double eps)
{
	return (fabs(a - b) < eps);
}
inline void toEuler(const osg::Quat& q, osg::Vec3& euler)
{
	const double sqw = q.w() * q.w();
	const double sqx = q.x() * q.x();
	const double sqy = q.y() * q.y();
	const double sqz = q.z() * q.z();
	const double test = 2.0 * (q.y() * q.w() - q.x() * q.z());

	if (fequal(test, 1.0, 0.000001))
	{
		// heading = rotation about z-axis
		euler.z() = (double)(-2.0 * atan2(q.x(), q.w()));
		// bank = rotation about x-axis
		euler.x() = 0;
		// attitude = rotation about y-axis
		euler.y() = M_PI_2;
	}
	else if (fequal(test, -1.0, 0.000001))
	{
		// heading = rotation about z-axis
		euler.z() = double(2.0 * atan2(q.x(), q.w()));
		// bank = rotation about x-axis
		euler.x() = 0;
		// attitude = rotation about y-axis
		euler.y() = -M_PI_2;
	}
	else
	{
		// heading = rotation about z-axis
		euler.z() = (double)atan2(2.0 * (q.x() * q.y() + q.z() * q.w()), (sqx - sqy - sqz + sqw));
		// bank = rotation about x-axis
		euler.x() = (double)atan2(2.0 * (q.y() * q.z() + q.x() * q.w()), (-sqx - sqy + sqz + sqw));
		// attitude = rotation about y-axis
		euler.y() = (double)asin(clamp(test, -1.0, 1.0));
	}

	euler.x() *= RADtoDEG;
	euler.y() *= RADtoDEG;
	euler.z() *= RADtoDEG;
}

// sets new Quaternion based on euler angles
inline void set(osg::Quat &q ,osg::Vec3 &v)
{
	double angle;

	angle = v.x() * 0.5;
	const double sr = sin(angle);
	const double cr = cos(angle);

	angle = v.y() * 0.5;
	const double sp = sin(angle);
	const double cp = cos(angle);

	angle = v.z() * 0.5;
	const double sy = sin(angle);
	const double cy = cos(angle);

	const double cpcy = cp * cy;
	const double spcy = sp * cy;
	const double cpsy = cp * sy;
	const double spsy = sp * sy;

	q.x() = (double)(sr * cpcy - cr * spsy);
	q.y() = (double)(cr * spcy + sr * cpsy);
	q.z() = (double)(cr * cpsy - sr * spcy);
	q.w() = (double)(cr * cpcy + sr * spsy);

	normalize(q);
}

void PreTranslate(osg::Matrix &m,const osg::Vec3& t)
{
	m(3,0) += (double)m(0,0) * t.x() + m(1,0) * t.y() + m(2,0) * t.z();
	m(3,1) += (double)m(0,1) * t.x() + m(1,1) * t.y() + m(2,1) * t.z();
	m(3,2) += (double)m(0,2) * t.x() + m(1,2) * t.y() + m(2,2) * t.z();
}


// Computes: M' = T*M
void PostTranslate(osg::Matrix &m,const osg::Vec3& t)
{
	m(3,0) += (double)t.x();
	m(3,1) += (double)t.y();
    m(3,2) += (double)t.z();
}

void Joints::MoveJoint ( int j, int axis_id, float da )
{
	if ( j >= m_Joints.size() ) return;

	osg::Vec3 axis;
	switch ( axis_id ) {
	case 0: axis.set(1,0,0); break;
	case 1: axis.set(0,1,0); break;
	case 2: axis.set(0,0,1); break;
	};
	bool allow = (axis * ( m_Joints[j].dof ) > 0 );
	if ( !allow ) return;

	osg::Quat delta;
	delta.makeRotate ( da, axis );
	normalize(delta);
	
	m_Joints[j].orient = m_Joints[j].orient * delta;		// local rotation
	normalize(m_Joints[j].orient);

	m_Joints[j].origOri = m_Joints[j].orient;
	//toEuler(m_Joints[j].orient,m_Joints[j].angs );


}

Joint* Joints::FindJoint ( std::string name )
{
	for (int n=0; n < getNumJoints(); n++ ) {
		if ( name.compare(m_Joints[n].name) == 0 )
			return &m_Joints[n];
	}
	return 0x0;
}

void Joints::EvaluateJoints ( std::vector<Joint>& joints, osg::Matrix& world )
{
	EvaluateJointsRecurse ( joints, 0, world );

	// Compute end effector
	int n = joints.size()-1;	
	osg::Matrix  local = m_Joints[n].Mworld;	
	osg::Vec3 b (0.f, m_Joints[n].length, 0.f); 
	local.postMult ( m_Joints[n].pos * -1.0f);					
	b = b * local; 
	m_Effector = m_Joints[n].pos + b;
}

// recursive funcs
void Joints::EvaluateJointsRecurse ( std::vector<Joint>& joints, int curr_jnt, osg::Matrix world )
{
	// Evaluation of joint chain
	//
	// local orientation
	osg::Matrix orient;
	osg::Vec3 a; 
	// joints[curr_jnt].orient.toEuler ( a );			// cast to Euler ZYX angles first
	// orient.RotateZYX ( a );		
	orient.makeRotate(joints[curr_jnt].orient);     	// Ri' = orientation angles (animated)		
	
	// set world transform
	if ( curr_jnt > 0 ) 
		world *= orient;								// Mw = M(w-1) Ri' v''
	
	joints[curr_jnt].Mworld = world;					
	joints[curr_jnt].pos = world.getTrans();			// Tworld
	
														// translate children to end of bone
	PreTranslate(world, osg::Vec3(0.f, joints[curr_jnt].length, 0.f) );		// v'' = bone length

	// recurse	
	int child_jnt = joints[curr_jnt].child;
	while ( child_jnt != JUNDEF ) {
		EvaluateJointsRecurse ( joints, child_jnt, world );
		child_jnt = joints[child_jnt].next;
	}
}

void Joints::StartIK ()
{
	// Count degrees-of-freedom (DOFs)
	int M = 0;
	for (int n=0; n < m_Joints.size(); n++ ) {
		M += m_Joints[n].dof.x();
		M += m_Joints[n].dof.y();
		M += m_Joints[n].dof.z();
	}

	// Construct Jacobian
	m_Jacobian.Resize ( M, 3 );
}

void Joints::InverseKinematics ( osg::Vec3 goal, int maxiter )
{	
	osg::Matrix world;
	float dE;
	float amt;
	int iter = 0;

	m_Goal = goal;
	dE = (m_Goal - m_Effector).length();

	while ( dE > 0.1f && iter++ < maxiter  ) {

		// check convergence
		dE = (m_Goal - m_Effector).length();

		amt = pow(dE * 0.2f, 1.5);		
		if ( amt > 0.5f ) amt = 0.5f;

		// compute jacobian
		ComputeJacobian ();

		// apply jacobian transpose
		ApplyJacobianTranspose ( amt );

		// re-evaluate joints
		world.makeRotate(m_Joints[0].orient);
		PostTranslate(world, m_Joints[0].pos );
		EvaluateJoints( m_Joints, world );
	}
}

void Joints::LimitQuaternion ( osg::Quat& o, int limitaxis_id, float limitang )
{
	osg::Vec3 angs;	
	osg::Vec3 a1, a2;

	toEuler (o, angs );
	a1 = angs;

	char c=' ';
	switch ( abs(limitaxis_id) ) {
	case 1: angs.x() = limitang; c='X'; break;
	case 2: angs.y() = limitang; c='Y'; break;
	case 3: angs.z() = limitang; c='Z'; break;
	}
	set (o, angs );
	normalize(o);
	toEuler( o, a2 );

	/*printf ( "Limit %c%c\n", (limitaxis_id<0) ? '-' : '+', c );
	printf ( "  before: <%3.2f, %3.2f, %3.2f>\n", a1.x,a1.y,a1.z);
	printf ( "  after:  <%3.2f, %3.2f, %3.2f>\n", a2.x,a2.y,a2.z);*/

}

void Joints::ApplyJacobianTranspose ( float amt )
{
	osg::Vec3 jT;
	osg::Vec3 dE; 
	osg::Quat dq, o1;
	osg::Vec3 angs, a1, a2, a3, a4, a5, a6;
	float dang;
	float lz;
	std::string msg;
	bool limit=false;

	// effector delta (dE)
	dE = m_Goal - m_Effector;
	dE.normalize();

	int M=0;
	for (int n=0; n < m_Joints.size(); n++ ) {
				
		toEuler(m_Joints[n].orient, angs );
		if (angs.z() < -90) angs.z() = 360 + angs.z();
		
		if ( m_Joints[n].dof.x()==1 ) {		
			jT.x() = m_Jacobian(M, 0);
			jT.y() = m_Jacobian(M, 1); 
			jT.z() = m_Jacobian(M, 2);
			dang = jT.x() * dE.x() + jT.y() * dE.y() + jT.z() * dE.z();			// multiply one row of J^T by dE vector
			dang *= amt;			
			if ( angs.x() + dang < m_Joints[n].min_limit.x() ) {									
				LimitQuaternion ( m_Joints[n].orient, -1, m_Joints[n].min_limit.x() );				
			} else if ( angs.x() + dang > m_Joints[n].max_limit.x() ) {
				LimitQuaternion ( m_Joints[n].orient, 1, m_Joints[n].max_limit.x() );				
			} else {				
				dq.makeRotate ( dang, osg::Vec3(1,0,0) );	// rotate around local X-axis	
				m_Joints[n].orient = m_Joints[n].orient * dq;
				normalize(m_Joints[n].orient);

			}									
			M++;
		}

		if ( m_Joints[n].dof.y()==1 ) {		
			jT.x() = m_Jacobian(M, 0);
			jT.y() = m_Jacobian(M, 1); 
			jT.z() = m_Jacobian(M, 2);
			dang = jT.x() * dE.x() + jT.y() * dE.y() + jT.z() * dE.z();			// multiply one row of J^T by dE vector
			dang *= amt;
			if ( angs.y() + dang < m_Joints[n].min_limit.y() ) {					
				LimitQuaternion ( m_Joints[n].orient, -2, m_Joints[n].min_limit.y() );						
			} else if ( angs.y() + dang > m_Joints[n].max_limit.y() ) {			
				LimitQuaternion ( m_Joints[n].orient, 2, m_Joints[n].max_limit.y() );					
			} else {				
				dq.makeRotate ( dang, osg::Vec3(0,1,0) );	// rotate around local Y-axis			
				m_Joints[n].orient = m_Joints[n].orient * dq;
				normalize(m_Joints[n].orient);
			}
			M++;
		}		

		if ( m_Joints[n].dof.z()==1 ) {			
			jT.x() = m_Jacobian(M, 0);
			jT.y() = m_Jacobian(M, 1); 
			jT.z() = m_Jacobian(M, 2);
			msg="";
			dang = jT.x() * dE.x() + jT.y() * dE.y() + jT.z() * dE.z();			// multiply one row of J^T by dE vector
			dang *= amt;			
			if ( angs.z() + dang < m_Joints[n].min_limit.z() ) {	
				LimitQuaternion ( m_Joints[n].orient, -3, m_Joints[n].min_limit.z() );				
			} else if ( angs.z() + dang > m_Joints[n].max_limit.z()  ) {				
				LimitQuaternion ( m_Joints[n].orient, 3, m_Joints[n].max_limit.z() );							
			} else {				
				dq.makeRotate ( dang, osg::Vec3(0,0,1) );	// rotate around local Z-axis
				m_Joints[n].orient = m_Joints[n].orient * dq;
				toEuler(m_Joints[n].orient,a4);
				normalize(m_Joints[n].orient);
				toEuler(m_Joints[n].orient,a5);
			}
			M++;
		}

		toEuler(m_Joints[n].orient, angs );
		//printf ("J%d  <%3.2f, %3.2f, %3.2f>  %3.2f\n", n, angs.x, angs.y, angs.z, a1.z );
	}

}

void Joints::ComputeJacobian ()
{
	osg::Vec3 r, c, axis, delta;
	osg::Vec3 dE;
	osg::Matrix mtx;
	
	// effector delta (dE)
	dE = m_Goal - m_Effector;
	dE.normalize();

	// process each joint to find DOFs
	int M =0;
	for (int n=0; n < m_Joints.size(); n++ ) {
		
		r = m_Effector - m_Joints[n].pos;			// r = e - joint_pos
		r.normalize();
		mtx = m_Joints[n].Mworld;					// local orientation of joint
		PostTranslate(mtx , m_Joints[n].pos * -1.0f );
		
		if ( m_Joints[n].dof.x() == 1) {
			// Use x-axis rotation on this joint
			axis.set( 1, 0, 0 );					// get the joints x-axis
			axis = axis* mtx;							// axis in world space			
			axis.normalize();
			delta = axis^r;				// J(phi) = axis X (E - p)    // X=cross product, E=end effector, p=joint position, axis=joint axis (in world space)
			delta.normalize();
			m_Jacobian(M,0) = delta.x();				// write to jacobian
			m_Jacobian(M,1) = delta.y();
			m_Jacobian(M,2) = delta.z();			
			M++;
		}
		if ( m_Joints[n].dof.y() == 1) {
			// Use y-axis rotation on this joint
			axis.set( 0, 1, 0 );					// get the joints y-axis
			axis = axis * mtx;							// rotation axis in world space
			axis.normalize();
			delta = axis ^ r;
			delta.normalize();
			m_Jacobian(M,0) = delta.x();				// write to jacobian
			m_Jacobian(M,1) = delta.y();
			m_Jacobian(M,2) = delta.z();			
			M++;
		}
		if ( m_Joints[n].dof.z() == 1) {
			// Use z-axis rotation on this joint
			axis.set( 0, 0, 1 );					// get the joints z-axis
			axis = axis * mtx;							// rotation axis in world space
			axis.normalize();
			delta = axis ^ r;
			delta.normalize();
			m_Jacobian(M,0) = delta.x();				// write to jacobian
			m_Jacobian(M,1) = delta.y();
			m_Jacobian(M,2) = delta.z();			
			M++;
		}	
	}

}



#define VNAME		F
#define VTYPE		float

// Constructors/Destructors

MatrixF::MatrixF(void) { data = NULL; Resize(0, 0); }

MatrixF::~MatrixF(void)
{
	if (data != NULL)
		free(data);
}
MatrixF::MatrixF(const int r, const int c) { data = NULL; Resize(r, c); }

// Member Functions

VTYPE MatrixF::GetVal(int c, int r)
{
#ifdef DEBUG_MATRIX
	if (data == NULL) Error.Print(ErrorLev::Matrix, ErrorDef::MatrixIsNull, true);
	if (r < 0 || r >= rows) Error.Print(ErrorLev::Matrix, ErrorDef::RowOutOfBounds, true);
	if (c < 0 || c >= cols) Error.Print(ErrorLev::Matrix, ErrorDef::ColOutOfBounds, true);
#endif
	return *(data + (r * cols + c));
}

VTYPE& MatrixF::operator () (const int c, const int r)
{
#ifdef DEBUG_MATRIX
	if (data == NULL)
		Error.Print(ErrorLev::Matrix, ErrorDef::MatrixIsNull, true);
	if (r < 0 || r >= rows)
		Error.Print(ErrorLev::Matrix, ErrorDef::RowOutOfBounds, true);
	if (c < 0 || c >= cols)
		Error.Print(ErrorLev::Matrix, ErrorDef::ColOutOfBounds, true);
#endif
	return *(data + (r * cols + c));
}
MatrixF& MatrixF::operator= (const unsigned char op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ = (VTYPE)op; return *this; }
MatrixF& MatrixF::operator= (const int op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ = (VTYPE)op; return *this; }
MatrixF& MatrixF::operator= (const double op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ = (VTYPE)op; return *this; }
MatrixF& MatrixF::operator= (const MatrixF& op) {
#ifdef DEBUG_MATRIX		
	if (op.data == NULL)						Debug.Print(DEBUG_MATRIX, "MatrixF::m=op: Operand matrix (op) data is null\n");
#endif
	if (rows != op.rows || cols != op.cols || data == NULL) Resize(op.cols, op.rows);
	memcpy(data, op.data, len * sizeof(VTYPE));
	return *this;
}

MatrixF& MatrixF::operator+= (const unsigned char op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ += (VTYPE)op; return *this; }
MatrixF& MatrixF::operator+= (const int op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ += (VTYPE)op; return *this; }
MatrixF& MatrixF::operator+= (const double op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ += (VTYPE)op; return *this; }
MatrixF& MatrixF::operator+= (const MatrixF& op) {
#ifdef DEBUG_MATRIX
	if (data == NULL)							Debug.Print(DEBUG_MATRIX, "MatrixF::m+=op: Matrix data is null\n");
	if (op.data == NULL)						Debug.Print(DEBUG_MATRIX, "MatrixF::m+=op: Operand matrix (op) data is null\n");
	if (rows != op.rows || cols != op.cols)		Debug.Print(DEBUG_MATRIX, "MatrixF::m+=op: Matricies must be the same size\n");
#endif
	VTYPE* n, * ne;
	VTYPE* b;
	n = data; ne = data + len; b = op.data;
	for (; n < ne;) *n++ += (VTYPE)*b++;
	return *this;
}

MatrixF& MatrixF::operator-= (const unsigned char op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ -= (VTYPE)op; return *this; }
MatrixF& MatrixF::operator-= (const int op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ -= (VTYPE)op; return *this; }
MatrixF& MatrixF::operator-= (const double op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ -= (VTYPE)op; return *this; }
MatrixF& MatrixF::operator-= (const MatrixF& op) {
#ifdef DEBUG_MATRIX
	if (data == NULL)							Debug.Print(DEBUG_MATRIX, "MatrixF::m-=op: Matrix data is null\n");
	if (op.data == NULL)						Debug.Print(DEBUG_MATRIX, "MatrixF::m-=op: Operand matrix (op) data is null\n");
	if (rows != op.rows || cols != op.cols)		Debug.Print(DEBUG_MATRIX, "MatrixF::m-=op: Matricies must be the same size\n");
#endif
	VTYPE* n, * ne;
	VTYPE* b;
	n = data; ne = data + len; b = op.data;
	for (; n < ne;) *n++ -= (VTYPE)*b++;
	return *this;
}

MatrixF& MatrixF::operator*= (const unsigned char op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ *= (VTYPE)op; return *this; }
MatrixF& MatrixF::operator*= (const int op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ *= (VTYPE)op; return *this; }
MatrixF& MatrixF::operator*= (const double op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ *= (VTYPE)op; return *this; }

MatrixF& MatrixF::operator*= (const MatrixF& op) {
#ifdef DEBUG_MATRIX
	if (data == NULL)							Debug.Print(DEBUG_MATRIX, "MatrixF::m*=op: Matrix data is null\n");
	if (op.data == NULL)						Debug.Print(DEBUG_MATRIX, "MatrixF::m*=op: Operand matrix (op) data is null\n");
	if (rows != op.rows || cols != op.cols)		Debug.Print(DEBUG_MATRIX, "MatrixF::m*=op: Matricies must be the same size\n");
#endif
	VTYPE* n, * ne;
	VTYPE* b;
	n = data; ne = data + len; b = op.data;
	for (; n < ne;) *n++ *= (VTYPE)*b++;
	return *this;
}

MatrixF& MatrixF::operator/= (const unsigned char op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ /= (VTYPE)op; return *this; }
MatrixF& MatrixF::operator/= (const int op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ /= (VTYPE)op; return *this; }
MatrixF& MatrixF::operator/= (const double op) { VTYPE* n = data, * nlen = data + len; for (; n < nlen;) *n++ /= (VTYPE)op; return *this; }
MatrixF& MatrixF::operator/= (const MatrixF& op) {
#ifdef DEBUG_MATRIX
	if (data == NULL)							Debug.Print(DEBUG_MATRIX, "MatrixF::m/=op: Matrix data is null\n");
	if (op.data == NULL)						Debug.Print(DEBUG_MATRIX, "MatrixF::m/=op: Operand matrix (op) data is null\n");
	if (rows != op.rows || cols != op.cols)		Debug.Print(DEBUG_MATRIX, "MatrixF::m/=op: Matricies must be the same size\n");
#endif
	VTYPE* n, * ne;
	VTYPE* b;
	n = data; ne = data + len; b = op.data;
	for (; n < ne;)
		if (*b != (VTYPE)0) {
			*n++ /= (VTYPE)*b++;
		}
		else {
			*n++ = (VTYPE)0; b++;
		}
	return *this;
}

MatrixF& MatrixF::Multiply(const MatrixF& op) {
#ifdef DEBUG_MATRIX 
	if (data == NULL)						debug.Print(DEBUG_MATRIX, "MatrixF::m*=op: Matrix data is null\n");
	if (op.data == NULL)					debug.Print(DEBUG_MATRIX, "MatrixF::m*=op: Operand matrix (op) data is null\n");
	if (cols != op.rows)					debug.Print(DEBUG_MATRIX, "MatrixF::m*=op: Matricies not compatible (m.cols != op.rows)\n");
#endif
	if (cols == op.rows) {
		VTYPE* newdata, * n, * ne, * a, * as;		// Pointers into A and new A matricies
		float* b, * bs, * bce, * be;				// Pointers into B matrix
		int newr = rows, newc = op.cols;		// Set new rows and columns
		int newlen = newr * newc;				// Determine new matrix size
		newdata = (VTYPE*)malloc(newlen * sizeof(VTYPE));			// Allocate new matrix to hold multiplication
		if (newdata == NULL) { printf((char*)"MatrixF::m*=op: Cannot allocate new matrix.\n"); exit(-1); }
		ne = newdata + newlen;					// Calculate end of new matrix
		int bskip = op.cols;					// Calculate row increment for B matrix	
		bce = op.data + bskip;					// Calculate end of first row in B matrix
		be = op.data + op.rows * op.cols;			// Calculate end of B matrix	
		as = data; bs = op.data;				// Goto start of A and B matricies
		for (n = newdata; n < ne;) {				// Compute C = A*B		
			a = as; b = bs;						// Goto beginning of row in A, top of col in B
			*n = (VTYPE)0;						// Initialize n element in C
			for (; b < be;) { *n += (*a++) * (*b); b += bskip; }	// Compute n element in C
			if (++bs >= bce) {					// If last col in B..
				bs = op.data;					// Go back to first column in B
				as += cols;					// Goto next row in A
			}
			n++;								// Goto next element in C
		}
		free(data);							// Destroy old A matrix
		data = newdata; rows = newr; cols = newc; len = newlen;		// Replace with new A matrix	
	}
	return *this;
}

MatrixF& MatrixF::Multiply4x4(const MatrixF& op) {
#ifdef DEBUG_MATRIX 
	if (data == NULL)						Debug.Print(DEBUG_MATRIX, "MatrixF::Multiply4x4 m*=op: Matrix data is null\n");
	if (op.data == NULL)					Debug.Print(DEBUG_MATRIX, "MatrixF::Multiply4x4 m*=op: Operand matrix (op) data is null\n");
	if (rows != 4 || cols != 4)				Debug.Print(DEBUG_MATRIX, "MatrixF::Multiply4x4 m*=op: Matrix m is not 4x4");
	if (op.rows != 4 || op.cols != 4)		Debug.Print(DEBUG_MATRIX, "MatrixF::Multiply4x4 m*=op: Matrix op is not 4x4");
#endif
	register double c1, c2, c3, c4;					// Temporary storage
	VTYPE* n, * a, * b1, * b2, * b3, * b4;
	a = data;	n = data;
	b1 = op.data; b2 = op.data + 4; b3 = op.data + 8; b4 = op.data + 12;

	c1 = *a++;	c2 = *a++;	c3 = *a++; c4 = *a++;					// Calculate First Row
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1) + c2 * (*b2) + c3 * (*b3) + c4 * (*b4);
	b1 -= 3; b2 -= 3; b3 -= 3; b4 -= 3;

	c1 = *a++;	c2 = *a++;	c3 = *a++; c4 = *a++;					// Calculate Second Row
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1) + c2 * (*b2) + c3 * (*b3) + c4 * (*b4);
	b1 -= 3; b2 -= 3; b3 -= 3; b4 -= 3;

	c1 = *a++;	c2 = *a++;	c3 = *a++; c4 = *a++;					// Calculate Third Row
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1) + c2 * (*b2) + c3 * (*b3) + c4 * (*b4);
	b1 -= 3; b2 -= 3; b3 -= 3; b4 -= 3;

	c1 = *a++;	c2 = *a++;	c3 = *a++; c4 = *a;						// Calculate Four Row
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n++ = c1 * (*b1++) + c2 * (*b2++) + c3 * (*b3++) + c4 * (*b4++);
	*n = c1 * (*b1) + c2 * (*b2) + c3 * (*b3) + c4 * (*b4);

	return *this;
}


MatrixF& MatrixF::Resize(const int x, const int y)
{
	if (data != NULL) {
		if (rows == y && cols == x) return *this;
		free(data);
	}
	rows = y; cols = x;
	if (y > 0 && x > 0) {
		len = rows * cols;
		if (len != 0) {
			data = new VTYPE[len];
#ifdef DEBUG_MATRIX
			if (data == NULL) Debug.Print(DEBUG_MATRIX, "MatrixF::Size: Out of memory for construction.\n");
#endif
		}
	}

#ifdef MATRIX_INITIALIZE
	if (data != NULL) memset(data, 0, sizeof(VTYPE) * len);
#endif		
	return *this;
}
MatrixF& MatrixF::ResizeSafe(const int x, const int y)
{
	VTYPE* newdata;
	int newlen;
	VTYPE* n, * ne;
	VTYPE* b, * be;
	int bskip;

	if (data != NULL) {
		newlen = x * y;
		newdata = (VTYPE*)malloc(newlen * sizeof(VTYPE));
#ifdef DEBUG_MATRIX
		if (newdata == NULL)
			Debug.Print(DEBUG_MATRIX, "MatrixF::SizeSafe: Out of memory for construction.\n");
#endif		
		if (y >= rows && x >= cols) {			// New size is larger (in both r and c)			
			memset(newdata, 0, newlen * sizeof(VTYPE));	// Clear new matrix
			ne = data + len;					// Calculate end of current matrix
			b = newdata;						// Start of new matrix
			be = newdata + cols;				// Last filled column+1 in new matrix
			bskip = x - cols;
			for (n = data; n < ne;) {				// Fill new matrix with old
				for (; b < be;) *b++ = *n++;
				b += bskip;
				be += x;
			}
		}
		else if (y < rows && x < cols) {		// New size is smaller (in both r and c)
			ne = newdata + newlen;			// Calculate end of new matrix
			b = data;						// Start of old matrix
			be = data + x;					// Last retrieved column+1 in old matrix
			bskip = cols - x;
			for (n = newdata; n < ne;) {		// Fill new matrix with old
				for (; b < be;) *n++ = *b++;
				b += bskip;
				be += x;
			}
		}
		else {							// Asymetrical resize
#ifdef DEBUG_MATRIX
			Debug.Print(DEBUG_MATRIX, "MatrixF::SizeSafe: Asymetrical resize NOT YET IMPLEMENTED.\n");
#endif
			exit(202);
		}
		free(data);
		rows = y; cols = x;
		data = newdata; len = newlen;
	}
	else {
		len = (rows = y) * (cols = x);
		data = (VTYPE*)malloc(len * sizeof(VTYPE));
#ifdef DEBUG_MATRIX
		if (data == NULL)
			Debug.Print(DEBUG_MATRIX, "MatrixF::SizeSafe: Out of memory for construction.\n");
#endif
	}
	return *this;
}
MatrixF& MatrixF::InsertRow(const int r)
{
	VTYPE* newdata;
	VTYPE* r_src, * r_dest;
	int newlen;

	if (data != NULL) {
		newlen = (rows + 1) * cols;
		newdata = (VTYPE*)malloc(newlen * sizeof(VTYPE));
#ifdef DEBUG_MATRIX
		if (newdata == NULL)
			Debug.Print(DEBUG_MATRIX, "MatrixF::InsertRow: Out of memory for construction.\n");
#endif
		memcpy(newdata, data, r * cols * sizeof(VTYPE));
		if (r < rows) {
			r_src = data + r * cols;
			r_dest = newdata + (r + 1) * cols;
			if (r < rows) memcpy(r_dest, r_src, (rows - r) * cols * sizeof(VTYPE));
		}
		r_dest = newdata + r * cols;
		memset(r_dest, 0, cols * sizeof(VTYPE));
		rows++;
		free(data);
		data = newdata; len = newlen;
	}
	else {
#ifdef DEBUG_MATRIX
		Debug.Print(DEBUG_MATRIX, "MatrixF::InsertRow: Cannot insert row in a null matrix.\n");
#endif
	}
	return *this;
}
MatrixF& MatrixF::InsertCol(const int c)
{
	VTYPE* newdata;
	int newlen;

	if (data != NULL) {
		newlen = rows * (cols + 1);
		newdata = (VTYPE*)malloc(newlen * sizeof(VTYPE));
#ifdef DEBUG_MATRIX
		if (newdata == NULL)
			Debug.Print(DEBUG_MATRIX, "MatrixF::InsertCol: Out of memory for construction.\n");
#endif
		VTYPE* n, * ne;
		VTYPE* b, * be;
		int bskip, nskip;

		if (c > 0) {
			n = data;							// Copy columns to left of c
			ne = data + len;
			nskip = (cols - c);
			b = newdata;
			be = newdata + c;
			bskip = (cols - c) + 1;
			for (; n < ne;) {
				for (; b < be; ) *b++ = *n++;
				b += bskip;
				be += (cols + 1);
				n += nskip;
			}
		}
		if (c < cols) {
			n = data + c;						// Copy columns to right of c
			ne = data + len;
			nskip = c;
			b = newdata + (c + 1);
			be = newdata + (cols + 1);
			bskip = c + 1;
			for (; n < ne;) {
				for (; b < be; ) *b++ = *n++;
				b += bskip;
				be += (cols + 1);
				n += nskip;
			}
		}
		cols++;
		for (n = newdata + c, ne = newdata + len; n < ne; n += cols) *n = (VTYPE)0;
		free(data);
		data = newdata; len = newlen;
	}
	else {
#ifdef DEBUG_MATRIX
		Debug.Print(DEBUG_MATRIX, "MatrixF::InsertCol: Cannot insert col in a null matrix.\n");
#endif
	}
	return *this;
}
MatrixF& MatrixF::Transpose(void)
{
	VTYPE* newdata;
	int r = rows;

	if (data != NULL) {
		if (rows == 1) {
			rows = cols; cols = 1;
		}
		else if (cols == 1) {
			cols = rows; rows = 1;
		}
		else {
			len = rows * cols;
			newdata = (VTYPE*)malloc(len * sizeof(VTYPE));
#ifdef DEBUG_MATRIX
			if (newdata == NULL)
				Debug.Print(DEBUG_MATRIX, "MatrixF::Transpose: Out of memory for construction.\n");
#endif	
			VTYPE* n, * ne;
			VTYPE* b, * be;
			n = data;						// Goto start of old matrix
			ne = data + len;
			b = newdata;					// Goto start of new matrix
			be = newdata + len;
			for (; n < ne; ) {				// Copy rows of old to columns of new
				for (; b < be; b += r) *b = *n++;
				b -= len;
				b++;
			}
		}
		free(data);
		data = newdata;
		rows = cols; cols = r;
	}
	else {
#ifdef DEBUG_MATRIX
		Debug.Print(DEBUG_MATRIX, "MatrixF::Transpose: Cannot transpose a null matrix.\n");
#endif
	}
	return *this;
}
MatrixF& MatrixF::Identity(const int order)
{
	Resize(order, order);
	VTYPE* n, * ne;
	memset(data, 0, len * sizeof(VTYPE));	// Fill matrix with zeros
	n = data;
	ne = data + len;
	for (; n < ne; ) {
		*n = 1;								// Set diagonal element to 1
		n += cols;
		n++;								// Next diagonal element
	}
	return *this;
}

// rotates points >>counter-clockwise<< when looking down the X+ axis toward the origin
MatrixF& MatrixF::RotateX(const double ang)
{
	Resize(4, 4);
	VTYPE* n = data;
	double c, s;
	c = cos(ang * 3.141592 / 180);
	s = sin(ang * 3.141592 / 180);
	*n = 1; n += 5;
	*n++ = (VTYPE)c;	*n = (VTYPE)s; n += 3;
	*n++ = (VTYPE)-s;	*n = (VTYPE)c; n += 5;
	*n = 1;
	return *this;
}

// rotates points >>counter-clockwise<< when looking down the Y+ axis toward the origin
MatrixF& MatrixF::RotateY(const double ang)
{
	Resize(4, 4);
	VTYPE* n = data;
	double c, s;
	c = cos(ang * 3.141592 / 180);
	s = sin(ang * 3.141592 / 180);
	*n = (VTYPE)c;		n += 2;
	*n = (VTYPE)-s;	n += 3;
	*n = 1;				n += 3;
	*n = (VTYPE)s;		n += 2;
	*n = (VTYPE)c;		n += 5;
	*n = 1;
	return *this;
}

// rotates points >>counter-clockwise<< when looking down the Z+ axis toward the origin
MatrixF& MatrixF::RotateZ(const double ang)
{
	Resize(4, 4);
	VTYPE* n = data;
	double c, s;
	c = cos(ang * 3.141592 / 180);
	s = sin(ang * 3.141592 / 180);
	*n++ = (VTYPE)c;	*n = (VTYPE)s; n += 3;
	*n++ = (VTYPE)-s;	*n = (VTYPE)c; n += 5;
	*n = 1; n += 5; *n = 1;
	return *this;
}
MatrixF& MatrixF::Ortho(double sx, double sy, double vn, double vf)
{
	// simplified version of OpenGL's glOrtho function
	VTYPE* n = data;
	*n++ = (VTYPE)(1.0 / sx); *n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0;
	*n++ = (VTYPE)0.0; *n++ = (VTYPE)(1.0 / sy); *n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0;
	*n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0; *n++ = (VTYPE)(-2.0 / (vf - vn)); *n++ = (VTYPE)(-(vf + vn) / (vf - vn));
	*n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0; *n++ = (VTYPE)0; *n++ = (VTYPE)1.0;
	return *this;
}

MatrixF& MatrixF::Translate(double tx, double ty, double tz)
{
	Resize(4, 4);
	VTYPE* n = data;
	*n++ = (VTYPE)1.0; *n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0;
	*n++ = (VTYPE)0.0; *n++ = (VTYPE)1.0; *n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0;
	*n++ = (VTYPE)0.0; *n++ = (VTYPE)0.0; *n++ = (VTYPE)1.0; *n++ = (VTYPE)0.0;
	*n++ = (VTYPE)tx; *n++ = (VTYPE)ty; *n++ = (VTYPE)tz; *n++ = (VTYPE)1.0;
	return *this;
}

MatrixF& MatrixF::Basis(const osg::Vec3& c1, const osg::Vec3& c2, const osg::Vec3& c3)
{
	Resize(4, 4);
	VTYPE* n = data;
	*n++ = (VTYPE)c1.x(); *n++ = (VTYPE)c2.x(); *n++ = (VTYPE)c3.x(); *n++ = (VTYPE)0;
	*n++ = (VTYPE)c1.y(); *n++ = (VTYPE)c2.y(); *n++ = (VTYPE)c3.y(); *n++ = (VTYPE)0;
	*n++ = (VTYPE)c1.z(); *n++ = (VTYPE)c2.z(); *n++ = (VTYPE)c3.z(); *n++ = (VTYPE)0;
	*n++ = (VTYPE)0; *n++ = (VTYPE)0; *n++ = (VTYPE)0; *n++ = (VTYPE)0;
	return *this;
}

#define		SWAP(a, b)		{temp=(a); (a)=(b); (b)=temp;}

MatrixF& MatrixF::GaussJordan(MatrixF& b)
{
	// Gauss-Jordan solves the matrix equation Ax = b
	// Given the problem:
	//		A*x = b		(where A is 'this' matrix and b is provided)
	// The solution is:
	//		Ainv*b = x
	// This function returns Ainv in A and x in b... that is:
	//		A (this) -> Ainv
	//		b -> solution x
	//

	MatrixF index_col, index_row;
	MatrixF piv_flag;
	int r, c, c2, rs, cs;
	double piv_val;
	int piv_row, piv_col;
	double pivinv, dummy, temp;

#ifdef DEBUG_MATRIX
	if (rows != cols) Debug.Print(DEBUG_MATRIX, "MatrixF::GaussJordan: Number of rows and cols of A must be equal.\n");
	if (rows != b.rows) Debug.Print(DEBUG_MATRIX, "MatrixF::GaussJordan: Number of rows of A and rows of b must be equal.\n");
	if (b.cols != 1) Debug.Print(DEBUG_MATRIX, "MatrixF::GaussJordan: Number of cols of b must be 1.\n");
#endif

	index_col.Resize(cols, 1);
	index_row.Resize(cols, 1);
	piv_flag.Resize(cols, 1);
	piv_flag = 0;
	for (c = 0; c < cols; c++) {
		piv_val = 0.0;
		for (rs = 0; rs < rows; rs++) {
			if (piv_flag(rs, 0) != 1)
				for (cs = 0; cs < cols; cs++) {
					if (piv_flag(cs, 0) == 0) {
						if (fabs((*this) (cs, rs)) >= piv_val) {
							piv_val = fabs((*this) (cs, rs));
							piv_row = rs;
							piv_col = cs;
						}
					}
					else if (piv_flag(cs, 0) > 1) {
#ifdef DEBUG_MATRIX
						Debug.Print(DEBUG_MATRIX, "MatrixF::GaussJordan: Singular matrix (dbl pivs).\n");
						//Print ();
#endif
					}
				}
		}
		piv_flag(piv_col, 0)++;
		if (piv_row != piv_col) {
			for (c2 = 0; c2 < cols; c2++) SWAP((*this) (c2, piv_row), (*this) (c2, piv_col));
			for (c2 = 0; c2 < b.cols; c2++) SWAP(b(c2, piv_row), b(c2, piv_col));
		}
		index_row(c, 0) = piv_row;
		index_col(c, 0) = piv_col;
		if ((*this) (piv_col, piv_col) == 0.0) {
#ifdef DEBUG_MATRIX
			Debug.Print(DEBUG_MATRIX, "MatrixF::GaussJordan: Singular matrix (0 piv).\n");
			//Print ();
#endif
		}
		pivinv = 1.0 / ((*this) (piv_col, piv_col));
		(*this) (piv_col, piv_col) = 1.0;
		for (c2 = 0; c2 < cols; c2++) (*this) (c2, piv_col) *= pivinv;
		for (c2 = 0; c2 < b.cols; c2++) b(c2, piv_col) *= pivinv;
		for (r = 0; r < rows; r++) {
			if (r != piv_col) {
				dummy = (*this) (piv_col, r);
				(*this) (piv_col, r) = 0.0;
				for (c2 = 0; c2 < cols; c2++) (*this) (c2, r) -= (*this) (c2, piv_col) * dummy;
				for (c2 = 0; c2 < b.cols; c2++) b(c2, r) -= b(c2, piv_col) * dummy;
			}
		}
	}
	for (c = cols - 1; c >= 0; c--) {
		if (index_row(c, 0) != index_col(c, 0))
			for (r = 0; r < rows; r++)
				SWAP((*this) (index_row(c, 0), r), (*this) (index_col(c, 0), r));
	}
	return *this;
}
MatrixF& MatrixF::Submatrix(MatrixF& b, int mx, int my)
{
	VTYPE* pEnd = data + rows * cols;		// end of matrix
	VTYPE* pVal = data;
	VTYPE* pNewVal = b.data;
	VTYPE* pNewEnd = pNewVal + mx;
	int pNewSkip = cols - mx;

	for (pVal = data; pVal < pEnd;) {
		for (; pNewVal < pNewEnd;) *pVal++ = *pNewVal++;
		pNewVal += pNewSkip;
		pNewEnd += mx;
	}
	return *this;
}

// N-Vector Dot Product
// Elements may be in rows or columns, but:
// - If in rows, number of columns must be one and number of rows must match.
// - If in cols, number of rows must be one and number of cols must match.
double MatrixF::Dot(MatrixF& b)
{
	double d = 0.0;
	VTYPE* pA = data;
	VTYPE* pB = b.data;

	if (rows == 1 && b.rows == 1 && cols == b.cols) {
		VTYPE* pAEnd = data + cols;
		d = 0.0;
		for (; pA < pAEnd;)
			d += (*pA++) * (*pB++);
	}
	else if (cols == 1 && b.cols == 1 && rows == b.rows) {
		VTYPE* pAEnd = data + rows;
		d = 0.0;
		for (; pA < pAEnd;)
			d += (*pA++) * (*pB++);
	}
	return d;
}

#define I(x, y)		( (y*xres) + x )
#define Ix(r)		( r % xres )			// X coordinate from row	
#define Iy(r)		( r / xres )			// Y coordinate from row

MatrixF& MatrixF::MatrixVector5(MatrixF& x, int mrows, MatrixF& b)
{
	double v;

	// A( 2, r ) * B ( r ) + A(1,r)*B(r-1) + A(3,r)*B(r+1) + A(0, r)*B( R-( r ) ) + A(4, r)*B( R+( r ) )
	for (int r = 0; r < mrows; r++) {
		v = GetVal(2, r) * x(0, r);
		if (r > 0) v += GetVal(1, r) * x(0, r - 1);
		if (r < mrows - 1) v += GetVal(3, r) * x(0, r + 1);
		if ((int)GetVal(5, r) >= 0) v += GetVal(0, r) * x(0, (int)GetVal(5, r));
		if ((int)GetVal(6, r) >= 0) v += GetVal(4, r) * x(0, (int)GetVal(6, r));
		b(0, r) = v;
	}
	return *this;
}

MatrixF& MatrixF::ConjugateGradient(MatrixF& b)
{
	return *this;
}

// Sparse Conjugate Gradient 2D (special case)
// This compute conjugate gradients on a 
// sparse "5-7" x N positive definite matrix. 
// Only 'mrows' subset of the row-size of A and b will be used.
MatrixF& MatrixF::ConjugateGradient5(MatrixF& b, int mrows)
{
	double a, g, rdot;
	int i, imax;
	MatrixF x, xnew;				// solution vector
	MatrixF r, rnew;				// residual
	MatrixF p, ptemp;				// search direction
	MatrixF v;

	x.Resize(1, mrows);
	xnew.Resize(1, mrows);
	r.Resize(1, mrows);
	rnew.Resize(1, mrows);
	p.Resize(1, mrows);
	ptemp.Resize(1, mrows);
	v.Resize(1, mrows);

	r.Submatrix(b, 1, mrows);
	MatrixVector5(x, mrows, v);				// (Ax -> v)
	r -= v;										// r = b - Ax
	p = r;

	imax = 20;
	for (i = 0; i < imax; i++) {
		MatrixVector5(p, mrows, v);			// v = Ap
		rdot = r.Dot(r);
		a = rdot / p.Dot(v);					// a = (r . r) / (p . v)		
		xnew = p;
		xnew *= a;
		xnew += x;								// x = x + p*a
		v *= a;
		rnew = r;								// rnew = r - v*a
		rnew -= v;
		g = rnew.Dot(rnew) / rdot;			// g = (rnew . rnew) / (r . r)
		p *= g;
		p += rnew;								// p = rnew + p*g
		r = rnew;
		x = xnew;
	}
	for (int rx = 0; rx < mrows; rx++)
		b(0, rx) = x(0, rx);
	return *this;
}

int MatrixF::GetX() { return cols; }
int MatrixF::GetY() { return rows; }
int MatrixF::GetRows(void) { return rows; }
int MatrixF::GetCols(void) { return cols; }
int MatrixF::GetLength(void) { return len; }
VTYPE* MatrixF::GetData(void) { return data; }

VTYPE MatrixF::GetF(const int r, const int c) { return (VTYPE)(*(data + r * cols + c)); }

void MatrixF::GetRowVec(int r, osg::Vec3& v)
{
	VTYPE* n = data + r * cols;
	v.x() = (float)*n++; v.y() = (float)*n++; v.z() = (float)*n++;
}

void MatrixF::Print(char* fname)
{
	char buf[2000];

#ifdef _MSC_VER
	FILE* fp;
	fopen_s(&fp, fname, "w+");
#else
	FILE* fp = fopen(fname, "w+");
#endif

	for (int r = 0; r < rows; r++) {
		buf[0] = '\0';
		for (int c = 0; c < cols; c++) {
#ifdef _MSC_VER
			sprintf_s(buf, "%s %04.3f", buf, GetVal(c, r));
#else
			sprintf(buf, "%s %04.3f", buf, GetVal(c, r));
#endif
		}
		fprintf(fp, "%s\n", buf);
	}
	fprintf(fp, "---------------------------------------\n%s\n", buf);
	fflush(fp);
	fclose(fp);
}




#undef VTYPE
#undef VNAME