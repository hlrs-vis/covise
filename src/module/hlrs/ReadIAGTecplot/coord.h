#ifndef COORD_H
#define COORD_H

#include <cmath>


class Coordinate
{
public:
	Coordinate() { pos[0] = 0;  pos[1] = 0; pos[2] = 0; };
	Coordinate(float v) { pos[0] = v;  pos[1] = v; pos[2] = v; };
	Coordinate(float x, float y, float z) { pos[0] = x;  pos[1] = y; pos[2] = z; };
	float pos[3];
	float &operator[](int i) { return pos[i]; };
	Coordinate &operator=(float v) { pos[0] = v; pos[1] = v; pos[2] = v; return *this; };/** Binary vector add. */
	inline const Coordinate operator + (const Coordinate& rhs) const
	{
		return Coordinate(pos[0] + rhs.pos[0], pos[1] + rhs.pos[1], pos[2] + rhs.pos[2]);
	}

	/** Unary vector add. Slightly more efficient because no temporary
	* intermediate object.
	*/
	inline Coordinate& operator += (const Coordinate& rhs)
	{
		pos[0] += rhs.pos[0];
		pos[1] += rhs.pos[1];
		pos[2] += rhs.pos[2];
		return *this;
	}

	/** Binary vector subtract. */
	inline const Coordinate operator - (const Coordinate& rhs) const
	{
		return Coordinate(pos[0] - rhs.pos[0], pos[1] - rhs.pos[1], pos[2] - rhs.pos[2]);
	}

	/** Unary vector subtract. */
	inline Coordinate& operator -= (const Coordinate& rhs)
	{
		pos[0] -= rhs.pos[0];
		pos[1] -= rhs.pos[1];
		pos[2] -= rhs.pos[2];
		return *this;
	}

	/** Negation operator. Returns the negative of the Coordinate. */
	inline const Coordinate operator - () const
	{
		return Coordinate(-pos[0], -pos[1], -pos[2]);
	}

	/** Length of the vector = sqrt( vec . vec ) */
	inline float length() const
	{
		return sqrtf(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
	}

	/** Multiply by scalar. */
	inline const Coordinate operator * (float rhs) const
	{
		return Coordinate(pos[0] * rhs, pos[1] * rhs, pos[2] * rhs);
	}

	/** Unary multiply by scalar. */
	inline Coordinate& operator *= (float rhs)
	{
		pos[0] *= rhs;
		pos[1] *= rhs;
		pos[2] *= rhs;
		return *this;
	}

	/** Divide by scalar. */
	inline const Coordinate operator / (float rhs) const
	{
		return Coordinate(pos[0] / rhs, pos[1] / rhs, pos[2] / rhs);
	}

	/** Unary divide by scalar. */
	inline Coordinate& operator /= (float rhs)
	{
		pos[0] /= rhs;
		pos[1] /= rhs;
		pos[2] /= rhs;
		return *this;
	}

	/** Cross product. */
	inline const Coordinate operator ^ (const Coordinate& rhs) const
	{
		return Coordinate(pos[1] * rhs.pos[2] - pos[2] * rhs.pos[1],
			pos[2] * rhs.pos[0] - pos[0] * rhs.pos[2],
			pos[0] * rhs.pos[1] - pos[1] * rhs.pos[0]);
	}

	/** Dot product. */
	inline float operator * (const Coordinate& rhs) const
	{
		return pos[0] * rhs.pos[0] + pos[1] * rhs.pos[1] + pos[2] * rhs.pos[2];
	}

	float X() const { return pos[0]; };
	float Y() const { return pos[1]; };
	float Z() const { return pos[2]; };
	float &X() { return pos[0]; };
	float &Y() { return pos[1]; };
	float &Z() { return pos[2]; };

};

inline std::ostream & operator<<(std::ostream & out, Coordinate const & iSrc) {
	return out << iSrc.X() << '/' << iSrc.Y() << '/' << iSrc.Z();
}
#endif
