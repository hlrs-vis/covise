Converts ascii files to pts divided models...

ascii syntax (i is intensity, this value is ignored)
------------
x, y, z, r, g, b, i


Command line arguments (specify file(s) with point data (above syntax) followed by name of output
----------------------
PointConvert file1.xyz file2.xyz file3.xyz result.pts


Note
----

Currently there are two params under main, one to set a maximum number of
points per cube and the other specifies the number of segments along the
longest dimension to divide up the space the points are bound in.
