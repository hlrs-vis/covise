#
#   converted with COVISE (C) map_converter
#   from t.net
#
# get file names
import os





def printUsage(error):
    print("ERROR:", error)
    print
    print("USAGE  : calcCovise <outFile.covise> \"<expression>\" [<scalar1.covise>] [<scalar2.covise>] [<vector1.covise>] [<vector2.covise>]")
    print("EXAMPLE: calcCovise outfile.covise \"v1*s1\" scalarInFile.covise vectorInFile.covise")
    print
    print("outFile    : The output filename. Can be vector or scalar depending on the expression.")
    print("expression : Use any expression containing s1, s2, v1 or v2 for the given input files.")
    print("             For a complete syntax, see the documentation of the Calc module.")
    print("             You can also use one of the following keywords:")
    print("               sadd: Adds two scalar values \"s1+s2\"")
    print("               ssub: Subtracts two scalar values \"s1-s2\"")
    print("               smul: Multiplies two scalar values \"s1*s2\"")
    print("               sdiv: Divides a scalar value by another \"s1/s2\"")
    print("               sneg: Negates a scalar value \"neg(s1)\"")
    print("               vadd: Adds two vectors \"v1+v2\"")
    print("               vsub: Subtracts two vectors \"v1-v2\"")
    print("               vmul: Multiplies a vector by a scalar value \"v1*s1\"")
    print("               vdiv: Divides a vector by a scalar value \"v1/s1\"")
    print("               vneg: Negates a vector \"neg(v1)\"")
    print("scalar1")
    print("scalar2")
    print("vector1")
    print("vector2    : The input files. Only the files used in the expression must be stated.")
    print("             The order is always (s1, s2, v1, v2), no matter of the order in the expression.")
    sys.exit()





arguments = os.getenv("CONVERTFILES")
files = arguments.split(' ')

# check minimum required arguments
if len(files)<3:
    printUsage("You need at least three arguments")

# check output filename
outfile = files[0]
if not outfile.endswith(".covise"):
    printUsage("outFile has to be a covise file")

# process expression
expression = files[1]
# convert keywords
if (expression == "sadd"):
    expression = "s1+s2"
if (expression == "ssub"):
    expression = "s1-s2"
if (expression == "smul"):
    expression = "s1*s2"
if (expression == "sdiv"):
    expression = "s1/s2"
if (expression == "sneg"):
    expression = "neg(s1)"
if (expression == "vadd"):
    expression = "v1+v2"
if (expression == "vsub"):
    expression = "v1-v2"
if (expression == "vmul"):
    expression = "v1*s1"
if (expression == "vdiv"):
    expression = "v1/s2"
if (expression == "vneg"):
    expression = "neg(v1)"

# check expression
s1used = (expression.find("s1") > -1)
s2used = (expression.find("s2") > -1)
v1used = (expression.find("v1") > -1)
v2used = (expression.find("v2") > -1)
# at least one infile is nescessary
if not (s1used or s2used or v1used or v2used):
    printUsage("Expression must use at least one infile")
# determine output format
vectorOut = v1used or v2used

# assure that there is the correct number of arguments
if len(files) != 2+int(s1used)+int(s2used)+int(v1used)+int(v2used):
    printUsage("Not enough filenames for the expression used")

# check given filenames
current = 2
if s1used:
    s1filename = files[current]
    if not files[current].endswith(".covise"):
        printUsage("scalar1 has to be a covise file")
    current=current+1
if s2used:
    s2filename = files[current]
    if not files[current].endswith(".covise"):
        printUsage("scalar2 has to be a covise file")
    current=current+1
if v1used:
    v1filename = files[current]
    if not files[current].endswith(".covise"):
        printUsage("vector1 has to be a covise file")
    current=current+1
if v2used:
    v2filename = files[current]
    if not files[current].endswith(".covise"):
        printUsage("vector2 has to be a covise file")
    current=current+1





# Print Summary

print("Expression:", expression)
print("Output filename:", outfile,)
if (vectorOut):
    print("(Vector)")
else:
    print("(Scalar)")
if s1used:
    print("Scalar1 filename:", s1filename)
if s2used:
    print("Scalar2 filename:", s2filename)
if v1used:
    print("Vector1 filename:", v1filename)
if v2used:
    print("Vector1 filename:", v2filename)
print





##################################
# GO
##################################

theNet = net()

Calc_1 = Calc()
theNet.add( Calc_1 )
Calc_1.set_expression( expression )

if s1used:
    RWCovise_s1 = RWCovise()
    theNet.add( RWCovise_s1 )
    RWCovise_s1.set_grid_path(s1filename )
    RWCovise_s1.set_stepNo( 0 )
    RWCovise_s1.set_rotate_output( "FALSE" )
    theNet.connect( RWCovise_s1, "mesh", Calc_1, "s_indata1" )
if s2used:
    RWCovise_s2 = RWCovise()
    theNet.add( RWCovise_s2 )
    RWCovise_s2.set_grid_path(s2filename )
    RWCovise_s2.set_stepNo( 0 )
    RWCovise_s2.set_rotate_output( "FALSE" )
    theNet.connect( RWCovise_s2, "mesh", Calc_1, "s_indata2" )
if v1used:
    RWCovise_v1 = RWCovise()
    theNet.add( RWCovise_v1 )
    RWCovise_v1.set_grid_path(v1filename )
    RWCovise_v1.set_stepNo( 0 )
    RWCovise_v1.set_rotate_output( "FALSE" )
    theNet.connect( RWCovise_v1, "mesh", Calc_1, "v_indata1" )
if v2used:
    RWCovise_v2 = RWCovise()
    theNet.add( RWCovise_v2 )
    RWCovise_v2.set_grid_path(v2filename )
    RWCovise_v2.set_stepNo( 0 )
    RWCovise_v2.set_rotate_output( "FALSE" )
    theNet.connect( RWCovise_v2, "mesh", Calc_1, "v_indata2" )

RWCovise_out = RWCovise()
theNet.add( RWCovise_out)
RWCovise_out.set_grid_path( outfile )
RWCovise_out.set_stepNo( 0 )
RWCovise_out.set_rotate_output( "FALSE" )
if (vectorOut):
    theNet.connect( Calc_1, "outdata2", RWCovise_out, "mesh_in" )
else:
    theNet.connect( Calc_1, "outdata1", RWCovise_out, "mesh_in" )

runMap()
theNet.finishedBarrier()
sys.exit()

