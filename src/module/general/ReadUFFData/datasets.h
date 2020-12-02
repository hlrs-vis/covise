/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DATASETS_H_
#define DATASETS_H_
/*
Universal Dataset Number 15

 
Name:   Nodes
Status: Obsolete
Owner:  Simulation
Revision Date: 30-Aug-1987
Additional Comments: This dataset is written by I-DEAS Test.
-----------------------------------------------------------------------
 
             Record 1: FORMAT(4I10,1P3E13.5)
                       Field 1 -    node label
                       Field 2 -    definition coordinate system number
                       Field 3 -    displacement coordinate system number
                       Field 4 -    color
                       Field 5-7 -  3 - Dimensional coordinates of node
                                    in the definition system
 
             NOTE:  Repeat record for each node
*/
struct dataset15
{
    struct
    {
        long label; //node label
        long defcosysnum; //definition coordinate system number
        long discosysnum; //displacement coordinate system number
        int color; //should also be _int64 but probably not necessary
        float p[3]; //coordinate p[0] = x, p[1] = y, p[2] = z
    } record1; //nodelabel, defcosysnum, discosysnum, color, p
};

/*
Universal Dataset Number 55
von zopeown Zuletzt verändert: 02.05.2007 06:50

 
Name:   Data at Nodes
Status: Obsolete
Owner:  Simulation
Revision Date: 07-Mar-1997
Additional Comments: This dataset is written and read by I-DEAS Test.
-----------------------------------------------------------------------
 
          RECORD 1:      Format (40A2)
               FIELD 1:          ID Line 1
 
          RECORD 2:      Format (40A2)
               FIELD 1:          ID Line 2
 
          RECORD 3:      Format (40A2)
 
               FIELD 1:          ID Line 3
 
          RECORD 4:      Format (40A2)
               FIELD 1:          ID Line 4
 
          RECORD 5:      Format (40A2)
               FIELD 1:          ID Line 5
 
          RECORD 6:      Format (6I10)
 
          Data Definition Parameters
 
               FIELD 1: Model Type
                           0:   Unknown
                           1:   Structural
                           2:   Heat Transfer
                           3:   Fluid Flow
 
               FIELD 2:  Analysis Type
                           0:   Unknown
                           1:   Static
                           2:   Normal Mode
                           3:   Complex eigenvalue first order
                           4:   Transient
                           5:   Frequency Response
                           6:   Buckling
                           7:   Complex eigenvalue second order
 
               FIELD 3:  Data Characteristic
                           0:   Unknown
                           1:   Scalar
                           2:   3 DOF Global Translation Vector
                           3:   6 DOF Global Translation & Rotation Vector
                           4:   Symmetric Global Tensor
                           5:   General Global Tensor
 

 
               FIELD 4: Specific Data Type
                           0:   Unknown
                           1:   General
                           2:   Stress
                           3:   Strain (Engineering)
                           4:   Element Force
                           5:   Temperature
                           6:   Heat Flux
                           7:   Strain Energy
                           8:   Displacement
                           9:   Reaction Force
                           10:   Kinetic Energy
                           11:   Velocity
                           12:   Acceleration
                           13:   Strain Energy Density
                           14:   Kinetic Energy Density
                           15:   Hydro-Static Pressure
                           16:   Heat Gradient
                           17:   Code Checking Value
                           18:   Coefficient Of Pressure
 
               FIELD 5:  Data Type
                           2:   Real
                           5:   Complex
 
               FIELD 6:  Number Of Data Values Per Node (NDV)
 

 
          Records 7 And 8 Are Analysis Type Specific
 
          General Form
 
          RECORD 7:      Format (8I10)
 
               FIELD 1:          Number Of Integer Data Values
                           1 < Or = Nint < Or = 10
               FIELD 2:          Number Of Real Data Values
                           1 < Or = Nrval < Or = 12
               FIELDS 3-N:       Type Specific Integer Parameters
 
 
          RECORD 8:      Format (6E13.5)
               FIELDS 1-N:       Type Specific Real Parameters
 
 
          For Analysis Type = 0, Unknown
 
          RECORD 7:
 
               FIELD 1:   1
               FIELD 2:   1
               FIELD 3:   ID Number
 
          RECORD 8:
 
               FIELD 1:   0.0
 
          For Analysis Type = 1, Static
 
          RECORD 7:
               FIELD 1:    1
               FIELD 2:    1
               FIELD 3:    Load Case Number
 
          RECORD 8:
               FIELD 11:    0.0
 
          For Analysis Type = 2, Normal Mode
 

 
         RECORD 7:
 
               FIELD 1:    2
               FIELD 2:    4
               FIELD 3:    Load Case Number
               FIELD 4:    Mode Number
 
          RECORD 8:
               FIELD 1:    Frequency (Hertz)
               FIELD 2:    Modal Mass
               FIELD 3:    Modal Viscous Damping Ratio
               FIELD 4:    Modal Hysteretic Damping Ratio
 
          For Analysis Type = 3, Complex Eigenvalue
 
          RECORD 7:
               FIELD 1:    2
               FIELD 2:    6
               FIELD 3:    Load Case Number
               FIELD 4:    Mode Number
 
          RECORD 8:
 
               FIELD 1:    Real Part Eigenvalue
               FIELD 2:    Imaginary Part Eigenvalue
               FIELD 3:    Real Part Of Modal A
               FIELD 4:    Imaginary Part Of Modal A
               FIELD 5:    Real Part Of Modal B
               FIELD 6:    Imaginary Part Of Modal B
 
 
          For Analysis Type = 4, Transient
 
          RECORD 7:
 
               FIELD 1:    2
               FIELD 2:    1
               FIELD 3:    Load Case Number
               FIELD 4:    Time Step Number
 
          RECORD 8:
               FIELD 1: Time (Seconds)
 

 
          For Analysis Type = 5, Frequency Response
 
          RECORD 7:
 
               FIELD 1:    2
               FIELD 2:    1
               FIELD 3:    Load Case Number
               FIELD 4:    Frequency Step Number
 
          RECORD 8:
               FIELD 1:    Frequency (Hertz)
 
          For Analysis Type = 6, Buckling
 
          RECORD 7:
 
               FIELD 1:    1
               FIELD 2:    1
               FIELD 3:    Load Case Number
 
          RECORD 8:
 
               FIELD 1: Eigenvalue
 
          RECORD 9:      Format (I10)
 
               FIELD 1:          Node Number
 
          RECORD 10:     Format (6E13.5)
               FIELDS 1-N:       Data At This Node (NDV Real Or
                         Complex Values)
 
          Records 9 And 10 Are Repeated For Each Node.
 

 
          Notes:
          1        Id Lines May Not Be Blank.  If No Information Is
                      Required, The Word "None" Must Appear  Columns 1-4.
 
          2        For Complex Data There Will Be 2*Ndv Data Items At Each
                      Node. The Order Is Real Part For Value 1,  Imaginary
                      Part For Value 1, Etc.
          3        The Order Of Values For Various Data  Characteristics
                      Is:
                          3 DOF Global Vector:
                                  X, Y, Z

 
                          6 DOF Global Vector:
                                  X, Y, Z,
                                  Rx, Ry, Rz

 
                          Symmetric Global Tensor:
                                  Sxx, Sxy, Syy,
                                  Sxz, Syz, Szz

 
                          General Global Tensor:
                                  Sxx, Syx, Szx,
                                  Sxy, Syy, Szy,
                                  Sxz, Syz, Szz

 
                          Shell And Plate Element Load:
                                  Fx, Fy, Fxy,
                                  Mx, My, Mxy,
                                  Vx, Vy
 
          4        Id Line 1 Always Appears On Plots In Output Display.
          5        If Specific Data Type Is "Unknown," ID Line 2 Is
                      Displayed As Data Type In Output Display.
          6        Typical Fortran I/O Statements For The Data Sections
                      Are:
 
                                   Read(Lun,1000)Num
                                   Write
                          1000 Format (I10)
                                   Read(Lun,1010) (VAL(I),I=1,NDV)
                                   Write
                          1010 format (6e13.5)
 
 
                          Where:     Num Is Node Number 
                                     Val Is Real Or Complex Data  Array
                                     Ndv Is Number Of Data Values  Per Node
 
          7        Data Characteristic Values Imply The Following Values
                      Of Ndv:
                                      Scalar: 1
                                      3 DOF Global Vector: 3
                                      6 DOF Global Vector: 6
                                      Symmetric Global Tensor: 6
                                      General Global Tensor: 9
 
          8        Data Associated With I-DEAS Test Has The Following
                      Special Forms of Specific Data Type and ID Line 5.

 
                   For Record 6 Field 4-Specific Data Type, values 0
                      through 12 are as defined above.  13 and 15 
                      through 19 are:

 
                               13: excitation force
                               15: pressure
                               16: mass
                               17: time
                               18: frequency
                               19: rpm

 
                   The form of ID Line 5 is:
 
                   Format (6I10)
                   FIELD 1:  Reference Coordinate Label

 
                   FIELD 2:  Reference Coordinate Direction
                                1: X Direction
                               -1: -X Direction
                                2: Y Direction
                               -2: -Y Direction
                                3: Z Direction
                               -3: -Z Direction
 
                   FIELD 3:  Numerator Signal Code
                                see Specific Data Type above
 
                   FIELD 4:  Denominator Signal Code
                                see Specific Data Type above

 
                   FIELD 5:  Response Coordinate Label

 
                   FIELD 6:  Response Coordinate Direction
                                see Reference Coordinate Direction above

 
                   Also note that the modal mass in record 8 is calculated
                   from the parameter table by I-DEAS Test.
 
          9        Any Record With All 0.0's Data Entries Need Not (But
                      May) Appear.
 
          10       A Direct Result Of 9 Is That If No Records 9 And 10
                      Appear, All Data For The Data Set Is 0.0.
 
          11       When New Analysis Types Are Added, Record 7 Fields 1
                      And 2 Are Always > Or = 1 With Dummy Integer And
                      Real Zero Data If Data Is Not Required. If Complex
                      Data Is Needed, It Is Treated As Two Real Numbers,
                      Real Part Followed By Imaginary Point.
 
          12       Dataloaders Use The Following ID Line Convention:
 
                              1.   (80A1) Model
                                  Identification
                              2.   (80A1) Run
                                  Identification
                              3.   (80A1) Run
                                  Date/Time
                              4.   (80A1) Load Case
                                  Name
 
                          For Static:
 
                              5.   (17h Load Case
                                  Number;, I10) For
                                  Normal Mode:
                              5.   (10h Mode Same,
                                  I10, 10H Frequency,
                                  E13.5)
          13       No Maximum Value For Ndv .
 
          14       Typical Fortran I/O Statements For Processing Records 7
                      And 8.

 
                            Read (LUN,1000)NINT,NRVAL,(IPAR(I),I=1,NINT
                       1000 Format (8I10)
                            Read (Lun,1010) (RPAV(I),I=1,NRVAL)
                       1010 Format (6E13.5)
 
          15       For Situations With Reduced # Dof's, Use 3 DOF
                      Translations Or 6 DOF Translation And Rotation With
                      Unused Values = 0.
*/
struct dataset55
{
    struct
    {
        char idLine[80];
    } record1;

    struct
    {
        char idLine[80];
    } record2;

    struct
    {
        char idLine[80];
    } record3;

    struct
    {
        char idLine[80];
    } record4;

    struct
    {
        char idLine[80];
    } record5;

    struct
    {
        int modelType;
        int analysisType;
        int dataCharacteristic;
        int specificDataType;
        int dataType;
        int numberOfDataValuesPerNode;
    } record6;

    //Record 7 and 8 are analysis type specific, see description for further information
    struct
    {
        int numberOfIntegerDataValues;
        int numberOfRealDataValues;
        //Type specific integer parameters
        int iParams[6];
    } record7;

    struct
    {
        float fParams[6];
    } record8;

    //records 9 and 10 are repeated for each node
    struct
    {
        int *nodeNumber;
    } record9;

    struct
    {
        int numDataValues; //just for help. is calculated by record6 dataType and numberOfDataValuesPerNode
        float **dataValues; //TODO: Maybe there are more values than 6 for complex data
    } record10;
};

/*
Universal Dataset Number: 58
  Universal Dataset
  Number: 58
  Name: Function at Nodal DOF
  Status: Current
  Owner: Test
  Revision Date: 23-Apr-1993
  
    Record 1: Format(80A1)
      Field 1 - ID Line 1
      NOTE
      ID Line 1 is generally used for the function description.
    Record 2: Format(80A1)
      Field 1 - ID Line 2
    Record 3: Format(80A1)
      Field 1 - ID Line 3
      NOTE
      ID Line 3 is generally used to identify when the function was created. The date is in the form
      DD-MMM-YY, and the time is in the form HH:MM:SS, with a general Format(9A1,1X,8A1).
    Record 4: Format(80A1)
      Field 1 - ID Line 4
    Record 5: Format(80A1)
      Field 1 - ID Line 5
    Record 6: Format(2(I5,I10),2(1X,10A1,I10,I4))
    DOF Identification
      Field 1 - Function Type
         0 - General or Unknown
          1 - Time Response
         2 - Auto Spectrum
         3 - Cross Spectrum
         4 - Frequency Response Function
         5 - Transmissibility
         6 - Coherence
         7 - Auto Correlation
         8 - Cross Correlation
         9 - Power Spectral Density (PSD)
        10 - Energy Spectral Density (ESD)
        11 - Probability Density Function
        12 - Spectrum
        13 - Cumulative Frequency Distribution
        14 - Peaks Valley
        15 - Stress/Cycles
        16 - Strain/Cycles
        17 - Orbit
        18 - Mode Indicator Function
        19 - Force Pattern
        20 - Partial Power
        21 - Partial Coherence
        22 - Eigenvalue
        23 - Eigenvector
        24 - Shock Response Spectrum
        25 - Finite Impulse Response Filter
        26 - Multiple Coherence
        27 - Order Function
      Field 2 - Function Identification Number
      Field 3 - Version Number, or sequence number
      Field 4 - Load Case Identification Number
        0 - Single Point Excitation
      Field 5 - Response Entity Name ("NONE" if unused)
      Field 6 - Response Node
      Field 7 - Response Direction
        0 - Scalar
        1 - +X Translation  4 - +X Rotation
         -1 - -X Translation -4 - -X Rotation
        2 - +Y Translation  5 - +Y Rotation
         -2 - -Y Translation -5 - -Y Rotation
        3 - +Z Translation  6 - +Z Rotation
         -3 - -Z Translation -6 - -Z Rotation
      Field 8 - Reference Entity Name ("NONE" if unused)
      Field 9 - Reference Node
      Field 10 - Reference Direction (same as field 7)
        NOTE
        Fields 8, 9, and 10 are only relevant if field 4
        is zero.
    Record 7: Format(3I10,3E13.5)
      Data Form
      Field 1 - Ordinate Data Type
        2 - real, single precision
        4 - real, double precision
        5 - complex, single precision
        6 - complex, double precision
      Field 2 - Number of data pairs for uneven abscissa
            spacing, or number of data values for even abscissa spacing
      Field 3 - Abscissa Spacing
        0 - uneven
        1 - even (no abscissa values stored)
      Field 4 - Abscissa minimum (0.0 if spacing uneven)
      Field 5 - Absciissa increment (0.0 if spacing uneven)
      Field 6 - Z-axis value (0.0 if unused)
    Record 8: Format(I10,3I5,2(1X,20A1)) Abscissa Data Characteristics
      Field 1 - Specific Data Type
        0 - unknown
        1 - general
        2 - stress
        3 - strain
        5 - temperature
        6 - heat flux
        8 - displacement
        9 - reaction force
        11 - velocity
        12 - acceleration
        13 - excitation force
        15 - pressure
        16 - mass
        17 - time
        18 - frequency
        19 - rpm
        20 - order
      Field 2 - Length units exponent
      Field 3 - Force units exponent
      Field 4 - Temperature units exponent
        NOTE
        Fields 2, 3 and 4 are relevant only if the
        Specific Data Type is General, or in the case of
        ordinates, the response/reference direction is a
        scalar, or the functions are being used for
        nonlinear connectors in System Dynamics Analysis.
        See Addendum ’A’ for the units exponent table.
      Field 5 - Axis label ("NONE" if not used)
      Field 6 - Axis units label ("NONE" if not used)
        NOTE
        If fields 5 and 6 are supplied, they take
        precedence over program generated labels and units.
    Record 9: Format(I10,3I5,2(1X,20A1))
        Ordinate (or ordinate numerator) Data Characteristics
    Record 10: Format(I10,3I5,2(1X,20A1))
        Ordinate Denominator Data Characteristics
    Record 11: Format(I10,3I5,2(1X,20A1))
        Z-axis Data Characteristics
        NOTE
        Records 9, 10, and 11 are always included and
        have fields the same as record 8. If records 10
        and 11 are not used, set field 1 to zero.
    Record 12:
      Data Values
      Ordinate Abscissa
      Case Type Precision Spacing Format
      -------------------------------------------------------------
      1 real single even 6E13.5
      2 real single uneven 6E13.5
      3 complex single even 6E13.5
      4 complex single uneven 6E13.5
      5 real double even 4E20.12
      6 real double uneven 2(E13.5,E20.12)
      7 complex double even 4E20.12
      8 complex double uneven E13.5,2E20.12
      --------------------------------------------------------------
      NOTE
      See Addendum ’B’ for typical FORTRAN READ/WRITE
      statements for each case.
  General Notes:
  1. ID lines may not be blank. If no information is required,
     the word "NONE" must appear in columns 1 through 4.
    2. ID line 1 appears on plots in Finite Element Modeling and is
     used as the function description in System Dynamics Analysis.
  3. Dataloaders use the following ID line conventions
     ID Line 1 - Model Identification
       ID Line 2 - Run Identification
     ID Line 3 - Run Date and Time
     ID Line 4 - Load Case Name
  4. Coordinates codes from MODAL-PLUS and MODALX are decoded into
     node and direction.
  5. Entity names used in System Dynamics Analysis prior to I-DEAS
     Level 5 have a 4 character maximum. Beginning with Level 5,
     entity names will be ignored if this dataset is preceded by
     dataset 259. If no dataset 259 precedes this dataset, then the
     entity name will be assumed to exist in model bin number 1.
  6. Record 10 is ignored by System Dynamics Analysis unless load
     case = 0. Record 11 is always ignored by System Dynamics
  Analysis.
  7. In record 6, if the response or reference names are "NONE"
     and are not overridden by a dataset 259, but the corresponding
     node is non-zero, System Dynamics Analysis adds the node
     and direction to the function description if space is
     sufficient
  8. ID line 1 appears on XY plots in Test Data Analysis along
     with ID line 5 if it is defined. If defined, the axis units
     labels also appear on the XY plot instead of the normal
     labeling based on the data type of the function.
  9. For functions used with nonlinear connectors in System
     Dynamics Analysis, the following requirements must be
     adhered to:
     a) Record 6: For a displacement-dependent function, the
        function type must be 0; for a frequency-dependent
        function, it must be 4. In either case, the load case
        identification number must be 0.
     b) Record 8: For a displacement-dependent function, the
        specific data type must be 8 and the length units
        exponent must be 0 or 1; for a frequency-dependent
        function, the specific data type must be 18 and the
        length units exponent must be 0. In either case, the
        other units exponents must be 0.
     c) Record 9: The specific data type must be 13. The
        temperature units exponent must be 0. For an ordinate
        numerator of force, the length and force units
        exponents must be 0 and 1, respectively. For an
        ordinate numerator of moment, the length and force
        units exponents must be 1 and 1, respectively.
     d) Record 10: The specific data type must be 8 for
      stiffness and hysteretic damping; it must be 11
      for viscous damping. For an ordinate denominator of
      translational displacement, the length units exponent
      must be 1; for a rotational displacement, it must
        be 0. The other units exponents must be 0.
     e) Dataset 217 must precede each function in order to
      define the function’s usage (i.e. stiffness, viscous
      damping, hysteretic damping).
  Addendum A
  In order to correctly perform units conversion, length, force, and
  temperature exponents must be supplied for a specific data type of
  General; that is, Record 8 Field 1 = 1. For example, if the function
  has the physical dimensionality of Energy (Force * Length), then the
  required exponents would be as follows:
  Length = 1
  Force = 1 Energy = L * F
  Temperature = 0
  Units exponents for the remaining specific data types should not be
  supplied. The following exponents will automatically be used.
  Table - Unit Exponents
  -------------------------------------------------------
  Specific Direction
  ---------------------------------------------
  Data Translational Rotational
  ---------------------------------------------
  Type Length Force Temp Length Force Temp
  -------------------------------------------------------
  0  0  0  0  0  0  0
  1  (requires input to fields 2,3,4)
  2  -2  1  0  -1  1  0
  3  0  0  0  0  0  0
  5  0  0  1  0  0  1
  6  1  1  0  1  1  0
  8  1  0  0  0  0  0
  9  0  1  0  1  1  0
  11  1  0  0  0  0  0
  12  1  0  0  0  0  0
  13  0  1  0  1  1  0
  15 -2  1  0  -1  1  0
  16 -1  1  0  1  1  0
  17  0  0  0  0  0  0
  18  0  0  0  0  0  0
  19  0  0  0  0  0  0
  --------------------------------------------------------
  NOTE
  Units exponents for scalar points are defined within
  System Analysis prior to reading this dataset.
  Addendum B
  There are 8 distinct combinations of parameters which affect the
  details of READ/WRITE operations. The parameters involved are
  Ordinate Data Type, Ordinate Data Precision, and Abscissa Spacing.
  Each combination is documented in the examples below. In all cases,
  the number of data values (for even abscissa spacing) or data pairs
  (for uneven abscissa spacing) is NVAL. The abscissa is always real
  single precision. Complex double precision is handled by two real
  double precision variables (real part followed by imaginary part)
  because most systems do not directly support complex doubleprecision.
*/

struct dataset58
{
    //Id Line 1
    struct
    {
        char idLine[80];
    } record1; //idLine

    //Id Line 2
    struct
    {
        char idLine[80];
    } record2; //idLine

    //Id Line 3
    struct
    {
        char idLine[80];
    } record3; //idLine

    //Id Line 4
    struct
    {
        char idLine[80];
    } record4; //idLine

    //Id Line 5
    struct
    {
        char idLine[80];
    } record5; //idLine

    //DOF Identification
    struct
    {

        int functionType;
        int functionID;
        int versionNumber;
        int loadCaseIdendificationNumber;
        char responseEntityName[10];
        int responseNode;
        int responseDirection;
        char referenceEntityName[10];
        int referenceNode;
        int referenceDirection;
    } record6; //functionType, functionID, versionNumber, loadCaseIdendificationNumber, responseEntityName, responseNode, responseDirection, referenceEntityName, referenceNode, referenceDirection

    //Data Form
    struct
    {
        int ordinateDataType;
        union
        {
            int numDataPairs;
            int numDataValues;
        };
        int abscissaSpacing;
        float abscissaMinimum;
        float abscissaIncrement;
        float zAxisValue;
    } record7; // ordinateDataType, numberDataPairs, abscissaSpacing, abscissaMinimum, abscissaIncrement, zAxisValue

    //Abscissa Data Characteristics
    struct
    {
        int specificDataType;
        int lengthUnitsExponent;
        int forceUnitsExponent;
        int temperatureUnitsExponent;
        char axisLabel[20];
        char axisUnitsLabel[20];
    } record8; //specificDataType, lengthUnitsExponent, forceUnitsExponent, temperatureUnitsExponent, axisLabel, axisUnitsLabel

    //Ordinate (or ordinate numerator) Data Characteristics
    struct
    {
        int specificDataType;
        int lengthUnitsExponent;
        int forceUnitsExponent;
        int temperatureUnitsExponent;
        char axisLabel[20];
        char axisUnitsLabel[20];
    } record9; //specificDataType, lengthUnitsExponent, forceUnitsExponent, temperatureUnitsExponent, axisLabel, axisUnitsLabel

    //Ordinate Denominator Data Characteristics
    struct
    {
        int specificDataType;
        int lengthUnitsExponent;
        int forceUnitsExponent;
        int temperatureUnitsExponent;
        char axisLabel[20];
        char axisUnitsLabel[20];
    } record10; //specificDataType, lengthUnitsExponent, forceUnitsExponent, temperatureUnitsExponent, axisLabel, axisUnitsLabel

    /*
  Z-axis Data Characteristics
  NOTE
  Records 9, 10, and 11 are always included and
  have fields the same as record 8. If records 10
  and 11 are not used, set field 1 to zero.
  */
    struct
    {
        int specificDataType;
        int lengthUnitsExponent;
        int forceUnitsExponent;
        int temperatureUnitsExponent;
        char axisLabel[20];
        char axisUnitsLabel[20];
    } record11; //specificDataType, lengthUnitsExponent, forceUnitsExponent, temperatureUnitsExponent, axisLabel, axisUnitsLabel

    //Data Values
    struct
    {
        union // Data for either single, or double precision
        {
            double *datad;
            float *dataf;
        };

        unsigned int num;
    } record12; //data
};

/*
Universal Dataset Number 82

 
Name:   Tracelines
Status: Obsolete
Owner:  Simulation
Revision Date: 27-Aug-1987
Additional Comments:  This dataset is written by I-DEAS Test.
-----------------------------------------------------------------------
 
             Record 1: FORMAT(3I10)
                       Field 1 -    trace line number
                       Field 2 -    number of nodes defining trace line
                                    (maximum of 250)
                       Field 3 -    color
 
             Record 2: FORMAT(80A1)
                       Field 1 -    Identification line
 
             Record 3: FORMAT(8I10)
                       Field 1 -    nodes defining trace line
                               =    > 0 draw line to node
                               =    0 move to node (a move to the first
                                    node is implied)
             Notes: 1) MODAL-PLUS node numbers must not exceed 8000.
                    2) Identification line may not be blank.
                    3) Systan only uses the first 60 characters of the
                       identification text.
                    4) MODAL-PLUS does not support trace lines longer than
                       125 nodes.
                    5) Supertab only uses the first 40 characters of the
                       identification line for a name.
                    6) Repeat Datasets for each Trace_Line
*/

struct dataset82
{ //long should be __int64 because 10-digit numbers should be possible to load, but __int64 is microsoft specific
    struct
    {
        long traceLineNumber; //Traceline number
        int numNodes; //number of nodes defining traceline
        long color;
    } record1; //traceLineNumber, numNodes, color

    struct
    {
        char idLine[80]; //Identification line
    } record2; //idLine

    struct
    {
        int *traceNodes; //nodes defining trace line
    } record3; //traceNodes
};

/*
Universal Dataset Number 151

 
Name:   Header
Status: Current
Owner:  General
Revision Date: 13-Dec-1993
-----------------------------------------------------------------------

 
Record 1:       FORMAT(80A1)
                Field 1      -- model file name
Record 2:       FORMAT(80A1)
                Field 1      -- model file description
Record 3:       FORMAT(80A1)
                Field 1      -- program which created DB
Record 4:       FORMAT(10A1,10A1,3I10)
                Field 1      -- date database created (DD-MMM-YY)
                Field 2      -- time database created (HH:MM:SS)
                Field 3      -- Version from database
                Field 4      -- Subversion from database
                Field 5      -- File type
                                =0  Universal
                                =1  Archive
                                =2  Other
Record 5:       FORMAT(10A1,10A1)
                Field 1      -- date database last saved (DD-MMM-YY)
                Field 2      -- time database last saved (HH:MM:SS)
Record 6:       FORMAT(80A1)
                Field 1      -- program which created universal file
Record 7:       FORMAT(10A1,10A1,4I5)
                Field 1      -- date universal file written (DD-MMM-YY)
                Field 2      -- time universal file written (HH:MM:SS)
                Field 3      -- Release which wrote universal file
                Field 4      -- Version number
                Field 5      -- Host ID
                                MS1.  1-Vax/VMS 2-SGI     3-HP7xx,HP-UX
                                      4-RS/6000 5-Alp/VMS 6-Sun 7-Sony
                                      8-NEC     9-Alp/OSF
                Field 6      -- Test ID
                Field 7      -- Release counter per host

*/

//File Header
struct dataset151
{
    struct
    {
        char modelName[80]; //the models name
    } record1; //modelName

    struct
    {
        char modelFileDesc[80]; //model file description
    } record2; //modelFileDesc

    struct
    {
        char DBProgram[80]; //program which created DB
    } record3; //DBProgram

    struct
    {
        char DBCreateDate[10]; //date database created (DD-MMM-YY)
        char DBCreateTime[10]; //time    -"-      (HH:MM:SS)
        unsigned int DBVersion; //database version
        unsigned int DBSubversion; //database subversion
        unsigned short fileType; //File type =0  Universal, =1  Archive, =2  Other
    } record4; /*DBCreateDate, DBCreateTime, DBVersion, DBSubversion, fileType*/

    struct
    {
        char DBLastSavedDate[10]; //date database last saved (DD-MMM-YY)
        char DBLastSavedTime[10]; //time database last saved (HH:MM:SS)
    } record5; //DBLastSavedDate, DBLastSavedTime

    struct
    {
        char UFProgramName[80]; //program which created universal file
    } record6; //UFProgramName

    struct
    {
        char UFWrittenDate[10]; //date universal file written (DD-MMM-YY)
        char UFWrittenTime[10]; //time universal file written (HH:MM:SS)
        unsigned int release; //Release which wrote universal file
        unsigned int version; //Version number
        unsigned int hostID; //Host ID MS1.  1-Vax/VMS 2-SGI, 3-HP7xx,HP-UX, 4-RS/6000, 5-Alp/VMS, 6-Sun, 7-Sony, 8-NEC, 9-Alp/OSF
        unsigned int testID; //Test ID
        unsigned int releaseCounterPerHost; //Release counter per host
    } record7; //UFWrittenDate, UFWrittenTime, release, version, hostID, testID, releaseCounterPerHost
};

/*
Universal Dataset Number 164

Universal Dataset
Number: 164
Name:   Units
Status: Current
Owner:  General
Revision Date: 19-AUG-1987
-----------------------------------------------------------------------

Record 1:       FORMAT(I10,20A1,I10)
                Field 1      -- units code
                                = 1 - SI: Meter (newton)
                                = 2 - BG: Foot (pound f)
                                = 3 - MG: Meter (kilogram f)
                                = 4 - BA: Foot (poundal)
                                = 5 - MM: mm (milli newton)
                                = 6 - CM: cm (centi newton)
                                = 7 - IN: Inch (pound f)
                                = 8 - GM: mm (kilogram f)
                                = 9 - US: USER_DEFINED
                                = 10- MN: mm (newton)
                Field 2      -- units description (used for
                                documentation only)
                Field 3      -- temperature mode
                                = 1 - absolute
                                = 2 - relative
Record 2:       FORMAT(3D25.17)
                Unit factors for converting universal file units to SI.
                To convert from universal file units to SI divide by
                the appropriate factor listed below.
                Field 1      -- length
                Field 2      -- force
                Field 3      -- temperature
                Field 4      -- temperature offset

Example:

    -1
   164
         2Foot (pound f)               2
  3.28083989501312334D+00  2.24808943099710480D-01  1.79999999999999999D+00
  4.59670000000000002D+02
    -1

*/
struct dataset164
{
    //actual double has not the required precision, but there is no data type with higher precision
    struct
    {
        char unitsDesc[20];
        long unitsCode;
        long tempMode;
    } record1; //unitsDesc, unitsCode, tempMode

    struct
    {
        double facLength;
        double facForce;
        double facTemp;
        double facTempOff;
    } record2; //facLength, facForce, facTemp, facTempOff
};

/*
Universal Dataset Number 2411
von zopeown Zuletzt verändert: 02.05.2007 06:56

Name:   Nodes - Double Precision
Status: Current
Owner:  Simulation
Revision Date: 23-OCT-1992 
----------------------------------------------------------------------------

Record 1:        FORMAT(4I10)
                 Field 1       -- node label
                 Field 2       -- export coordinate system number
                 Field 3       -- displacement coordinate system number
                 Field 4       -- color
Record 2:        FORMAT(1P3D25.16)
                 Fields 1-3    -- node coordinates in the part coordinate
                                  system
 
Records 1 and 2 are repeated for each node in the model.
 
Example:
 
    -1
  2411
       121         1         1        11
   5.0000000000000000D+00   1.0000000000000000D+00   0.0000000000000000D+00
       122         1         1        11
   6.0000000000000000D+00   1.0000000000000000D+00   0.0000000000000000D+00
    -1
 
----------------------------------------------------------------------------
*/

struct dataset2411
{
    struct
    {
        int nodeLabel;
        int exportCoordSysNum;
        int dispCoordSysNum;
        int color;
    } record1;

    struct
    {
        double coords[3];
    } record2;
};
#endif
