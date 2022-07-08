namespace OpenFOAMInterface.BIM.Enums
{
    /// <summary>
    /// Initial boundary parameter for simulation.
    /// </summary>
    public enum InitialFOAMParameter
    {
        U = 0,
        p,
        nut,
        epsilon,
        k,
        T,
        p_rgh,
        alphat
    }

    /// <summary>
    /// Preconditioner for fvSolution-Solver.
    /// </summary>
    public enum Preconditioner
    {
        DIC,
        Diagonal,
        DILU,
        FDIC = 0,
        GAMG
    }

    /// <summary>
    /// Patchtype for initialParameter.
    /// </summary>
    public enum PatchType
    {
        wall = 0,
        inlet,
        outlet,
        floor,
        sky,
        symmetry,
        sidewalls,
        none
    }

    //TO-DO: ADD INLET OUTLET BOUNDARY TYPE ENUMS LIKE FIXEDFLUXPRESSURE OR FIXEDVALUE.

    /// <summary>
    /// Enum-Objects for simulationmodel LES.
    /// </summary>
    public enum LESModel
    {
        //TO-DO: DONT DELETE THIS! EVERY MODEL NEEDS TO BE IMPLEMENTED WITH BOUNDARYCONDITION DEPENDENCY
        //NOT IMPLEMENTED FOAMPARAMETER DEPENDENCY YET!
        //DeardorffDiffStress = 0,
        //Smagorinsky,
        //SpalartAllmarasDDES,
        //SpalartAllmarasDES,
        //SpalartAllmarasIDDES,
        //WALE,
        //dynamicKEqn,
        //dynamicLagrangian,
        //kEqn,
        //kOmegaSSTDES
        //
        //implement in AddLESModelParameterToList
        //
    }

    /// <summary>
    /// Enum-Objects for simulationmodel RAS.
    /// </summary>
    public enum RASModel
    {
        //LRR = 0,
        ////NOT IMPLEMENTED FOAMPARAMETER DEPENDENCY YET!
        //LamBremhorstKE,
        //LaunderSharmaKE,
        //LienCubicKE,
        //LienLeschzine,
        RNGkEpsilon,
        //NOT IMPLEMENTED FOAMPARAMETER DEPENDENCY YET!
        //SSG,
        //ShihQuadraticKE,
        //buoyantKEpsilon,
        //SpalartAllmaras,
        kEpsilon,
        //NOT IMPLEMENTED FOAMPARAMETER DEPENDENCY YET!
        //kOmega,
        //kOmegaSST,
        //kOmegaSSTLM,
        //kOmegaSSTSAS,
        //kkLOmega,
        //qZeta,
        //realizableKE,
        //v2f
        //
        //implement in AddRASModelParameterToList 
        //
    }

    /// <summary>
    /// Enum for simulationtype.
    /// </summary>
    public enum SimulationType
    {
        laminar = 0,
        RAS,
        LES
    }

    /// <summary>
    /// Enum for TransportModel in TransportProperties.
    /// </summary>
    public enum TransportModel
    {
        Newtonian = 0,
        //BirdCarreau,
        //CrossPowerLaw,
        //powerLaw,
        //HerschelBulkley,
        //Casson,
        //strainRateFunction
    }

    /// <summary>
    /// ExtractionMethode for SurfaceFeatuerExtract.
    /// </summary>
    public enum ExtractionMethod
    {
        none = 0,
        extractFromFile,
        extractFromSurface
    }

    /// <summary>
    /// MethodDecompose in DecomposeParDict.
    /// </summary>
    public enum MethodDecompose
    {
        simple = 0,
        hierarchical,
        scotch,
        manual
    }

    /// <summary>
    /// OpenFOAM simulation environment.
    /// </summary>
    public enum OpenFOAMEnvironment
    {
        blueCFD = 0,
        //docker, //not implemented yet
        wsl,
        ssh
    }

    /// <summary>
    /// Agglomerator for fvSolution.
    /// </summary>
    public enum Agglomerator
    {
        faceAreaPair = 0
    }


    /// <summary>
    /// Smoother for fvSolution.
    /// </summary>
    public enum Smoother
    {
        GaussSeidel = 0,
        symGaussSeidel,
        DIC,
        DILU,
        DICGaussSeidel
    }

    /// <summary>
    /// Solver for fvSolution.
    /// </summary>
    public enum SolverFV
    {
        PCG = 0,
        PBiCGStab,
        PBiCG,
        smoothSolver,
        GAMG,
        diagonal
    }

    /// <summary>
    /// cacheAgglomeration in fvSolution.
    /// </summary>
    public enum CacheAgglomeration
    {
        on = 0,
        off
    }

    /// <summary>
    /// startFrom in controlDict.
    /// </summary>
    public enum StartFrom
    {
        firstTime = 0,
        startTime,
        latestTime
    }

    /// <summary>
    /// stop in controlDict.
    /// </summary>
    public enum StopAt
    {
        endTime = 0,
        writeNow,
        noWriteNow,
        nextWrite
    }

    /// <summary>
    /// writeControl in controlDict.
    /// </summary>
    public enum WriteControl
    {
        timeStep = 0,
        runTime,
        adjustableRunTime,
        cpuTime,
        clockTime
    }

    /// <summary>
    /// format in controlDict.
    /// </summary>
    public enum WriteFormat
    {
        ascii = 0,
        binary
    }

    /// <summary>
    /// writeCompresion in controlDict.
    /// </summary>
    public enum WriteCompression
    {
        on = 0,
        off
    }

    /// <summary>
    /// Timeformat in controlDict.
    /// </summary>
    public enum TimeFormat
    {
        Fixed = 0,
        scientific,
        general
    }

    /// <summary>
    /// Solver for controlDict.
    /// </summary>
    public enum SolverControlDict
    {
        //Incompressible
        simpleFoam = 0,
        adjointShapeOptimizationFoam,
        boundaryFoam,
        icoFoam,
        nonNewtonianIcoFoam,
        pimpleDyMFoam,
        pimpleFoam,
        pisoFoam,
        porousSimpleFoam,
        shallowWaterFoam,
        SRFPimpleFoam,
        SRFSimpleFoam,

        //HeatTransfer
        buoyantBoussinesqSimpleFoam,
        buoyantBoussinesqPimpleFoam,
        buoyantPimpleFoam,
        buoyantSimpleFoam,
        chtMultiRegionFoam,
        chtMultiRegionSimpleFoam
    }

    /// <summary>
    /// The file format of STL.
    /// </summary>
    public enum SaveFormat
    {
        binary = 0,
        ascii
    }

    /// <summary>
    /// The type of mesh.
    /// </summary>
    public enum MeshType
    {
        Snappy = 0,
        cfMesh
    }

    /// <summary>
    /// The range of elements to be exported.
    /// </summary>
    public enum ElementsExportRange
    {
        All = 0,
        OnlyVisibleOnes
    }
}