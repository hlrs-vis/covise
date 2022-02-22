using Autodesk.Revit.DB;
using System.Windows.Media.Media3D;
using System.Collections.Generic;
using System;

namespace OpenFOAMInterface.BIM.Structs
{
    using Enums;

    namespace General
    {
        using FOAM;

        /// <summary>
        /// SSH struct contains all informations about the tunnel-connection.
        /// </summary>
        readonly public struct SSH
        {
            private static SSH def = new SSH();
            public static ref readonly SSH Default => ref def;
            public SSH() : this("username", "hostname", "openfoam alias", "path/to/compute/dir/on/server",
                true, false, true, 22, "eval salloc -n 16")
            { }

            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="_user">The user to login.</param>
            /// <param name="_ip">IP of the server.</param>
            /// <param name="_alias">Alias for starting openfoam.</param>
            /// <param name="_caseFolder">Casefolder on server.</param>
            /// <param name="_download">if true, case folder will be downloaded from server after simulation.</param>
            /// <param name="_delete">if true, case folder will be deleted after simulation.</param>
            /// <param name="_port">SSH Port.</param>
            /// <param name="_slurmCommand">Slurm command specify tags.</param>
            /// <param name="_slurm">use slurm.</param>
            // public SSH(string user, string ip, string alias, string caseFolder, bool download, bool delete, bool slurm, int port, string slurmCommand)
            public SSH(in string user, in string ip, in string alias, in string caseFolder, bool download, bool delete, bool slurm, int port, in string slurmCommand)
            {
                this.User = user;
                this.ServerIP = ip;
                this.OfAlias = alias;
                this.ServerCaseFolder = caseFolder;
                this.Download = download;
                this.Delete = delete;
                this.Slurm = slurm;
                this.Port = port;
                this.SlurmCommand = slurmCommand;
            }

            /// <summary>
            /// Username.
            /// </summary>
            public string User { get; }

            /// <summary>
            /// IP of the server (/local computer-name)
            /// </summary>
            public string ServerIP { get; }

            /// <summary>
            /// Alias to start openFOAM-Environment on the server.
            /// </summary>
            public string OfAlias { get; }

            /// <summary>
            /// Folder on server openfoam case will be copied to.
            /// </summary>
            public string ServerCaseFolder { get; }

            /// <summary>
            /// Threads used.
            /// </summary>
            public string SlurmCommand { get; }

            /// <summary>
            /// Port server.
            /// </summary>
            public int Port { get; }

            /// <summary>
            /// Download after simulation.
            /// </summary>
            public bool Download { get; }

            /// <summary>
            /// Delete after simulation.
            /// </summary>
            public bool Delete { get; }

            /// <summary>
            /// Use slurm.
            /// </summary>
            public bool Slurm { get; }

            /// <summary>
            /// Connection string.
            /// </summary>
            /// <returns>user + @ + serverIP as string.</returns>
            public string ConnectionString() => User + "@" + ServerIP;
        }

        /// <summary>
        /// Struct for intializing Settings variables.
        /// </summary>
        readonly public struct InitialSettingsParameter
        {
            private static InitialSettingsParameter def = new InitialSettingsParameter();
            public static ref readonly InitialSettingsParameter Default => ref def;
            public InitialSettingsParameter() : this(SaveFormat.ascii, ElementsExportRange.OnlyVisibleOnes, ControlDictParameters.Default, true, false, false) { }
            public InitialSettingsParameter(in SaveFormat format, in ElementsExportRange export, in ControlDictParameters control, bool includeLinkedModels, bool exportColor, bool exportSharedCoordinates)
            {
                this.Format = format;
                this.ExportRange = export;
                this.ControlDict = control;
                this.IncludeLinkedModels = includeLinkedModels;
                this.ExportColor = exportColor;
                this.ExportSharedCoordinates = exportSharedCoordinates;
            }

            /// <summary>
            /// Struct ControlDictParamertes.
            /// </summary>
            public ControlDictParameters ControlDict { get; }

            /// <summary>
            /// SaveFormat enum
            /// </summary>
            public SaveFormat Format { get; }

            /// <summary>
            /// ExportRange enum.
            /// </summary>
            public ElementsExportRange ExportRange { get; }

            /// <summary>
            /// IncludedLinkedModels.
            /// </summary>
            public bool IncludeLinkedModels { get; }

            /// <summary>
            /// ExportColor enum.
            /// </summary>
            public bool ExportColor { get; }

            /// <summary>
            /// ExportSharedCoordinater for STL.
            /// </summary>
            public bool ExportSharedCoordinates { get; }

            public readonly override string ToString() => $"{ControlDict.ToString()}, {Format}, {ExportRange}, {IncludeLinkedModels}, {ExportColor}, {ExportSharedCoordinates}";
        }
    }

    namespace FOAM
    {
        using Model;

        /// <summary>
        /// Struct for controldict entries.
        /// </summary>
        public struct ControlDictParameters
        {
            private static ControlDictParameters def = new ControlDictParameters();
            public static ref readonly ControlDictParameters Default => ref def;

            /// <summary>
            /// Default constructor ControlDictParameters.
            /// </summary>
            public ControlDictParameters() : this(
                SolverControlDict.buoyantBoussinesqSimpleFoam,
                StartFrom.latestTime,
                StopAt.endTime,
                WriteControl.timeStep,
                WriteFormat.ascii,
                WriteCompression.off,
                TimeFormat.general, false, 0, 101, 1, 100, 2, 8, 7, 4)
            { }

            public ControlDictParameters(in SolverControlDict solver, in StartFrom from, in StopAt at,
                                         in WriteControl writeControl, in WriteFormat writeFormat, in WriteCompression writeCompression,
                                         in TimeFormat time, bool runTimeModifiable, double start, double end,
                                         double deltaT, double writeInterval, double purgeWrite,
                                         double writePrecision, double timePrecision, int numberOfSubdomains)
            {
                _appControlDictSolver = solver;
                _writeFromat = writeFormat;
                _endTime = end;
                _writeInterval = writeInterval;
                _numberOfSubdomains = numberOfSubdomains;
                this.StartFrom = from;
                this.StopAt = at;
                this.WriteControl = writeControl;
                this.WriteCompression = writeCompression;
                this.TimeFormat = time;
                this.RunTimeModifiable = runTimeModifiable;
                this.StartTime = start;
                this.DeltaT = deltaT;
                this.PurgeWrite = purgeWrite;
                this.WritePrecision = writePrecision;
                this.TimePrecision = timePrecision;
            }

            private SolverControlDict _appControlDictSolver;
            public SolverControlDict AppControlDictSolver { readonly get => _appControlDictSolver; set => _appControlDictSolver = value; }

            private WriteFormat _writeFromat;
            /// <summary>
            /// ASCII or Binary.
            /// </summary>
            public WriteFormat WriteFormat { readonly get => _writeFromat; set => _writeFromat = value; }

            private double _endTime;
            /// <summary>
            /// End time for ControlDict.
            /// </summary>
            public double EndTime { readonly get => _endTime; set => _endTime = value; }

            private double _writeInterval;
            /// <summary>
            /// WriterInterval for ControlDict.
            /// </summary>
            public double WriteInterval { readonly get => _writeInterval; set => _writeInterval = value; }

            private int _numberOfSubdomains;
            /// <summary>
            /// Number of CPU's
            /// </summary>
            public int NumberOfSubdomains { readonly get => _numberOfSubdomains; set => _numberOfSubdomains = value; }

            /// <summary>
            /// Where to start from after rerun of simulation.
            /// </summary>
            public readonly StartFrom StartFrom { get; }

            /// <summary>
            /// Condition for stop.
            /// </summary>
            public readonly StopAt StopAt { get; }

            /// <summary>
            /// Specify control scheme.
            /// </summary>
            public readonly WriteControl WriteControl { get; }

            /// <summary>
            /// Compression on or off. 
            /// </summary>
            public readonly WriteCompression WriteCompression { get; }

            /// <summary>
            /// Formate of timesteps.
            /// </summary>
            public readonly TimeFormat TimeFormat { get; }

            /// <summary>
            /// Bool for ControlDict.
            /// </summary>
            public readonly bool RunTimeModifiable { get; }

            /// <summary>
            /// Start time for ControlDict.
            /// </summary>
            public readonly double StartTime { get; }

            /// <summary>
            /// DeltaT for ControlDict.
            /// </summary>
            public readonly double DeltaT { get; }

            /// <summary>
            /// PurgeWrite for ControlDict.
            /// </summary>
            public readonly double PurgeWrite { get; }

            /// <summary>
            /// WritePrecision for ControlDict.
            /// </summary>
            public readonly double WritePrecision { get; }

            /// <summary>
            /// TimePrecision for ControlDict.
            /// </summary>
            public readonly double TimePrecision { get; }

            public readonly override string ToString() => $"{RunTimeModifiable}, {StartTime}, {EndTime}, {DeltaT}, {WriteInterval}, {PurgeWrite}, {WritePrecision}, {TimePrecision}, {NumberOfSubdomains}";
        }

        /// <summary>
        /// K-Epsilon turbulence model datatype.
        /// </summary>
        public struct KEpsilon
        {
            public KEpsilon() : this(0, 0) { }
            public KEpsilon(double k, double epsilon)
            {
                _k = k;
                _epsilon = epsilon;
            }

            /// <summary>
            ///  Contructor which calculates via CalculateKEpsilon.
            /// </summary>
            /// <param name="area">Area of inlet surface.</param>
            /// <param name="boundary">Boundary of inlet surface.</param>
            /// <param name="meanFlowVelocity">Mean flow velocity through inlet.</param>
            /// <param name="tempInlet"> CUrrent temp on inlet.</param>
            public KEpsilon(in double area, in double boundary, in double meanFlowVelocity, in double tempInlet) : this(0, 0) => CalculateKEpsilon(area, boundary, meanFlowVelocity, tempInlet);

            private double _k;
            /// <summary>
            /// Turbulence energie.
            /// </summary>
            public double K { readonly get => _k; set => _k = value; }

            private double _epsilon;
            /// <summary>
            /// Dissipation rate.
            /// </summary>
            public double Epsilon { readonly get => _epsilon; set => _epsilon = value; }

            /// <summary>
            /// Calculate k and epsilon with OpenFOAMCalculator-class.
            /// </summary>
            /// <param name="area">Area of inlet surface.</param>
            /// <param name="boundary">Boundary of inlet surface.</param>
            /// <param name="meanFlowVelocity">Mean flow velocity through inlet.</param>
            /// <param name="temp"> Current temp.</param>
            private void CalculateKEpsilon(in double area, in double boundary, in double meanFlowVelocity, in double temp)
            {
                OpenFOAM.OpenFOAMCalculator calculator = new();

                double kinematicViscosity = calculator.InterpolateKinematicViscosity(temp - 273.15);
                double characteristicLength = calculator.CalculateHydraulicDiameter(area, boundary);
                double reynoldsNumber = calculator.CalculateReynoldsnumber(Math.Abs(meanFlowVelocity), kinematicViscosity, characteristicLength);
                double turbulenceLengthScale = calculator.EstimateTurbulencLengthScalePipe(characteristicLength);
                double turbulenceIntensity = calculator.EstimateTurbulenceIntensityPipe(reynoldsNumber);

                if (meanFlowVelocity != 0)
                {
                    _k = calculator.CalculateK(Math.Abs(meanFlowVelocity), turbulenceIntensity);
                    _epsilon = calculator.CalculateEpsilon(turbulenceLengthScale, 0);
                }
            }
        }

        /// <summary>
        /// Represents an initial parameter from the null folder.
        /// </summary>
        public struct NullParameter
        {
            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="_name">Name for parameter.</param>
            /// <param name="_internalField">Value for internalfield.</param>
            /// <param name="_turbulenceModel">Turbulence-model.</param>
            /// <param name="_solverInc">Incompressible-Solver</param>
            /// <param name="_stype">Turbulence-type.</param>
            public NullParameter(in string name, in dynamic internalField, in dynamic turbulenceModel,
                in SolverControlDict solverInc = SolverControlDict.simpleFoam, in SimulationType stype = SimulationType.RAS)
            {
                this.Name = name;
                this.InternalField = internalField;
                this.TurbulenceModel = turbulenceModel;
                this.Solver = solverInc;
                this.SimulationType = stype;
                _patches = new Dictionary<string, FOAMParameterPatch<dynamic>>();
            }

            /// <summary>
            /// Name of Parameter.
            /// </summary>
            public readonly string Name { get; }

            /// <summary>
            /// Value of internalField.
            /// </summary>
            public readonly dynamic InternalField { get; }

            private Dictionary<string, FOAMParameterPatch<dynamic>> _patches;
            /// <summary>
            /// List of inlet-, outlet-, wall-patches.
            /// </summary>
            public Dictionary<string, FOAMParameterPatch<dynamic>> Patches { readonly get => _patches; set => _patches = value; }

            /// <summary>
            /// Solver for incompressible CFD.
            /// </summary>
            public readonly SolverControlDict Solver { get; }

            // private SimulationType _simulationType;
            /// <summary>
            /// Turbulence simulationType.
            /// </summary>
            // public SimulationType SimulationType { readonly get => _simulationType; set => _simulationType = value; }
            public readonly SimulationType SimulationType { get; }

            /// <summary>
            /// Turbulence-model.
            /// </summary>
            public readonly dynamic TurbulenceModel { get; }
        }

        /// <summary>
        /// Patch for boundaryField in Parameter-Dictionaries.
        /// </summary>
        /// <typeparam name="T">Type for value.</typeparam>
        readonly public struct FOAMParameterPatch<T>
        {
            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="_type">Type of Patch</param>
            /// <param name="_uniform">uniform or nonuniform.</param>
            /// <param name="_value">Vector3D or double.</param>
            /// <param name="_patchType">Enum pathtype. </param>
            public FOAMParameterPatch(in string _type, in string _uniform, in T _value, in PatchType _patchType)
            {
                this.Type = _patchType;
                if (!_value.Equals(default) && !_uniform.Equals(""))
                {
                    this.Attributes = new Dictionary<string, object>
                    {
                        { "type", _type },
                        { "value " + _uniform, _value}
                    };
                }
                else
                {
                    this.Attributes = new Dictionary<string, object>
                    {
                        { "type", _type }
                    };
                }
            }

            /// <summary>
            /// Getter-Method for patchType.
            /// </summary>
            public PatchType Type { get; }

            /// <summary>
            /// Getter for Attributes
            /// </summary>
            public Dictionary<string, object> Attributes { get; }
        }

        /// <summary>
        /// Coeffs-Parameter for DecomposeParDict.
        /// </summary>
        public struct CoeffsMethod
        {
            //Attributes
            /// <summary>
            /// Distribution n-Vector in DecomposeParDict.
            /// </summary>
            Vector3D n;

            /// <summary>
            /// Delta of DecomposeParDict.
            /// </summary>
            double delta;

            /// <summary>
            /// Getter for n-Vector.
            /// </summary>
            public Vector3D N { get => n; }

            /// <summary>
            /// Getter for Delta.
            /// </summary>
            public double Delta { get => delta; set => delta = value; }

            /// <summary>
            /// Initialize Vector N with the number of cpu's.
            /// </summary>
            /// <param name="numberOfSubdomains">Number of physical CPU's.</param>
            public void SetN(int numberOfSubdomains)
            {
                //Algo for subDomains
            }

            /// <summary>
            /// Initialize Vector N with given Vecotr _n.
            /// </summary>
            /// <param name="_n">Explicit Vector for N.</param>
            public void SetN(Vector3D _n)
            {
                n = _n;
            }

            /// <summary>
            /// Creates Dictionary and adds attributes to it.
            /// </summary>
            /// <returns>Dictionary filled with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> attributes = new Dictionary<string, object>();
                attributes.Add("n", n);
                attributes.Add("delta", delta);
                return attributes;
            }
        }
        /// <summary>
        /// P-FvSolution.
        /// </summary>
        public struct PFv
        {
            //Parameter for the p-Dictionary in FvSolutionDictionary
            /// <summary>
            /// Parameter for the p-Dicitonionary in FvSolutionDicitonary.
            /// </summary>
            FvSolutionParameter param;

            /// <summary>
            /// Agglomerator-Enum.
            /// </summary>
            Agglomerator agglomerator;

            /// <summary>
            /// CachAgglomeration-Enum.
            /// </summary>
            CacheAgglomeration cacheAgglomeration;

            /// <summary>
            /// Interger for nCellsInCoarsesLevel.
            /// </summary>
            int nCellsInCoarsesLevel;

            /// <summary>
            /// Integer for nPostSweeps.
            /// </summary>
            int nPostSweeps;

            /// <summary>
            /// Integer for nPreSweepsre.
            /// </summary>
            int nPreSweepsre;

            /// <summary>
            /// Integer for mergeLevels.
            /// </summary>
            int mergeLevels;

            //Getter-Setter
            public FvSolutionParameter Param { get => param; set => param = value; }
            public Agglomerator Agglomerator { get => agglomerator; set => agglomerator = value; }
            public CacheAgglomeration CacheAgglomeration { get => cacheAgglomeration; set => cacheAgglomeration = value; }
            public int NCellsInCoarsesLevel { get => nCellsInCoarsesLevel; set => nCellsInCoarsesLevel = value; }
            public int NPostSweeps { get => nPostSweeps; set => nPostSweeps = value; }
            public int NPreSweepsre { get => nPreSweepsre; set => nPreSweepsre = value; }
            public int MergeLevels { get => mergeLevels; set => mergeLevels = value; }

            /// <summary>
            /// Creates a Dictionary of data.
            /// </summary>
            /// <returns>Dictionary filled with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> pList = new Dictionary<string, object>
                {
                    {"agglomerator" , Agglomerator},
                    {"relTol" , Param.RelTol },
                    {"tolerance" , Param.Tolerance },
                    {"nCellsInCoarsesLevel", NCellsInCoarsesLevel },
                    {"smoother" , Param.Smoother },
                    {"solver" , Param.Solver },
                    {"cacheAgglomeration" , CacheAgglomeration },
                    {"nPostSweeps" , NPostSweeps },
                    {"nPreSweepsre" , NPreSweepsre },
                    {"mergeLevels", MergeLevels }
                };
                return pList;
            }
        }

        /// <summary>
        /// Fv-SolutionParam
        /// </summary>
        public struct FvSolutionParameter
        {
            public FvSolutionParameter(in Smoother smoother, in SolverFV solver, in Preconditioner precond, in double relTol, in double tolerance, in int nSweeps)
            {
                _smoother = smoother;
                _solver = solver;
                _preconditioner = precond;
                _relTol = relTol;
                _tolerance = tolerance;
                _nSweeps = nSweeps;
            }

            /// <summary>
            /// Smoother-type.
            /// </summary>
            private Smoother _smoother;
            public Smoother Smoother { readonly get => _smoother; set => _smoother = value; }

            /// <summary>
            /// Solver for FvSolutionDict.
            /// </summary>
            private SolverFV _solver;
            public SolverFV Solver { readonly get => _solver; set => _solver = value; }

            /// <summary>
            /// Double for relTol in FvSolutionDict.
            /// </summary>
            private double _relTol;
            public double RelTol { readonly get => _relTol; set => _relTol = value; }

            /// <summary>
            /// Double for tolerance in FvSolutionDict.
            /// </summary>
            private double _tolerance;
            public double Tolerance { readonly get => _tolerance; set => _tolerance = value; }

            /// <summary>
            /// Double for nSweeps in FvSolutionDict.
            /// </summary>
            private int _nSweeps;
            public int NSweeps { readonly get => _nSweeps; set => _nSweeps = value; }

            /// <summary>
            /// Manipulates the matrix equation (AP^(-1))*Px=b to solve it more readily.
            /// </summary>
            private Preconditioner _preconditioner;
            public Preconditioner Preconditioner { readonly get => _preconditioner; set => _preconditioner = value; }

            /// <summary>
            /// Creates a Dictionary of data.
            /// </summary>
            /// <returns>Dictionary filled with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> paramList = new Dictionary<string, object>
                {
                    {"relTol" , RelTol },
                    {"tolerance" , Tolerance },
                    {"nSweeps" , NSweeps},
                    {"smoother" , Smoother },
                    {"solver" , Solver },
                    {"preconditioner", Preconditioner }

                };
                return paramList;
            }
        }

        /// <summary>
        /// Turbulence attributes for the openfoam dictionary turbulenceProperties.
        /// </summary>
        readonly public struct TurbulenceParameter
        {
            /// <summary>
            /// Constructor.
            /// </summary>
            /// <param name="simType">Simulationtype enum.</param>
            /// <param name="simModel">Simulation model enum.</param>
            /// <param name="turbulence">true = on, false = off</param>
            /// <param name="printCoeff">true = on, false = off</param>
            public TurbulenceParameter(SimulationType simType, Enum simModel, bool turbulence = true, bool printCoeff = true)
            {
                this.SimType = simType;
                this.StructModel = null;
                switch (simType)
                {
                    case SimulationType.RAS:
                        this.StructModel = new RAS((RASModel)simModel, turbulence, printCoeff);
                        break;
                    case SimulationType.LES:
                        this.StructModel = new LES((LESModel)simModel);
                        //TO-DO: Implement.
                        break;
                    case SimulationType.laminar:
                        //TO-DO: Implement.
                        break;
                    default:
                        break;
                }
            }

            /// <summary>
            /// Type of simulation.
            /// </summary>
            public SimulationType SimType { get; }

            /// <summary>
            /// Model for simulation
            /// </summary>
            public ValueType StructModel { get; }

            /// <summary>
            /// This methode creates and returns the attributes as dictionary<string, object>.
            /// </summary>
            /// <returns>Dictionary with attributes.</returns>
            public Dictionary<string, object> ToDictionary()
            {
                Dictionary<string, object> dict = new Dictionary<string, object>
                {
                    { "simulationType", SimType }
                };
                switch (SimType)
                {
                    case SimulationType.RAS:
                        dict.Add(SimType.ToString(), ((RAS)StructModel).ToDictionary());
                        break;
                    case SimulationType.LES:
                        //TO-DO: Implement LES.
                        break;
                    case SimulationType.laminar:
                        //TO-DO: Implement Laminar.
                        break;

                }
                return dict;
            }
        }

        namespace Model
        {
            /// <summary>
            /// RAS-Model attributes in turbulenceProperties.
            /// </summary>
            readonly public struct RAS
            {
                //internal enum for on and off
                public enum OnOff
                {
                    on = 0,
                    off
                }

                /// <summary>
                /// Constructor.
                /// </summary>
                /// <param name="model">Enum-Object for simulation model.</param>
                /// <param name="turb">turbulence true = on, false = off</param>
                /// <param name="printCoeff">printCoeef true = on, false = off</param>
                public RAS(RASModel model, bool turb, bool printCoeff)
                {
                    this.RASModel = model;
                    this.Turbulence = turb ? OnOff.on : OnOff.off;
                    this.PrintCoeffs = printCoeff ? OnOff.on : OnOff.off;
                }

                ///<summary>
                /// Enum for model name
                ///</summary> 
                public RASModel RASModel { get; }

                ///<summary>
                /// turbulence on or off
                ///</summary> 
                public OnOff Turbulence { get; }

                ///<summary>
                /// print coefficient on or off
                ///</summary> 
                public OnOff PrintCoeffs { get; }

                /// <summary>
                /// Returns all attributes as Dictionary<string,object>
                /// </summary>
                /// <returns>Dictionary filled with attributes.</returns>
                public Dictionary<string, object> ToDictionary()
                {
                    return new Dictionary<string, object>
                    {
                        { "RASModel", RASModel },
                        { "turbulence", Turbulence },
                        { "printCoeffs", PrintCoeffs}
                    };
                }
            }

            /// <summary>
            /// Simulationmodel LES-Parameter.
            /// </summary>
            readonly public struct LES
            {
                public LES(LESModel lESModel)
                {
                    this.LESModel = lESModel;
                }
                public LESModel LESModel { get; }
                //TO-DO: implement LES
            }

            /// <summary>
            /// Simulationmodel Laminar-Parameter.
            /// </summary>
            public struct Laminar
            {
                //SimulationType typeL;
                //TO-DO: Implement laminar
            }
        }
    }

    namespace Revit
    {
        /// <summary>
        /// Properties of a duct terminal.
        /// </summary>
        readonly public struct DuctProperties
        {
            public DuctProperties(in XYZ faceNormal, int rpm, in double externalPressure, in double area, in double boundary, in double flowRate, in double meanFlowVelocity, in double temp)
            {
                this.FaceNormal = faceNormal;
                this.RPM = rpm;
                this.ExternalPressure = externalPressure;
                this.Area = area;
                this.Boundary = boundary;
                this.FlowRate = flowRate;
                this.MeanFlowVelocity = meanFlowVelocity;
                this.Temperature = temp;
            }

            /// <summary>
            /// RPM for swirl diffuser.
            /// </summary>
            public int RPM { get; }

            /// <summary>
            /// External pressure.
            /// </summary>
            public double ExternalPressure { get; }

            /// <summary>
            /// Area of the surface.
            /// </summary>
            public double Area { get; }

            /// <summary>
            /// Boundary of the surface.
            /// </summary>
            public double Boundary { get; }

            /// <summary>
            /// Air flow rate in m�/s.
            /// </summary>
            public double FlowRate { get; }

            /// <summary>
            /// Mean flow velocity through surface.
            /// </summary>
            public double MeanFlowVelocity { get; }

            /// <summary>
            /// Face normal of the surface.
            /// </summary>
            public XYZ FaceNormal { get; }
            /// <summary>
            /// Flow Temperature.
            /// </summary>
            public double Temperature { get; }
        }
    }
}