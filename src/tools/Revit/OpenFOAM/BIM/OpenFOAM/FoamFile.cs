/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
using System;
using System.Windows.Media.Media3D;
using System.IO;
using System.Collections.Generic;
using System.Windows.Forms;
using System.Security;
using System.Collections;
using System.Windows;

namespace BIM.OpenFOAMExport.OpenFOAM
{
    /// <summary>
    /// Interface for FoamFiles.
    /// </summary>
    public abstract class FOAMFile : IDisposable
    {
        /// <summary>
        /// Name of the file
        /// </summary>
        protected string m_Name = string.Empty;

        /// <summary>
        /// Name of the folder this file is included
        /// </summary>
        protected string m_Location = string.Empty;

        /// <summary>
        /// Path to this file
        /// </summary>
        protected string m_Path = string.Empty;

        /// <summary>
        /// Class of this file for the Header
        /// </summary>
        protected string m_Class = string.Empty;

        /// <summary>
        /// List of attributes for the OpenFOAM-File orderd.
        /// </summary>
        protected Dictionary<string, object> m_Attributes;

        /// <summary>
        /// Default entries for dictionary FoamFile
        /// </summary>
        protected Dictionary<string, object> m_DefaultAttributes;

        /// <summary>
        /// Version of OpenFoam and Addin
        /// </summary>
        protected Version m_Version;

        /// <summary>
        /// file format: Binary or AscII
        /// </summary>
        protected SaveFormat m_SaveFormat = SaveFormat.ascii;

        /// <summary>
        ///Attributes of the File.
        /// </summary>
        public Dictionary<string, object> Attributes
        {
            get
            {
                return m_Attributes;
            }
        }

        /// <summary>
        /// Returns folder name this file is stored in.
        /// </summary>
        public string Location
        {
            get
            {
                return m_Location;
            }
        }

        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="name">Name of the file.</param>
        /// <param name="version">Version</param>
        /// <param name="path">Path to the foamfile.</param>
        /// <param name="attributes">Additional attributes besides default.</param>
        /// <param name="format">Ascii or Binary.</param>
        public FOAMFile(string name, Version version, string path, string _class, Dictionary<string, object> attributes, SaveFormat format)
        {
            m_Name = name;
            m_Path = path;

            if(_class == string.Empty)
            {
                m_Class = "dictionary";
            }
            else
            {
                m_Class = _class;
            }

            m_SaveFormat = format;
            m_Version = version;
            m_Attributes = new Dictionary<string, object>();
            m_DefaultAttributes = new Dictionary<string, object>();

            Init();

            if (attributes != null)
            {
                foreach (var attribute in attributes)
                {
                    m_Attributes.Add(attribute.Key, attribute.Value);
                }
            }
        }

        /// <summary>
        /// Initialize m_DefaultAttributes and create the file.
        /// </summary>
        public virtual void Init()
        {
            InitLocation();
            m_DefaultAttributes = new Dictionary<string, object> {
                { "version", m_Version.OFVer} ,
                { "format", m_SaveFormat.ToString()} ,
                { "class",  m_Class} ,
                { "location", m_Location} ,
                { "object", m_Name}
            };
            Attributes.Add("FoamFile", m_DefaultAttributes);
            CreateFile();
        }

        /// <summary>
        /// Init location attribute from path.
        /// </summary>
        public virtual void InitLocation()
        {
            if (m_Path.Contains("system"))
            {
                m_Location = "\"system\"";
            }
            else if (m_Path.Contains("constant"))
            {
                m_Location = "\"constant\"";
            }
            else if (m_Path.Contains("0"))
            {
                m_Location = "\"0\"";
            }
        }

        /// <summary>
        /// Implement interface of creating a file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public abstract bool CreateFile();

        /// <summary>
        /// Write all attributes of the Foamfile into it and close the file.
        /// </summary>
        public virtual void WriteFile()
        {
            WriteHeader();
            CreateDict(Attributes);
            CloseFile();
        }

        /// <summary>
        /// Implement interface for writing the header:
        /// </summary>
        public abstract void WriteHeader();

        /// <summary>
        /// Implement interface for write string into file.
        /// </summary>
        /// <param name="attribute">Attribute as string.</param>
        public abstract void WriteInFile(string attribute);

        /// <summary>
        /// Implement interface for iterating through dictionaries.
        /// </summary>
        /// <param name="dict">Contains all attributes.</param>
        public abstract void CreateDict(Dictionary<string, object> dict);

        /// <summary>
        /// Implement interface for iterating through a list.
        /// </summary>
        /// <param name="list">Contains Vector3D, double[] or Dictionaries<string,object></param>
        public abstract void CreateList(ArrayList list);

        /// <summary>
        /// Implement interface for creating a attribute of an KeyValuePair and write it to the File.
        /// </summary>
        /// <param name="attribute">Attribute of the Foamfile</param>
        public abstract bool CreateAttribute(KeyValuePair<string, object> attribute, Type type);

        /// <summary>
        /// Close file here if user forget it.
        /// </summary>
        public virtual void Dispose()
        {
            CloseFile();
        }

        /// <summary>
        /// Close file handle.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public abstract bool CloseFile();
    }

    /// <summary>
    /// FoamFile as Binary.
    /// </summary>
    public class FoamFileAsBinary : FOAMFile
    {
        //
       // private readonly BinaryWriter binaryWriter = null;

        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="name">Name of the file.</param>
        /// <param name="version">Version</param>
        /// <param name="path">Path to the foamfile.</param>
        /// <param name="attributes">Additional attributes besides default.</param>
        /// <param name="format">Ascii or Binary.</param>
        public FoamFileAsBinary(string name, Version version, string path, string _class, Dictionary<string, object> attributes, SaveFormat format)
            : base(name, version, path, _class, attributes, format)
        {

        }

        ///// <summary>
        ///// 
        ///// </summary>
        //public override void Init()
        //{
        //    throw new NotImplementedException();
        //}

        ///// <summary>
        ///// 
        ///// </summary>
        //public override void InitLocation()
        //{
        //    throw new NotImplementedException();
        //}

        /// <summary>
        /// Implement interface of creating a file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool CreateFile()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Write all attributes of the Foamfile into it and close the file.
        /// </summary>
        public override void WriteFile()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Writes the Header that is set in the version-object to the foamfile.
        /// </summary>
        public override void WriteHeader()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Write given string to foamfile.
        /// </summary>
        /// <param name="attribute">Attribute as string.</param>
        public override void WriteInFile(string attribute)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Iterates recursive through a dictionary and create inputs on depending entries in it.
        /// </summary>
        /// <param name="dict">Contains all attributes.</param>
        public override void CreateDict(Dictionary<string, object> dict)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Create a list entry in the foamfile.
        /// </summary>
        /// <param name="list">Contains Vector3D, double[] or Dictionaries<string,object></param>
        public override void CreateList(ArrayList list)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Create attribute from given KeyValuePair and write it to foamfile.
        /// </summary>
        /// <param name="attribute">Attribute of the Foamfile</param>
        public override bool CreateAttribute(KeyValuePair<string,object> attribute, Type type)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Implement interface of closing the file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool CloseFile()
        {
            bool succeed = true;

            return succeed;
        }
    }

    /// <summary>
    /// FoamFile as AscII.
    /// </summary>
    public class FoamFileAsAscII : FOAMFile
    {
        //foamfile
        StreamWriter foamFile = null;
        //number of tabs inside FoamFile
        int tabs = 0;

        /// <summary>
        /// Contructor.
        /// </summary>
        /// <param name="name">Name of the file.</param>
        /// <param name="version">Version</param>
        /// <param name="path">Path to the foamfile.</param>
        /// <param name="attributes">Additional attributes besides default.</param>
        /// <param name="format">Ascii or Binary.</param>
        public FoamFileAsAscII(string name, Version version, string path, string _class, Dictionary<string, object> attributes, SaveFormat format)
            : base(name, version, path, _class, attributes, format)
        {
        }

        /// <summary>
        /// Implement interface of creating a file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool CreateFile()
        {
            bool succeed = true;
            try
            {
                FileAttributes fileAttribute = FileAttributes.Normal;

                if (File.Exists(m_Path))
                {
                    fileAttribute = File.GetAttributes(m_Path);
                    FileAttributes tempAtt = fileAttribute & FileAttributes.ReadOnly;
                    if (FileAttributes.ReadOnly == tempAtt)
                    {
                        System.Windows.Forms.MessageBox.Show(OpenFOAMExportResource.ERR_FILE_READONLY, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                              MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                        return false;
                    }
                    File.Delete(m_Path);
                }

                foamFile = new StreamWriter(m_Path);
                foamFile.NewLine = "\n";
                fileAttribute = File.GetAttributes(m_Path) | fileAttribute;
                File.SetAttributes(m_Path, fileAttribute);
            }
            catch (SecurityException)
            {
                System.Windows.Forms.MessageBox.Show(OpenFOAMExportResource.ERR_SECURITY_EXCEPTION, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                succeed = false;
            }
            catch (IOException)
            {
                System.Windows.Forms.MessageBox.Show(OpenFOAMExportResource.ERR_IO_EXCEPTION, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                succeed = false;
            }
            catch (Exception)
            {
                System.Windows.Forms.MessageBox.Show(OpenFOAMExportResource.ERR_EXCEPTION, OpenFOAMExportResource.MESSAGE_BOX_TITLE,
                            MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                succeed = false;
            }
            return succeed;
        }

        /// <summary>
        /// Writes the Header that is set in the version-object to the foamfile.
        /// </summary>
        public override void WriteHeader()
        {
            WriteInFile(m_Version.Header.HeaderStr);
        }

        /// <summary>
        /// Write given string to foamfile.
        /// </summary>
        /// <param name="attribute">Attribute as string.</param>
        public override void WriteInFile(string attribute)
        {
            if (attribute.Equals("}") || attribute.Equals(");"))
            {
                tabs--;
            }
            for (int i = 0; i < tabs; i++)
            {
                foamFile.Write("\t");
            }
            foamFile.WriteLine(attribute);
            if (attribute.Equals("{") || attribute.Equals("("))
            {
                tabs++;
            }
        }

        /// <summary>
        /// Iterates recursive through a dictionary and create inputs on depending entries in it.
        /// </summary>
        /// <param name="dict">Contains all attributes.</param>
        public override void CreateDict(Dictionary<string, object> dict)
        {
            foreach (var attribute in dict)
            {
                Type type = dict[attribute.Key].GetType();
                if (type == null)
                {
                    continue;
                }
                else if (typeof(Dictionary<string, object>) == type)
                {
                    WriteInFile(attribute.Key);
                    WriteInFile("{");
                    WriteInFile("");
                    Dictionary<string, object> newDictLevel = dict[attribute.Key] as Dictionary<string, object>;
                    CreateDict(newDictLevel);
                    WriteInFile("");
                    WriteInFile("}");
                    WriteInFile("");
                    continue;
                }
                else if (typeof(ArrayList) == type)
                {
                    WriteInFile(attribute.Key);
                    WriteInFile("(");
                    WriteInFile("");
                    ArrayList newListLevel = dict[attribute.Key] as ArrayList;
                    CreateList(newListLevel);
                    WriteInFile("");
                    WriteInFile(");");
                    WriteInFile("");
                    continue;
                }
                CreateAttribute(attribute, type);
                //WriteInFile("");
            }
        }

        /// <summary>
        /// Create a list entry in the foamfile.
        /// </summary>
        /// <param name="list">Contains Vector3D, double[], int[] or Dictionaries<string,object></param>
        public override void CreateList(ArrayList list)
        {
            foreach (var obj in list)
            {
                string entry = string.Empty;
                Type type = obj.GetType();

                if(type == typeof(Vector3D) || type  == typeof(Vector))
                {
                    entry = VectorToString(obj);
                }
                else if (type == typeof(double[]) || type == typeof(int[]))
                {
                    Array moreDimVector = null;
                    if(type == typeof(double[]))
                    {
                        moreDimVector = obj as double[];
                    }
                    else if(type == typeof(int[]))
                    {
                        moreDimVector = obj as int[];
                    }
                    entry = ArrayToString(moreDimVector);
                }
                else if (type == typeof(KeyValuePair<string, object>))
                {
                    KeyValuePair<string, object> kvp = (KeyValuePair<string, object>)obj;
                    WriteInFile(kvp.Key);
                    WriteInFile("{");
                    Dictionary<string, object> newDictLevel = kvp.Value as Dictionary<string, object>;
                    CreateDict(newDictLevel);
                    WriteInFile("}");
                    continue;
                }
                else if (type == typeof(string))
                {
                    entry = obj as string;
                    WriteInFile(entry);
                    continue;
                }
                WriteInFile("(" + entry + ")");
            }
        }

        /// <summary>
        /// Create attribute from given KeyValuePair and write it to foamfile.
        /// </summary>
        /// <param name="attribute">Attribute of the Foamfile</param>
        public override bool CreateAttribute(KeyValuePair<string,object> attribute, Type type)
        {
            //Type type = attribute.Value.GetType();
            string objectValue;
            if (typeof(Vector3D) == type || typeof(Vector) == type)
            {
                string vec = VectorToString(attribute.Value);
                string vecFoam = attribute.Key + " (" + vec + ");";
                WriteInFile(vecFoam);
                return true;
            }
            else if(typeof(int[]) == type)
            {
                var array = attribute.Value as int[];
                string entry = ArrayToString(array);
                string moreDimVector = attribute.Key + " [" + entry + "];";
                WriteInFile(moreDimVector);
                return true;
            }
            else if (typeof(double) == type)
            {
                double d = (double)attribute.Value;
                objectValue = d.ToString(System.Globalization.CultureInfo.GetCultureInfo("en-US").NumberFormat);
            }
            else
            {
                objectValue = attribute.Value.ToString();
                if(typeof(bool) == type)
                {
                    objectValue = objectValue.ToLower();
                }
            }
            WriteInFile(attribute.Key + "\t\t" + objectValue + ";");
            return true;
        }

        /// <summary>
        /// Return vector as string.
        /// </summary>
        /// <typeparam name="T">Type of give vector object.</typeparam>
        /// <param name="vec">Vector object.</param>
        /// <returns>Vector as string.</returns>
        private string VectorToString<T>(T vec)
        {
            string formatString = vec.ToString().Replace(";", " ");
            return formatString.Replace(",", ".");
        }

        /// <summary>
        /// Return array entries as string with space in between.
        /// </summary>
        /// <param name="array">Array-object.</param>
        /// <returns>Entries as string.</returns>
        private string ArrayToString(Array array)
        {
            string entry = string.Empty;
            for (int i = 0; i < array.Length; i++)
            {
                if (i == array.Length - 1)
                {
                    entry += array.GetValue(i);
                    break;
                }
                entry += array.GetValue(i) + " ";
            }
            return entry;
        }

        /// <summary>
        /// Implement interface of closing the file.
        /// </summary>
        /// <returns>True if succeeded, false if failed.</returns>
        public override bool CloseFile()
        {
            bool succeed = true;
            if (null != foamFile)
            {
                foamFile.Close();
                foamFile = null;
            }
            return succeed;
        }
    }
}