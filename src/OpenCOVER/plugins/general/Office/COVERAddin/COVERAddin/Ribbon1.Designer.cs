namespace COVERAddin
{
    partial class Ribbon1 : Microsoft.Office.Tools.Ribbon.RibbonBase
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        public Ribbon1()
            : base(Globals.Factory.GetRibbonFactory())
        {
            InitializeComponent();
        }

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.tab1 = this.Factory.CreateRibbonTab();
            this.COVERGroup = this.Factory.CreateRibbonGroup();
            this.host = this.Factory.CreateRibbonEditBox();
            this.port = this.Factory.CreateRibbonEditBox();
            this.connectToCAVE = this.Factory.CreateRibbonButton();
            this.connected = this.Factory.CreateRibbonCheckBox();
            this.tab1.SuspendLayout();
            this.COVERGroup.SuspendLayout();
            // 
            // tab1
            // 
            this.tab1.Groups.Add(this.COVERGroup);
            this.tab1.Label = "OpenCOVER";
            this.tab1.Name = "tab1";
            // 
            // COVERGroup
            // 
            this.COVERGroup.Items.Add(this.host);
            this.COVERGroup.Items.Add(this.port);
            this.COVERGroup.Items.Add(this.connectToCAVE);
            this.COVERGroup.Items.Add(this.connected);
            this.COVERGroup.Label = "Connection";
            this.COVERGroup.Name = "COVERGroup";
            // 
            // host
            // 
            this.host.Label = "host";
            this.host.Name = "host";
            this.host.Text = "localhost";
            this.host.TextChanged += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.host_TextChanged);
            // 
            // port
            // 
            this.port.Label = "port";
            this.port.Name = "port";
            this.port.Text = "31315";
            // 
            // connectToCAVE
            // 
            this.connectToCAVE.Label = "connectToCAVE";
            this.connectToCAVE.Name = "connectToCAVE";
            this.connectToCAVE.Click += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.port_Click);
            // 
            // connected
            // 
            this.connected.Label = "connected";
            this.connected.Name = "connected";
            this.connected.Click += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.checkBox1_Click);
            // 
            // Ribbon1
            // 
            this.Name = "Ribbon1";
            this.RibbonType = "Microsoft.Word.Document";
            this.Tabs.Add(this.tab1);
            this.Load += new Microsoft.Office.Tools.Ribbon.RibbonUIEventHandler(this.Ribbon1_Load);
            this.tab1.ResumeLayout(false);
            this.tab1.PerformLayout();
            this.COVERGroup.ResumeLayout(false);
            this.COVERGroup.PerformLayout();

        }

        #endregion

        internal Microsoft.Office.Tools.Ribbon.RibbonTab tab1;
        internal Microsoft.Office.Tools.Ribbon.RibbonGroup COVERGroup;
        internal Microsoft.Office.Tools.Ribbon.RibbonEditBox host;
        internal Microsoft.Office.Tools.Ribbon.RibbonEditBox port;
        internal Microsoft.Office.Tools.Ribbon.RibbonButton connectToCAVE;
        internal Microsoft.Office.Tools.Ribbon.RibbonCheckBox connected;
    }

    partial class ThisRibbonCollection
    {
        internal Ribbon1 Ribbon1
        {
            get { return this.GetRibbon<Ribbon1>(); }
        }
    }
}
