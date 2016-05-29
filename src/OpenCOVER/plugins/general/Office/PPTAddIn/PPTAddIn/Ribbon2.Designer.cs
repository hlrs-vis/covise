namespace PPTAddIn
{
    partial class Ribbon2 : Microsoft.Office.Tools.Ribbon.RibbonBase
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        public Ribbon2()
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
            this.group1 = this.Factory.CreateRibbonGroup();
            this.host = this.Factory.CreateRibbonEditBox();
            this.port = this.Factory.CreateRibbonEditBox();
            this.connectToCAVE = this.Factory.CreateRibbonButton();
            this.connected = this.Factory.CreateRibbonCheckBox();
            this.tab1.SuspendLayout();
            this.group1.SuspendLayout();
            // 
            // tab1
            // 
            this.tab1.ControlId.ControlIdType = Microsoft.Office.Tools.Ribbon.RibbonControlIdType.Office;
            this.tab1.Groups.Add(this.group1);
            this.tab1.Label = "OpenCOVER";
            this.tab1.Name = "tab1";
            // 
            // group1
            // 
            this.group1.Items.Add(this.host);
            this.group1.Items.Add(this.port);
            this.group1.Items.Add(this.connectToCAVE);
            this.group1.Items.Add(this.connected);
            this.group1.Label = "connection";
            this.group1.Name = "group1";
            // 
            // host
            // 
            this.host.Label = "host";
            this.host.Name = "host";
            this.host.Text = "localhost";
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
            this.connectToCAVE.Click += new Microsoft.Office.Tools.Ribbon.RibbonControlEventHandler(this.connectToCAVE_Click);
            // 
            // connected
            // 
            this.connected.Label = "connected";
            this.connected.Name = "connected";
            // 
            // Ribbon2
            // 
            this.Name = "Ribbon2";
            this.RibbonType = "Microsoft.PowerPoint.Presentation";
            this.Tabs.Add(this.tab1);
            this.Load += new Microsoft.Office.Tools.Ribbon.RibbonUIEventHandler(this.Ribbon2_Load);
            this.tab1.ResumeLayout(false);
            this.tab1.PerformLayout();
            this.group1.ResumeLayout(false);
            this.group1.PerformLayout();

        }

        #endregion

        internal Microsoft.Office.Tools.Ribbon.RibbonTab tab1;
        internal Microsoft.Office.Tools.Ribbon.RibbonGroup group1;
        internal Microsoft.Office.Tools.Ribbon.RibbonEditBox host;
        internal Microsoft.Office.Tools.Ribbon.RibbonEditBox port;
        internal Microsoft.Office.Tools.Ribbon.RibbonButton connectToCAVE;
        internal Microsoft.Office.Tools.Ribbon.RibbonCheckBox connected;
    }

    partial class ThisRibbonCollection
    {
        internal Ribbon2 Ribbon2
        {
            get { return this.GetRibbon<Ribbon2>(); }
        }
    }
}
