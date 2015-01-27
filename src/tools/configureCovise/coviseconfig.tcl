#For each possible section like COVERConfig TRACKERCOnfig and so on there
#is a list containing the needed widgets


#Each control needed for a certain section is stored
#in a list for this purpose
proc prepareSectionLists { } {
#List containing the names of possible sections
    global sectionList
    set sectionList {}

    global ListForHosts
    set ListForHosts {}
    lappend ListForHosts [getPrimaryKey]
    lappend ListForHosts default

    #An array of lists for each section
    global ListForSection
    #For each section exists a maximal LabelWidth depending
    #on the length of longest label
    global LabelWidth
    #Counters for the number of items already used in a section
    #for more proper menus
    global Counters


    set ListForSection(HostConfig,hostnames,default) { }
    set sListFile [open token/sections]
    while {![eof $sListFile]} {
        set section [gets $sListFile]
        if {0 < [string length $section]} {
            lappend sectionList $section
            #List for a certain section 
            #            global ${section}List
            #            set ${section}List {}
            set ListForSection($section) {}
            set Counters($section) 0
            set LabelWidth($section) 5
        }
    }
    close $sListFile
    set ListForSection(ButtonConfig) makeButtonConfigGUI
    set ListForSection(PipeConfig) makePipeConfigGUI
    set ListForSection(WindowConfig) makeWindowConfigGUI
    set ListForSection(ScreenConfig) makeScreenConfigGUI
    set ListForSection(ChannelConfig) makeChannelConfigGUI
    set ListForSection(HostConfig) makeHostConfigGUI
    set ListForSection(License) makeLicenseGUI

}
        

#append the hostname to the ListForHosts
#if this not yet happened
proc addToListForHosts { hostname } {
    global ListForHosts
    global ListForSection
    if { [lsearch $ListForHosts $hostname] == -1 } {
        lappend ListForHosts [getPrimaryKey]
        lappend ListForHosts $hostname
        set ListForSection(HostConfig,hostnames,$hostname) { }
    }
}


##########################################################################################
#Helper function to compute the maximum length for a label in  a section
proc computeLabelWidth { section itemname } {
    global LabelWidth
    if { $LabelWidth($section) < [string length $itemname] } {
        set LabelWidth($section) [string length $itemname] 
    }
}
proc ENUMERATE { section enum values } {
    global ListForSection
    set ListForSection($section,$enum) $values
}

#The following procedures prepare building the gui
#The calls are done by grammar2tcl ( see below )
#The parameter section specifies a section in covise.config like
#COVERConfig or TRACKERConfig and so on.
#The parameter specifies an item in such a section like HANDSENSOR_OFFSET
#The parameter enumval specifies a possible value for such an item
#where it can be enumerated like POLHEMUS, MOTIONSTAR, FOB, CAVELIB and so on for the
#item TRACKING_SYSTEM in COVERConfig
proc ONEBOOL { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeONEBOOL $section $itemname"
    lappend ListForSectionSave($section) "saveONEBOOL $section $itemname"
    computeLabelWidth $section $itemname
}
proc THREEFLOAT { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeTHREEFLOAT $section $itemname"
    lappend ListForSectionSave($section) "saveTHREEFLOAT $section $itemname"
    computeLabelWidth $section $itemname
}
proc THREEINT { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeTHREEFLOAT $section $itemname"
    lappend ListForSectionSave($section) "saveTHREEFLOAT $section $itemname"
    computeLabelWidth $section $itemname
}

proc ONEENUM  { section itemname enumval } {
    global ListForSection
    global ListForSectionSave
    if { [lsearch $ListForSection($section) "makeONEENUM $section $itemname"] == -1 } {
        lappend ListForSection($section) "makeONEENUM $section $itemname"
        lappend ListForSectionSave($section) "saveONEENUM $section $itemname"
    }
    lappend ListForSection($section,$itemname) $enumval
    computeLabelWidth $section $itemname
}


proc ONESTRING  { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeONESTRING $section $itemname"
    lappend ListForSectionSave($section) "saveONESTRING $section $itemname"
    computeLabelWidth $section $itemname
}
proc ONESTRINGLIST  { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeONESTRING $section $itemname"
    lappend ListForSectionSave($section) "saveONESTRINGLIST $section $itemname"
    computeLabelWidth $section $itemname
}
proc TWOSTRING  { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeTWOSTRING $section $itemname"
    lappend ListForSectionSave($section) "saveTWOSTRING $section $itemname"
    computeLabelWidth $section $itemname
}

proc INTENUM { section itemname enumval } {
    global ListForSection
    global ListForSectionSave
    if { [lsearch $ListForSection($section) "makeINTENUM $section $itemname"] == -1 } {
        lappend ListForSection($section) "makeINTENUM $section $itemname"
        lappend ListForSectionSave($section) "saveINTENUM $section $itemname"
    }
    lappend ListForSection($section,$itemname) $enumval
    computeLabelWidth $section $itemname
}

proc ONEINT { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeONEINT $section $itemname"
    lappend ListForSectionSave($section) "saveONEINT $section $itemname"
    computeLabelWidth $section $itemname
}

proc ONEFLOAT { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeONEFLOAT $section $itemname"
    lappend ListForSectionSave($section) "saveONEFLOAT $section $itemname"
    computeLabelWidth $section $itemname
}

proc FOURFLOAT { section itemname } {
    global ListForSection
    global ListForSectionSave
    lappend ListForSection($section) "makeFOURFLOAT $section $itemname"
    lappend ListForSectionSave($section) "saveFOURFLOAT $section $itemname"
    computeLabelWidth $section $itemname
}
#########################################################
proc saveONEBOOL { section itemname body hostname} {
    global ListForSection
    global ListForHosts
    upvar $body b
        if { [canBeSaved ListForSection($section,$itemname,$hostname)] } {
            set b "$b     $itemname $ListForSection($section,$itemname,$hostname)\n"
            #if $ListForSection($section,$itemname,$hostname) {
            #    set b "$b     $itemname ON\n"
            #} else {
            #    set b "$b     $itemname OFF\n"
            #}
        }

}

proc saveTHREEFLOAT { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if {[canBeSaved ListForSection($section,$itemname,x,$hostname)] &&
    [canBeSaved ListForSection($section,$itemname,y,$hostname)] &&
    [canBeSaved ListForSection($section,$itemname,z,$hostname)]} {
        set b  "$b     $itemname $ListForSection($section,$itemname,x,$hostname) \
                $ListForSection($section,$itemname,y,$hostname) \
                $ListForSection($section,$itemname,z,$hostname)\n"
    }

}

proc saveTHREEINT { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if {[canBeSaved ListForSection($section,$itemname,x,$hostname)} &&
    [canBeSaved ListForSection($section,$itemname,y,$hostname)] &&
    [canBeSaved ListForSection($section,$itemname,z,$hostname)]] {
        set b "$b     $itemname $ListForSection($section,$itemname,$x,$hostname) \
                $ListForSection($section,$itemname,$y,$hostname) \
                $ListForSection($section,$itemname,$z,$hostname)\n"
    }
}

proc saveONEENUM  { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if [canBeSaved ListForSection($section,$itemname,$hostname)] {
        set b "$b     $itemname $ListForSection($section,$itemname,$hostname)\n"
    }
}

proc saveONESTRING  { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if [canBeSaved ListForSection($section,$itemname,$hostname)] {
        set b "$b     $itemname $ListForSection($section,$itemname,$hostname)\n"
    }

}
proc saveONESTRINGLIST  { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if [canBeSaved ListForSection($section,$itemname,$hostname)] {
        set b "$b     $itemname $ListForSection($section,$itemname,$hostname)\n"
    }

}

proc saveTWOSTRING { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if {[canBeSaved ListForSection($section,$itemname,x,$hostname)] &&
    [canBeSaved ListForSection($section,$itemname,y,$hostname)] &&
    [canBeSaved ListForSection($section,$itemname,z,$hostname)]} {
        set b  "$b     $itemname $ListForSection($section,$itemname,a,$hostname) \
                $ListForSection($section,$itemname,b,$hostname)\n"
    }

}

proc saveINTENUM { section itemname body hostname} {

}

proc saveONEINT { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if [canBeSaved ListForSection($section,$itemname,$hostname)] {
        set b  "$b     $itemname $ListForSection($section,$itemname,$hostname)\n"
    }
}


proc saveONEFLOAT { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if [canBeSaved ListForSection($section,$itemname,$hostname)] {
        set b "$b     $itemname $ListForSection($section,$itemname,$hostname)\n"
    }
}



proc saveFOURFLOAT { section itemname body hostname} {
    upvar $body b
    global ListForSection
    if {[canBeSaved ListForSection($section,$itemname,x,$hostname)] &&
    [canBeSaved ListForSection($section,$itemname,y,$hostname)] &&
    [canBeSaved ListForSection($section,$itemname,z,$hostname)] &&
    [canBeSaved ListForSection($section,$itemname,w,$hostname)]} {
        set b "$b     $itemname $ListForSection($section,$itemname,x,$hostname) \
                $ListForSection($section,$itemname,y,$hostname) \
                $ListForSection($section,$itemname,z,$hostname) \
                $ListForSection($section,$itemname,w,$hostname)\n"
    }
}


#determine whether a value is in a "saveable" state
#i.e. it must exist and must not be empty
proc canBeSaved { name } {
    global ListForSection
    if [info exists $name] {
        upvar $name val
        if { $val == "" } {
            return 0
        } else {
            return 1
        }
    } else {
        return 0
    }
}


source help.tcl


#save the help changed by the user
proc saveHelp { } {
    global Help
    set helpfile [open help.tcl w]
    foreach { index value } [array get Help ] {
        puts $helpfile "set Help($index) \{$value\}"
    }
    close $helpfile
}

################################################################################
#There are several sections which are unimplemented yet
#They can be handled in a simple text editor though

#commit changes to unimplemented sections
proc commitUnimplemented { widget } {
    set end [$widget index end]
    set lines [string range $end 0 [expr [string first  "." $end]-1]]
    global unimplementedList
    set  unimplementedList ""
    for { set i 1 } { $i < $lines } {incr i } {
        lappend unimplementedList [$widget get ${i}.0 ${i}.end]
    }
} 

#read the unimplmented section into the text widget
proc readUnimplemented { widget } {
    global unimplementedList
    $widget delete 0.0 end
    foreach i $unimplementedList {
        $widget insert end  "$i\n"
    }
}

#create and show the text widget to handle unimplemented sections
proc showUnimplemented { } {
    global unimplementedList
    destroy .unimplemented
    toplevel .unimplemented
    frame .unimplemented.upper
    set scroll [scrollbar .unimplemented.upper.scroll  -orient vert -command ".unimplemented.upper.text yview "]
    set text [text .unimplemented.upper.text -yscrollcommand "$scroll set"]
    pack $text $scroll -side left -fill y
    frame .unimplemented.fr
    button .unimplemented.fr.ok \
            -text "OK" \
            -command "commitUnimplemented $text; destroy .unimplemented"
    button .unimplemented.fr.cancel \
            -text "Cancel" \
            -command "destroy .unimplemented"
    button .unimplemented.fr.commit \
            -text "Commit changes" \
            -command "commitUnimplemented $text"
    button .unimplemented.fr.discard \
            -text "Discard changes" \
            -command "readUnimplemented $text"
    pack .unimplemented.fr.ok  \
            .unimplemented.fr.cancel \
            .unimplemented.fr.commit \
            .unimplemented.fr.discard \
            -side left
    pack .unimplemented.upper \
            .unimplemented.fr \
            -side top
    readUnimplemented $text
}

#show a help window for a certain item in a section
#Currently changes to the help can be made and saved by the 
#user. The OK button and the possibility to edit the help will 
#dissapear when help is ready
proc showHelp { section itemname } {
    global Help
    destroy .help$section$itemname
    toplevel .help$section$itemname
    text .help$section$itemname.text
    frame .help$section$itemname.fr
    button .help$section$itemname.fr.ok \
            -text "OK" \
            -command "set Help($section,$itemname) \[.help$section$itemname.text get 0.0 end\]; saveHelp;destroy .help$section$itemname"
    button .help$section$itemname.fr.cancel \
            -text "Cancel" \
            -command "destroy .help$section$itemname"
    pack .help$section$itemname.fr.ok  \
            .help$section$itemname.fr.cancel \
            -side left
    pack .help$section$itemname.text \
            .help$section$itemname.fr \
            -side top
    if {[info exists Help($section,$itemname)]} {
        .help$section$itemname.text insert end  $Help($section,$itemname)
    }
}

#show a help window for a certain  section
#Currently changes to the help can be made and saved by the 
#user. The OK button and the possibility to edit the help will 
#dissapear when help is ready
proc showSectionHelp { section } {
    global Help
    destroy .help$section
    toplevel .help$section
    text .help$section.text
    frame .help$section.fr
    button .help$section.fr.ok \
            -text "OK" \
            -command "set Help($section) \[.help$section.text get 0.0 end\]; saveHelp;destroy .help$section"
    button .help$section.fr.cancel \
            -text "Cancel" \
            -command "destroy .help$section"
    pack .help$section.fr.ok  \
            .help$section.fr.cancel \
            -side left
    pack .help$section.text \
            .help$section.fr \
            -side top
    if {[info exists Help($section)]} {
        .help$section.text insert end  $Help($section)
    }
}


##########################################################################################
#The following procedures actually build the widgets needed for the items in
#each section
#There shall be an own window for each section 
#whis is shown on demand

proc makeTHREEINT { section itemname hostname pk } {
    makeTHREENUMBERS $section $itemname isInteger green $hostname $pk
}

proc makeTHREENUMBERS { section itemname validatecommand entrycolor hostname pk} {
    global Counters
    global MAXITEMS
    global ListForSection
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    frame $f.fr${frameno}.f$itemname


    global LabelWidth

    label $f.fr${frameno}.f$itemname.l -text $itemname -width $LabelWidth($section) -anchor w
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    frame $f.fr${frameno}.f$itemname.f
    foreach vname { x y z } {
        entry $f.fr${frameno}.f$itemname.f.$vname \
                -textvariable ListForSection($section,$itemname,$vname,$hostname) \
                -width 6 \
                -background $entrycolor \
                -width 6 \
                -validate all \
                -validatecommand "$validatecommand %d %i %S %V %W" \
                -invalidcommand  "invalid %V %W"
    }
    pack $f.fr${frameno}.f$itemname.f.z \
            $f.fr${frameno}.f$itemname.f.y \
            $f.fr${frameno}.f$itemname.f.x \
            -side right -anchor w
    pack $f.fr${frameno}.f$itemname.l \
            $f.fr${frameno}.f$itemname.f \
            -side left -anchor w
    pack $f.fr${frameno}.f$itemname \
            -side top -fill x -anchor w
    incr Counters($section)
}

proc makeONENUMBER { section itemname validatecommand entrycolor hostname pk} {
    global Counters
    global MAXITEMS
    global ListForSection
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    frame $f.fr${frameno}.f$itemname
    global LabelWidth
    label $f.fr${frameno}.f$itemname.l -text $itemname -width $LabelWidth($section) -anchor w
    frame $f.fr${frameno}.f$itemname.f
    entry $f.fr${frameno}.f$itemname.f.x \
            -textvariable ListForSection($section,$itemname,$hostname) \
            -background $entrycolor -width 6 -validate all\
            -validatecommand "$validatecommand %d %i %S %V %W" \
            -invalidcommand  "invalid %V %W"
    pack  $f.fr${frameno}.f$itemname.f.x -side right
    pack $f.fr${frameno}.f$itemname.l \
            $f.fr${frameno}.f$itemname.f \
            -side left
    pack $f.fr${frameno}.f$itemname \
            -side top \
            -fill x
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    incr Counters($section)
}

proc makeINTENUM { section itemname hostname pk } {
    global Counters
    global MAXITEMS
    global ListForSection
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    frame $f.fr${frameno}.f$itemname \
            -relief sunken \
            -borderwidth 2
    global LabelWidth
    label $f.fr${frameno}.f$itemname.l \
            -text $itemname \
            -width $LabelWidth($section) \
            -anchor w
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    frame $f.fr${frameno}.f$itemname.f


    set enumframeno 0
    frame $f.fr${frameno}.f$itemname.f.f${enumframeno}
    pack  $f.fr${frameno}.f$itemname.f.f${enumframeno} -side top
    incr Counters($section)

    entry $f.fr${frameno}.f$itemname.f.f${enumframeno}.x \
            -textvariable ListForSection($section,$itemname,$hostname,x) \
            -background red -width 6 -validate all\
            -validatecommand "isInteger %d %i %S %V %W" \
            -invalidcommand  "invalid %V %W"
    pack $f.fr${frameno}.f$itemname.f.f${enumframeno}.x -side left
    set enumcount 1

    set varname $section\($itemname,enumval\)
    foreach i $ListForSection($section,$itemname) {
        global MAXENUM
        if { 0 == [expr $enumcount%$MAXENUM] }  {
            set enumframeno [expr $enumcount/$MAXENUM]
            frame $f.fr${frameno}.f$itemname.f.f${enumframeno}
            pack  $f.fr${frameno}.f$itemname.f.f${enumframeno} \
                    -side top
            incr Counters($section)
        }
        radiobutton $f.fr${frameno}.f$itemname.f.f${enumframeno}.rb$i -\
                text $i \
                -variable ListForSection($section,$itemname,enumval,$hostname) \
                -value $i
        pack $f.fr${frameno}.f$itemname.f.f${enumframeno}.rb$i \
                -side left
        incr enumcount
    }
    pack $f.fr${frameno}.f$itemname.l $f.fr${frameno}.f$itemname.f \
            -side left
    pack $f.fr${frameno}.f$itemname \
            -side top \
            -fill x
    
}

proc makeONEENUM  { section itemname hostname pk } {
    global Counters
    global MAXITEMS
    global ListForSection
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    frame $f.fr${frameno}.f$itemname -relief sunken -borderwidth 2
    global LabelWidth
    label $f.fr${frameno}.f$itemname.l \
            -text $itemname \
            -width $LabelWidth($section) \
            -anchor w
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    frame $f.fr${frameno}.f$itemname.f
    set enumcount 0
    foreach i $ListForSection($section,$itemname) {
        global MAXENUM
        set row [expr $enumcount/$MAXENUM] 
        set col [expr $enumcount%$MAXENUM] 

        radiobutton $f.fr${frameno}.f$itemname.f.rb$i \
                -text $i \
                -variable ListForSection($section,$itemname,$hostname) \
                -value $i
        grid $f.fr${frameno}.f$itemname.f.rb$i \
                -row $row \
                -col $col \
                -sticky w
        incr enumcount
    }
    incr Counters($section) [expr $enumcount/$MAXENUM]
    pack $f.fr${frameno}.f$itemname.l \
            $f.fr${frameno}.f$itemname.f \
            -side left
    pack $f.fr${frameno}.f$itemname \
            -side top \
            -fill x
}

proc makeTHREEFLOAT { section itemname hostname pk } {
    makeTHREENUMBERS $section $itemname isFloat red $hostname $pk
}

proc makeONEBOOL { section itemname hostname pk } {
    global Counters
    global MAXITEMS
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    set varname $section\($itemname\)
    checkbutton $f.fr${frameno}.cb$itemname \
            -text $itemname \
            -variable ListForSection($section,$itemname,$hostname)
    bind $f.fr${frameno}.cb$itemname <Button-3> "showHelp $section $itemname" 
    pack $f.fr${frameno}.cb$itemname \
            -side top \
            -pady 2 \
            -anchor w
    incr Counters($section)
}
proc makeONEBOOL  { section itemname hostname pk } {
    global Counters
    global MAXITEMS
    global ListForSection
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    frame $f.fr${frameno}.f$itemname -relief sunken -borderwidth 2
    global LabelWidth
    label $f.fr${frameno}.f$itemname.l \
            -text $itemname \
            -width $LabelWidth($section) \
            -anchor w
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    frame $f.fr${frameno}.f$itemname.f
    set enumcount 0
    foreach i { ON OFF } {
        global MAXENUM
        set row [expr $enumcount/$MAXENUM] 
        set col [expr $enumcount%$MAXENUM] 

        radiobutton $f.fr${frameno}.f$itemname.f.rb$i \
                -text $i \
                -variable ListForSection($section,$itemname,$hostname) \
                -value $i
        grid $f.fr${frameno}.f$itemname.f.rb$i \
                -row $row \
                -col $col \
                -sticky w
        incr enumcount
    }
    incr Counters($section) [expr $enumcount/$MAXENUM]
    pack $f.fr${frameno}.f$itemname.l \
            $f.fr${frameno}.f$itemname.f \
            -side left
    pack $f.fr${frameno}.f$itemname \
            -side top \
            -fill x
}


#Callback for an integer-entry to check whether the value is 
#valid for tha entry

proc isInteger { type position input validationType widget }  {
    switch $validationType {
        key {
            if {$type} {
                #checkstring which results if the input would be accepted
                set checkstring [string range [$widget get] 0 [expr $position-1]]$input[string range [$widget get] $position end]
                return [regexp {^-?[0-9]*$} $checkstring ]
            } else {
                return 1
            }
        }
        focus { return 1}
        focusin { return 1 }
        focusout {
            return [regexp {(^$|^-?[0-9]+$)} "[$widget get]" ]
            return 1
        }
        forced { return 1 }
    }
}

#Callback for an float-entry to check whether the value is 
#valid for that entry
proc isFloat { type position input validationType widget }  {
    switch $validationType {
        key {
            if {$type} {
                #checkstring which results if the input would be accepted
                set checkstring [string range [$widget get] 0 [expr $position-1]]$input[string range [$widget get] $position end]
                return [regexp {^-?(([0-9]*)|([0-9]+\.[0-9]*))$} $checkstring ]
            } else {
                return 1
            }
        }
        focus { return 1}
        focusin { return 1 }
        focusout {
            return [regexp {(^$|^-?[0-9]+(\.[0-9]+)?$)} "[$widget get]" ]
            return 1
        }
        forced { return 1 }
    }
}

#This procedure is called if isFloat or isInteger fails
#it refocusses on the entry causing the error
proc invalid { validationType widget } {
    switch $validationType {
        key {  bell ; return }
        focus { puts "focus" }
        focusin { puts "focusin" }
        focusout { 
            focus $widget
            set varname [$widget cget -textvariable]
            global $varname
            set $varname ""
        }
        forced { puts "forced"  }
    }
}

proc makeONEINT { section itemname hostname pk } {
    makeONENUMBER $section $itemname isInteger green $hostname $pk
}
proc makeONEFLOAT { section itemname hostname pk} {
    makeONENUMBER $section $itemname isFloat red $hostname $pk
}

proc makeONENUMBER { section itemname validatecommand entrycolor hostname pk} {
    global Counters
    global MAXITEMS
    global ListForSection
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    frame $f.fr${frameno}.f$itemname
    global LabelWidth
    label $f.fr${frameno}.f$itemname.l \
            -text $itemname \
            -width $LabelWidth($section) \
            -anchor w
    frame $f.fr${frameno}.f$itemname.f
    entry $f.fr${frameno}.f$itemname.f.x \
            -textvariable ListForSection($section,$itemname,$hostname) \
            -background $entrycolor \
            -width 6 \
            -validate all\
            -validatecommand "$validatecommand %d %i %S %V %W" \
            -invalidcommand  "invalid %V %W"
    pack  $f.fr${frameno}.f$itemname.f.x \
            -side right
    pack $f.fr${frameno}.f$itemname.l \
            $f.fr${frameno}.f$itemname.f \
            -side left
    pack $f.fr${frameno}.f$itemname \
            -side top \
            -fill x
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    incr Counters($section)
}


proc makeTWOSTRING { section itemname  hostname pk} {
    global Counters
    global MAXITEMS
    global ListForSection
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    frame $f.fr${frameno}.f$itemname


    global LabelWidth

    label $f.fr${frameno}.f$itemname.l -text $itemname -width $LabelWidth($section) -anchor w
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    frame $f.fr${frameno}.f$itemname.f
    foreach vname { a b } {
        entry $f.fr${frameno}.f$itemname.f.$vname \
                -textvariable ListForSection($section,$itemname,$vname,$hostname) \
                -width 16
    }
    pack $f.fr${frameno}.f$itemname.f.b \
            $f.fr${frameno}.f$itemname.f.a \
            -side right -anchor w
    pack $f.fr${frameno}.f$itemname.l \
            $f.fr${frameno}.f$itemname.f \
            -side left -anchor w
    pack $f.fr${frameno}.f$itemname \
            -side top -fill x -anchor w
    incr Counters($section)
}






proc makeONESTRING  { section itemname hostname pk} {
    global Counters
    global MAXITEMS
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    frame $f.fr${frameno}.f$itemname
    global LabelWidth
    label $f.fr${frameno}.f$itemname.l \
            -text $itemname \
            -width $LabelWidth($section) \
            -anchor w
    frame $f.fr${frameno}.f$itemname.f
    entry $f.fr${frameno}.f$itemname.f.x \
            -textvariable ListForSection($section,$itemname,$hostname) \
            -width 20
    pack  $f.fr${frameno}.f$itemname.f.x \
            -side right
    pack $f.fr${frameno}.f$itemname.l \
            $f.fr${frameno}.f$itemname.f \
            -side left
    pack $f.fr${frameno}.f$itemname \
            -side top \
            -fill x
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    incr Counters($section)
}

proc makeFOURFLOAT { section itemname hostname pk} {
    global Counters
    global MAXITEMS
    set frameno [expr $Counters($section)/$MAXITEMS]

    set f [getFrame $section $pk]
    
    frame $f.fr${frameno}.f$itemname
    global LabelWidth
    label $f.fr${frameno}.f$itemname.l \
            -text $itemname \
            -width $LabelWidth($section) \
            -anchor w
    frame $f.fr${frameno}.f$itemname.f
    foreach vname {x y z w } {
        entry $f.fr${frameno}.f$itemname.f.$vname \
                -textvariable ListForSection($section,$itemname,$vname,$hostname) \
                -width 6 \
                -background red \
                -width 6 \
                -validate all\
                -validatecommand "isFloat %d %i %S %V %W" \
                -invalidcommand  "invalid %V %W"
    }
    pack  $f.fr${frameno}.f$itemname.f.w \
            $f.fr${frameno}.f$itemname.f.z  \
            $f.fr${frameno}.f$itemname.f.y  \
            $f.fr${frameno}.f$itemname.f.x \
            -side right
    pack $f.fr${frameno}.f$itemname.l \
            $f.fr${frameno}.f$itemname.f \
            -side left
    pack $f.fr${frameno}.f$itemname \
            -side top \
            -fill x
    bind $f.fr${frameno}.f$itemname.l <Button-3> "showHelp $section $itemname" 
    incr Counters($section)
}


proc getFrame { section pk } {
    return .w${section}.f${pk}.canvasframe.f
}

proc getCanvas { section pk } {
    return .w${section}.f${pk}.canvasframe.canvas
}

#create a canvas in which the gui for a section
#is embedded. 
#This is especially useful if the gui is so
#big, that it needs scrollbars.
proc Scrolled_Canvas { c section} {
#    frame $c
    frame $c.canvasframe
    #horizontal and vertical scrollbar
    scrollbar $c.xscroll \
            -orient horizontal \
            -command [list $c.canvasframe.canvas xview]
    scrollbar $c.yscroll \
            -orient vertical \
            -command [list $c.canvasframe.canvas yview]
    canvas $c.canvasframe.canvas \
            -xscrollcommand [list $c.xscroll set] \
            -yscrollcommand [list $c.yscroll set] \
            -highlightthickness 0 \
            -borderwidth 0
    pack $c.canvasframe.canvas -fill both
    grid $c.canvasframe \
            -padx 1 -pady 1 \
            -row 0 -column 0 \
            -rowspan 1 \
            -sticky news
    grid $c.yscroll \
            -padx 0 -pady 1 \
            -row 0 -column 1 \
            -rowspan 1 -columnspan 1 \
            -sticky news
    grid $c.xscroll \
            -padx 1 -pady 0 \
            -row 1 -column 0 \
            -rowspan 1 -columnspan 1 \
            -sticky news
    grid rowconfig    $c 0 -weight 1 -minsize 0
    grid columnconfig    $c 0 -weight 1 -minsize 0
    frame $c.canvasframe.f
    
    #Now the frame is assigned to the canvas
    eval { $c.canvasframe.canvas create window 0 0 -window $c.canvasframe.f -anchor nw  }
    return $c.canvasframe
}



#helper procedure to unpack all host specific frames
proc unpackAll { section } {
    global ListForHosts
    foreach { pk hostname } $ListForHosts {
        pack forget .w${section}.f${pk}
    }
}


#creates frames for a section containing a scrolled canvas 
#for each hostname there is one such frame
#but only one is displayed
#The user can choose the section for another
#host by pressing a button
proc makeSectionGUI { section } {
    global ListForSection
    global ListForHosts
    toplevel .w${section}
    frame .w${section}.switch

    set hostFrame [Scrolled_Canvas .w${section}.switch $section]
    set cFrame $hostFrame.f
    set canvas $hostFrame.canvas
    foreach {pk hostname } $ListForHosts {
        if { $hostname == "default" } {
            set defaultPk $pk
        }
        global ListForHosts2
        set ListForHosts2($section) $defaultPk
        radiobutton $cFrame.b${pk}\
                -text ${hostname}\
                -variable ListForHosts2($section)\
                -value $pk\
                -command "changeFrame $section $pk"
        #pack  .w${section}.b${pk} -side top -anchor w
        set lastbutton $cFrame.b${pk}
        pack  $lastbutton -side top -anchor w

        frame .w${section}.f${pk}
        global Counters
        makeSectionFrame $section $hostname $pk

    }

    button .w${section}.destroy \
            -command "set Counters($section) 0; destroy .w${section}" \
            -text "Back to main"
    pack .w${section}.destroy \
            -side top \
            -fill both
    button .w${section}.help \
            -command "showSectionHelp $section" \
            -text "Help"
    pack .w${section}.help \
            -side top \
            -fill both


    pack .w${section}.switch  -side top -fill both -expand true
    tkwait vis .w${section}.switch
    setScrollRegion $canvas $cFrame 10 200


    #The application reminds the current visible frame
    #When the visible frame is changed by choosing another hostname
    #the scroll position of its ancestor can be adopted
    global LastPk
    set LastPk($section) $defaultPk
    pack .w${section}.f$defaultPk -side bottom -fill both -expand true

    foreach {pk hostname } $ListForHosts {
        setScrollRegion [getCanvas $section $pk] [getFrame $section $pk] 800 600
    }
}

#configure the scrollbars of the canvas
proc setScrollRegion {canvas window maxwidth maxheight} {
    $canvas configure -scrollregion "-1 -1 [expr [winfo reqwidth $window] + 1] \
            [expr [winfo reqheight $window] + 1]"

    #set height and width of the canvas in a way
    #tha it is big enough but fits on a normal screen
    set height [winfo reqheight $window]

    if { $height > $maxheight } {
        set height $maxheight
    }
    $canvas configure -height $height

    set width [winfo reqwidth $window]
    if { $width > $maxwidth } {
        set width $maxwidth
    }
    $canvas configure -width $width
}


proc changeFrame { section pk} {
    #unpackAll $section
    global LastPk
    set lastFrame ".w${section}.f$LastPk($section)"
    pack forget $lastFrame

    pack .w${section}.f${pk} -side top -fill both -expand true 


    #now adopt the scroll position of the former frame

    set lastWidget [getCanvas $section $LastPk($section)]

    #get the position of scrollbars of the last active frame
    set xpos [lindex [$lastWidget xview] 0]
    set ypos [lindex [$lastWidget yview] 0]
    #adapt the scrollbars of the current frame
    [getCanvas $section $pk] xview moveto $xpos
    [getCanvas $section $pk] yview moveto $ypos
    set LastPk($section) $pk
}

proc makeSectionFrame { section hostname pk } {
    set c  [Scrolled_Canvas .w${section}.f${pk} $section]
    
    set canvas $c.canvas
    set f $c.f
    for { set i 0 } { $i < 7 } { incr i } {
        frame $f.fr$i
        grid $f.fr$i -row 0 -column $i -sticky n
    }
    global ListForSection
    foreach i $ListForSection($section) {
        eval  "$i $hostname $pk"
    }
    global Counters
    set Counters($section) 0

}

#helper function used by saveValues.
#the help existing for a section
#is printed as comment before the section
proc makeHelpAsComment { section header } {
    upvar $header h
    global Help
    set h ""
    if {[info exists Help($section)]} {
        foreach line [split $Help($section) \n] {
            set  h "${h}#${line}\n"
        }
    }
}


#save the values to the formerly openend channel

proc saveValues  { channel } {
    global sectionList
    global ListForSectionSave
    set ListForSectionSave(HostConfig) { saveHostConfig }
    set ListForSectionSave(WindowConfig) { saveWindowConfig }
    set ListForSectionSave(ScreenConfig) { saveScreenConfig }
    set ListForSectionSave(ChannelConfig) { saveChannelConfig }
    set ListForSectionSave(PipeConfig) { savePipeConfig }
    set ListForSectionSave(ButtonConfig) { saveButtonConfig }
    
    global ListForHosts
    foreach { pk hostname } $ListForHosts { 
        foreach section $sectionList {
            if { $hostname == "default" } {
                set header "$section\n\{\n"
            } else {
                set header "$section: $hostname\n\{\n"
            }
            set body ""

            if [info exists ListForSectionSave($section)] {
                foreach i $ListForSectionSave($section) {
                    eval "$i body $hostname"
                }
            }
            if { $body != "" } {
                set help ""
                makeHelpAsComment $section help
                puts $channel "${help}${header}${body}\}"
                puts $channel \n
            }
        }
    }
}



#helper function to save the unimplemented sections
proc saveUnimplemented { channel } {
    global unimplementedList
    foreach i $unimplementedList {
        puts $channel $i
    }
}


#save your work.
#The user is prompted a fileselector box
proc save  { } {
    set file [tk_getSaveFile -initialfile covise.config]
    
    if { ${file} != "" } {
        if [catch "open ${file} w" desc] {
            tk_messageBox -icon error -message $desc -type ok
        } else {
            saveValues $desc
            saveUnimplemented $desc
            close $desc
        }
    }
}


proc makeGUI { } {
    global sectionList
    set i 1

    #The buttons for each section are displayed
    #in two contigous columns
    foreach { section1 section2 }  $sectionList {
        button .b${section1} -width 20 \
                -text $section1 \
                -command "makeSectionGUI $section1"
        grid .b${section1} -row $i -col 0 -sticky w
        #The number of sections in sectionList
        #may be odd.
        #in this case the last section2 is empty
        if { [string length ½section2] != 0 } {
            button .b${section2} -width 20 \
                -text $section2 \
                -command "makeSectionGUI $section2"
            grid .b${section2} -row $i -col 1 -sticky w
        }
        incr i
    }
    button .bUnimplemented -width 20 -text "Unimplemented" -command showUnimplemented
    grid .bUnimplemented -row $i -col 0 -sticky w
    button .bSave -width 20 -text "Save" -command save
    grid .bSave -row $i -col 1 -sticky w
    incr i
    button .bQuit -width 20 -text "Quit" -command "exit"
    grid .bQuit -row $i -col 0 -sticky n
}


#For some items in COVERConfig values are mandatory
#since other sections depend on them
proc setNeededDefaults { } {
    global ListForSection
    global ListForHosts
    foreach {pk hostname } $ListForHosts {
        foreach { item } {NUM_WINDOWS  NUM_PIPES NUM_SCREENS } {
            if ![info exists ListForSection(COVERConfig,$item,$hostname)] {
                set ListForSection(COVERConfig,$item,$hostname) 1
            }
        }
    }
}



source hostconfig.tcl
#source windowconfig.tcl
source pipeconfig.tcl
source buttonconfig.tcl
source license.tcl
source uiconfig.tcl

#we need the possibility to
#provide unique id's for some 
#reasons
proc getPrimaryKey {  } {
    global primaryKey
    incr primaryKey
    return $primaryKey
}
proc getLastPrimaryKey {  } {
    global primaryKey
    return $primaryKey
}
source windowconfig.tcl
source screenconfig.tcl
source channelconfig.tcl


proc main {  } {
    global primaryKey
    set primaryKey 0
    #Global variable specifying how many items shall be in one columns
    #i.e. the maximum of lines a menu can have
    global MAXITEMS
    set MAXITEMS 150
    #maximal number of radiobuttons allowed in on line 
    global MAXENUM
    set MAXENUM 2
    global ListForSection
    prepareSectionLists
    
    #The major part of the gui/design
    #can be extracted from the grammar
    #for covise.config
    eval [ exec grammar2tcl  ]


    

    global env
    set P $env(COVISEDIR)/$env(ARCHSUFFIX)/bin
    puts "PPPPPPPPPPPPPPPPPPPPPPPPP $P"
    #List containing all error messages generated by the parser
    global errList
    global statusList
    set errList ""
    set statusList ""
    #exec coviseconfig2tcl testfile
    global argv
    if { $argv == "" }  {
	set argv testfile
    }   


    makeGUI
    puts "exec ${P}/coviseconfig2tcl $argv"
    eval [exec ${P}/coviseconfig2tcl $argv]


    global sectionList
    global Counters
    foreach section $sectionList {
        set Counters($section) 0
    }
    eval [exec ${P}/coviseconfig2tcl standard]
    setNeededDefaults
    puts "The following errors occured $errList"
}




main


