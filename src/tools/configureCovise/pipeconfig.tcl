#if the number of pipes is changed in COVERConfig
#and the gui for PipeConfig is in use then
#it will be refreshed according to the new number of pipes
proc refreshPipeConfigGUI { arrayname arrayindex op } {
    global ListForSection
    if [winfo exists .wPipeConfig] {
	destroy .wPipeConfig
    }
    trace variable ListForSection(COVERConfig,NUM_PIPES) wu refreshPipeConfigGUI
}
trace variable ListForSection(COVERConfig,NUM_PIPES) wu refreshPipeConfigGUI


proc PIPECONFIGITEM {  softpipe hardpipe display hostname} {
    global ListForSection
    set ListForSection(PipeConfig,hardpipe,$softpipe,$hostname) $hardpipe
    set ListForSection(PipeConfig,display,$softpipe,$hostname) $display
}

proc savePipeConfig { } {
    global ListForSection
}

proc savePipeConfig {body hostname } {
    upvar $body b
    global ListForSection
    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_PIPES,$hostname) } { incr i } {
        if { [canBeSaved ListForSection(PipeConfig,hardpipe,$i,$hostname)] && [canBeSaved ListForSection(PipeConfig,display,$i,$hostname)] } {
            set b "$b     $i $ListForSection(PipeConfig,hardpipe,$i,$hostname) $ListForSection(PipeConfig,display,$i,$hostname)\n"
        }
    }
}

proc makePipeConfigGUI { } {
    destroy .wPipeConfig
    toplevel .wPipeConfig
    global ListForSection
    set i 0
    #create header for the panel
    foreach { l t } {softpipe SoftPipe  hardpipe Hardpipe  display Display} {
        label .wPipeConfig.$l -text $t
        grid .wPipeConfig.$l -row 0 -column $i -sticky w
        incr i 
    }

    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_PIPES) } { incr i }  {
        label .wPipeConfig.lSoftpipe$i -text $i
        grid .wPipeConfig.lSoftpipe$i -row [expr $i+1] -column 0
        set j 1
        foreach e { hardpipe display } {
            entry .wPipeConfig.e$e$i -textvariable ListForSection(PipeConfig,$e,$i)
            grid .wPipeConfig.e$e$i -column $j -row [expr $i+1]
            incr j
        }
    }
}

proc makePipeConfigGUI { hostname pk } {
    global ListForSection

    set f [getFrame PipeConfig $pk]

    set i 0
    #create header for the panel
    foreach { l t } {softpipe SoftPipe  hardpipe Hardpipe  display Display} {
        label $f.$l -text $t
        bind  $f.$l <Button-3> "showHelp PipeConfig $l" 
        grid $f.$l -row 0 -column $i -sticky w
        incr i 
    }

    for { set i 0 } { $i < $ListForSection(COVERConfig,NUM_PIPES,$hostname) } { incr i }  {
        label $f.lSoftpipe$i -text $i

        grid $f.lSoftpipe$i -row [expr $i+1] -column 0
        set j 1
        foreach e { hardpipe display } {
            entry $f.e$e$i -textvariable ListForSection(PipeConfig,$e,$i,$hostname)
            grid $f.e$e$i -column $j -row [expr $i+1]
            incr j
        }
    }
}

#NUM_PIPES <= Anzahl der Grafikkarten ( Pipes ) 
#Es können mehrere Windows von einer Pipe bedient werden.
#Softpipes werden von 0 aufsteigend numeriert und jeweils einem
#Display zugeordnet

#                             ist Teil           liegt in 
#Projektor <---------Channel <---------- Window <------- Softpipe
#In ChannelConfig kann z.B. festgelegt werden, daß Projektor 1
#die Koordinaten sowieso aus dem ersten Window verwendet
#Die Softpipes müssen auf verschiedenen Displays liegen.




