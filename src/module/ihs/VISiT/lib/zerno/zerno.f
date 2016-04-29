C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
	  SUBROUTINE MAINN(isp,lmax,numparts,
     *               nume, ellist,
     *               nump, px, py, pz,
     *               colnode, nodeinfo,
     *               colelem, eleminfo,
     *               coldir, numdir,
     *               dirindex, dirval,
     *               colwall, numwall, wall,
     *               colbal, numbal, balance,
     *               colpress, numpress, press, 
c fl    *               pressval,
     *               covise_run,write_files)
c
      implicit none

      include 'mpif.h'
      include 'common.zer'
c
c------------------- Parameter for COVISE-version ...
	  integer    covise_run
      integer    write_files
	  integer    numparts
      integer    colelem,eleminfo
      dimension  eleminfo(colelem)
      integer    colnode,nodeinfo
      dimension  nodeinfo(colnode)
      integer    nume, ellist
      dimension  ellist(nume*8)
      integer    nump
      real       px, py, pz
      dimension  px(nump),py(nump),pz(nump)
      integer    coldir,numdir, dirindex
      real       dirval
      dimension  dirindex(coldir*numdir)
      dimension  dirval(numdir)
      integer    colwall, numwall, wall
      dimension  wall(numwall*colwall)
      integer    colbal, numbal, balance
      dimension  balance(numbal*colbal)
      integer    colpress, numpress, press
      dimension  press(numpress*colpress)
c fl     real       pressval
c fl     dimension  pressval(numpress)
      
      integer    cov_coord, cov_lnods, cov_lnods_num
      integer    cov_lnods_proz, cov_coord_num
      integer    cov_coord_joi, cov_lnods_joi
      integer    cov_coord_mod, cov_lnods_mod
      integer    cov_coord_proz
      integer    cov_displ_kn,cov_displ_typ,
     *           cov_wand_el,cov_wand_kn,cov_wand_num,
     *           cov_syme_el,cov_syme_kn,
     *           cov_pres_el,cov_pres_kn,cov_pres_num,
     *           cov_conv_el,cov_conv_kn,cov_conv_num,
     *           cov_displ_wert

C
      integer ISP,icoord_zeig,ilnods_zeig,
     *        icoord,icoord_num,icoord_mod,
     *        ilnods,ilnods_num,ilnods_mod,
     *        ikomp_e,ikomp_d,iparti,
     *        luerr

      integer ielpar,iknpar,
     *        igemat,igemat_adr,nl_gemat,
     *        ifarb_geb,ifarb_adr,ifarb_per,nfarb_geb,
     *        ndat_max,nparti,zer_zeig,nzer,
     *        grad_max,grad_min

      integer ielmat,ielmat_adr,ielmat_stu,nl_elmat,
     *        ikelem,ikelem_adr,nl_kelem
     
      integer ikern_kn,ikern_kn_adr,
     *        ilapp_kn,ilapp_kn_adr,ilapp_kn_proz,
     *        idopp_kn,idopp_kn_adr,idopp_kn_proz,
     *        ikern_el,ikern_el_adr,
     *        ilapp_el,ilapp_el_adr,ilapp_el_proz,
     *        nlapp_el,nkern_max,nlapp_kn,
     *        idopp_el,idopp_el_adr,idopp_el_proz,
     *        ndopp_el,ndopp_kn

      integer ielra_kno,ielra_ele,ielra_num,ielra_wer,ielra_adr,
     *        iknra_kno,iknra_mat,iknra_wer,iknra_adr,
     *        ndisp_zeil,ndisp_spalt,
     *        irand_dim,igeom_dim

c     integer irecv_kn,irecv_adr_kn,isend_kn,isend_adr_kn,
c    *        irecv_buf,isend_buf,
c    *        ieldop_num,ieldop_joi,ieldop_fak,
c    *        ikndop_num,ikndop_joi,ikndop_fak

      integer iint1_help,iint2_help,iint3_help,
     *        igeb1_help,igeb2_help,igeb3_help,igeb4_help,
     *        igeb5_help,igeb6_help,
     *        ipar1_help,ipar2_help,ipar3_help,ipar4_help,
     *        ipar5_help,ipar6_help

      integer lmax,iende,ifine,speich_max_all,
     *        speich_mom,speich_max,speich_max_sub,nfrei

      integer my_num,proc_anz,lu,iii,
     *        ip1,ip2,ip3,ip4,ipfad,lentb

      integer nsp_kn_erg,nsp_el_erg,nsp_za_erg,
     *        ierg_kn,ierg_el,ierg_za

      character*80 geo_name,rbe_name,erg_name,
     *             geo_pfad,rbe_pfad,erg_pfad,
     *             para_file,prot_file,comment,
     *             mit_name,mit_pfad

      character*80 ober_name_geo,ober_name_ses

      character*80 text_1,text_2 

      character*4  otto

      logical ober_geom,print_all,schreiben,
     *        transi_erg,
     *        graph_all,graph_fla,redu_graph,dopplapp

      parameter(ndat_max=10)

      dimension  zer_zeig(ndat_max)

      dimension  isp(lmax)
c     **********************************************************


c     **********************************************************
      CALL MPI_INIT(ierr)
      CALL MPI_COMM_RANK(MPI_COMM_WORLD,my_num,ierr)
      CALL MPI_COMM_SIZE(MPI_COMM_WORLD,proc_anz,ierr)

      myid=my_num
      numprocs=proc_anz

      ifine=lmax
      iende=1
      speich_max=0

      luout=6
      lupro=53
      lupar=54

      nzwei=2
      ndrei=3

      do 120 iii=2,80 
         text_1(iii-1:iii)='*'
         text_2(iii-1:iii)='-'
 120  continue
c     **********************************************************


c      write(*,*) '****************'
c      write(*,*) 'colnode  =', colnode
c      write(*,*) 'colelem  =', colelem
c      write(*,*) 'coldir   =', coldir
c      write(*,*) 'numdir   =', numdir
c      write(*,*) 'colwall  =', colwall
c      write(*,*) 'colbal   =', colbal
c      write(*,*) 'colpress =', colpress
c      write(*,*) '*******************'
c     **********************************************************
c     STEUER-FILE EINLESEN UND SETZEN DER DIMENSIONEN USW.
c     IN ABHAENGIGKEIT DER EINGELESENEN DATEN:

      CALL LES_STF(geo_name,rbe_name,erg_name,
     *             geo_pfad,rbe_pfad,erg_pfad,
     *             print_all,
     *             ober_geom,redu_graph,
     *             zer_zeig,nzer,ndat_max,dopplapp,
     *             mit_name,mit_pfad)
      if (covise_run.ne.0) then
         ngebiet = numparts
      endif

c     if (myid.eq.0) then
c        write(6,*)'Anzahl Gebiete eingeben '
c        read(5,*) ngebiet
c     endif
      if (parallel) then
         CALL MPI_BCAST(ngebiet,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
      endif

      if (.not.parallel) then
        numprocs=1
        myid=0
      endif

      if (ncd.eq.2) then
         nkd=4
         nrbknie=2
      else
         nkd=8
         nrbknie=4
      endif

      if (covise_run.ne.0) then
          write(*,*)'COVISE-Version'
      else
	      write_files=1
      endif
          if (covise_run.ne.0.and.ncd.ne.3) then
                call erro_init(myid,parallel,luerr)
                write(luerr,*)'Fehler im Hauptprogramm.'
                write(luerr,*)'ZERNO mit COVISE geht nur 3-D.'
                call erro_ende(myid,parallel,luerr)
          endif

c     Initialisieren der Zeiger-Felder im Include-File:
      CALL ZEIGER_INIT()
c     **********************************************************


c     **********************************************************
c     OEFFNEN VON PROTOKOLL-FILES:

      write(otto,'(i4.4)') myid+1
      para_file=' '
      ipfad=lentb(para_file)
      ip1=ipfad+1
      ip2=ipfad+4
      para_file(ip1:ip2)='PRO_'
      ip3=ip2+1
      ip4=ip3+3
      para_file(ip3:ip4)=otto(1:4)
      open(lupar,file=para_file,status='unknown')

      if (myid.eq.0) then
          prot_file='zerno.info'
          open(lupro,file=prot_file,status='unknown')
      endif
c     **********************************************************

c       write(lupro,*)'myid=',myid
c       write(lupro,*)'numprocs==',numprocs
c       write(lupro,*)'parallel==',parallel

c     **********************************************************
c     DATEN EINLESEN:      

c     nlapp_el.....Gesamtanzahl Halo-Elemente 
c     nlapp_kn.....Gesamtanzahl Halo-Knoten        
c     nkern_max....Maximum von npoin_ges und nelem_ges

      nlapp_el=0
      nlapp_kn=0
      nkern_max=0
      if (covise_run.eq.0) then 
          CALL GEO_DIM(geo_name,geo_pfad,nkern_max,nlapp_el,nlapp_kn)
      else
         write (*,*) 'COV: Setting geometry dimensions ...'
         npoin=nump
         nelem=nume
         npoin_max=npoin
         nelem_max=nelem
         npoin_ges=npoin
         nelem_ges=nelem
         knmax_num=npoin
         elmax_num=nelem
      endif

c     Hilfsfelder fuer Knoten- und Element-Numerierung:
      CALL ALLOC_FINE(ifine,icoord_zeig,knmax_num)
      CALL ALLOC_FINE(ifine,ilnods_zeig,elmax_num)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

      IF (parti_rbe) THEN
         CALL RBE_DIM(rbe_name,isp(icoord_zeig),
     *                ndisp_zeil,ndisp_spalt,
     *                covise_run,
     *                numwall, numbal,
     *                numdir, coldir,
     *                dirindex, dirval,
     *                numpress)
      ELSE
         nwand=0
         npres=0
         nsyme=0
         nzykl=0
         nconv=0
         ntemp=0
         ndisp_zeil=0
      ENDIF
c      write(*,*) '***************************'
c      write(*,*) 'ndisp_zeil =', ndisp_zeil
c      write(*,*) 'ndisp_spalt=', ndisp_spalt
c      write(*,*) '***************************'
     

c     Dimensionen der Geometrie:    
      npoin_max=npoin
      nelem_max=nelem

c     Dimensionen der Elementrandbedingungen:            
      nelra_max=(nwand+npres+nsyme+nzykl+nconv)

c     Anzahl an Elementrandbedingungstypen:
      ntyp_elra=5

c     Dimensionen der Knotenrandbedingungen:
      nknra_max=(ndisp_zeil+ntemp)

c     Anzahl an Knotenrandbedingungstypen:
      ntyp_knra=2

c     Maximale Werteanzahl pro Knotenrandbedingung:
      nwer_knra=MAX(ndisp_spalt,1)

c     Adressrechnung fuer Partitionsfelder:
      CALL ALLOC_ENDE(iende,iknpar,npoin_max)
      CALL ALLOC_ENDE(iende,ielpar,nelem_max)

c     Adress-Rechnung fuer Geometrie:                
      CALL ALLOC_ENDE(iende,icoord,npoin_max*ncd)
      CALL ALLOC_ENDE(iende,icoord_num,npoin_max)
      CALL ALLOC_ENDE(iende,icoord_mod,npoin_max)
      CALL ALLOC_ENDE(iende,ilnods,nelem_max*nkd)
      CALL ALLOC_ENDE(iende,ilnods_num,nelem_max)
      CALL ALLOC_ENDE(iende,ilnods_mod,nelem_max)

c     Adress-Rechnung fuer Elementrandbedingungen:
      CALL ALLOC_ENDE(iende,ielra_kno,nelra_max*nrbknie)
      CALL ALLOC_ENDE(iende,ielra_ele,nelra_max)
      CALL ALLOC_ENDE(iende,ielra_num,nelra_max)
      CALL ALLOC_ENDE(iende,ielra_wer,nelra_max)
      CALL ALLOC_ENDE(iende,ielra_adr,ntyp_elra+1)

c     Adress-Rechnung fuer Knotenrandbedingungen:       
      CALL ALLOC_ENDE(iende,iknra_kno,nknra_max)
      CALL ALLOC_ENDE(iende,iknra_wer,nknra_max*nwer_knra)
      CALL ALLOC_ENDE(iende,iknra_mat,nknra_max*nwer_knra)
      CALL ALLOC_ENDE(iende,iknra_adr,ntyp_knra+1)


c     Hilfsfeld zum Einlesen:
      CALL ALLOC_FINE(ifine,iint1_help,npoin)
      CALL ALLOC_FINE(ifine,iint2_help,npoin)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

      if (myid.eq.0) write(6,*)'                   '
      if (myid.eq.0) write(6,*)'GEOMETRIE EINLESEN '

      IF (parti_geo) THEN

c        Einlesen der nicht partitionierten Geometrie:
         if (covise_run.eq.0) then
            CALL GEO_LES(isp(icoord),isp(icoord_num),isp(icoord_mod),
     *                isp(ilnods),isp(ilnods_num),isp(ilnods_mod),
     *                isp(icoord_zeig),isp(ilnods_zeig),
     *                geo_name)
         else
            write (*,*) 'GEO_LES_COV(): copying geometry data'
            CALL GEO_LES_COV(isp(icoord),isp(icoord_num),
     *                isp(icoord_mod),
     *                isp(ilnods),isp(ilnods_num),isp(ilnods_mod),
     *                isp(icoord_zeig),isp(ilnods_zeig),
     *                ellist,px,py,pz,
     *                colnode, nodeinfo, colelem, eleminfo)
         endif
           

      ELSE IF (parti_les) THEN

c        Einlesen der partitionierten Geometrie:
         CALL ALLOC_ENDE(iende,ikern_kn,nkern_max)                
         CALL ALLOC_ENDE(iende,ikern_el,nkern_max)                

         CALL ALLOC_ENDE(iende,ikern_kn_adr,ngebiet+1)                
         CALL ALLOC_ENDE(iende,ikern_el_adr,ngebiet+1)                
         CALL ALLOC_ENDE(iende,ilapp_kn_adr,ngebiet+1)                
         CALL ALLOC_ENDE(iende,ilapp_el_adr,ngebiet+1)                

         CALL ALLOC_ENDE(iende,ilapp_kn,nlapp_kn)
         CALL ALLOC_ENDE(iende,ilapp_el,nlapp_el)

         CALL ALLOC_ENDE(iende,ilapp_kn_proz,nlapp_kn) 
         CALL ALLOC_ENDE(iende,ilapp_el_proz,nlapp_el) 
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         CALL PAR_LES(isp(icoord),isp(icoord_num),isp(icoord_zeig),
     *                isp(ilnods),isp(ilnods_num),isp(ilnods_zeig),
     *                isp(ikern_kn),isp(ikern_kn_adr),
     *                isp(ilapp_kn),isp(ilapp_kn_adr),
     *                isp(ilapp_kn_proz),
     *                isp(ikern_el),isp(ikern_el_adr),
     *                isp(ilapp_el),isp(ilapp_el_adr),
     *                isp(ilapp_el_proz),
     *                nkern_max,nlapp_el,nlapp_kn,
     *                isp(iknpar),isp(ielpar),
     *                geo_pfad)

      ENDIF

      if (myid.eq.0) write(6,*)'GEOMETRIE EINGELESEN '
      if (myid.eq.0) write(6,*)'                     '

      IF (parti_rbe) THEN
         if (myid.eq.0) write(6,*)'RANDBEDINGUNGEN  EINLESEN '

         if (covise_run.eq.0) then
             CALL RBE_LES(isp(ielra_kno),isp(ielra_ele),
     *                isp(ielra_num),
     *                isp(ielra_wer),isp(ielra_adr),
     *                isp(iknra_kno),isp(iknra_mat),
     *                isp(iknra_wer),isp(iknra_adr),
     *                isp(icoord_zeig),isp(ilnods_zeig),
     *                isp(ilnods),isp(icoord_num),
     *                isp(iint1_help),rbe_name)
         else
             write (*,*) 'RBE_LES_COV(): copying boundary conditions'
             CALL RBE_LES_COV(isp(ielra_kno),isp(ielra_ele),
     *                isp(ielra_num),
     *                isp(ielra_wer),isp(ielra_adr),
     *                isp(iknra_kno),isp(iknra_mat),
     *                isp(iknra_wer),isp(iknra_adr),
     *                isp(icoord_zeig),isp(ilnods_zeig),
     *                isp(ilnods),isp(icoord_num),
     *                isp(iint1_help),
     *                coldir,
     *                numdir, dirindex, dirval,
     *                colwall, numwall, wall,
     *                colbal, numbal, balance,
     *                colpress, numpress, press)
c fl     *                pressval)
         endif
c
      
         if (myid.eq.0) write(6,*)'RANDBEDINGUNGEN  EINGELESEN '
      ENDIF

      CALL DEALLOC_ALLE(ifine,lmax)             
c     **********************************************************


c     **********************************************************
c     AUSGABE DER DIMENSIONEN:

      comment='DIMENSIONEN NACH DEM EINLESEN:         '
      CALL PRINT_DIM(comment,isp(iknra_adr),isp(ielra_adr))
c     **********************************************************


c     **********************************************************
c     BERECHNUNG DES KNOTENGRAPHEN:

c     write(6,*)'iende         =',iende         
c     write(6,*)'ifine         =',ifine         

      IF (parti_geo) THEN
         speich_mom=iende+(lmax-ifine)

         nfrei=ifine-iende
         CALL GRAPH_ENDE(isp(ilnods),isp(ilnods_num),isp(icoord_num),
     *                   ifine,iende,
     *                   ikomp_e,ikomp_d,redu_graph,
     *                   grad_max,grad_min,
     *                   isp(iende),nfrei,speich_max_sub)

         speich_max=MAX(speich_max,speich_mom+speich_max_sub)
      ENDIF

c     write(6,*)'speich_mom    =',speich_mom
c     write(6,*)'speich_max_sub=',speich_max_sub
c     write(6,*)'speich_max    =',speich_max
c     write(6,*)'iende         =',iende         
c     write(6,*)'ifine         =',ifine         
c     write(6,*)'nfrei         =',nfrei         
      CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK DER SPEICHERBELEGUNG:

      do 444 iii=1,2
         if (iii.eq.1) then
           lu=6
         else if (iii.eq.2) then
           lu=lupro
         endif
         write(lu,*)'                                 '
         write(lu,777) text_1
         write(lu,*)'SPEICHERBELEGUNG IM FORTRAN-TEIL:'
         write(lu,*)'--------------------------------'
         write(lu,*)'                                 '
         write(lu,*)'Dimensioniert             :',lmax
         write(lu,*)'Maximal notwendig         :',speich_max
         write(lu,*)'Anteil der Matrix-Struktur:',nl_kompakt+npoin+1
         write(lu,*)'Anzahl Matrixeintraege    :',nl_kompakt
         write(lu,*)'Maximaler Grad der Matrix :',grad_max
         write(lu,*)'Minimaler Grad der Matrix :',grad_min
         write(lu,*)'Anzahl Knoten             :',npoin_ges
         write(lu,*)'Anzahl Elemente           :',nelem_ges 
         write(lu,777) text_1
 444  continue
 777  format(1x,A70)
c     *****************************************************************


c     *****************************************************************
c     ZERLEGUNG MIT METIS:

      IF (parti_geo) THEN
         nparti=nzer

         CALL ALLOC_FINE(ifine,iparti,npoin_max*nparti)
         CALL ALLOC_FINE(ifine,iint1_help,npoin_max)
         CALL ALLOC_FINE(ifine,iint2_help,npoin_max)
         CALL ALLOC_FINE(ifine,iint3_help,npoin_max)

         CALL ALLOC_FINE(ifine,igeb1_help,nparti*ngebiet)
         CALL ALLOC_FINE(ifine,igeb2_help,nparti*ngebiet)
         CALL ALLOC_FINE(ifine,igeb3_help,nparti*ngebiet)
         CALL ALLOC_FINE(ifine,igeb4_help,nparti*ngebiet)
         CALL ALLOC_FINE(ifine,igeb5_help,nparti*ngebiet)
         CALL ALLOC_FINE(ifine,igeb6_help,nparti*ngebiet)

         CALL ALLOC_FINE(ifine,ipar1_help,nparti*ndrei)
         CALL ALLOC_FINE(ifine,ipar2_help,nparti*ndrei)
         CALL ALLOC_FINE(ifine,ipar3_help,nparti*ndrei)
         CALL ALLOC_FINE(ifine,ipar4_help,nparti*ndrei)
         CALL ALLOC_FINE(ifine,ipar5_help,nparti*ndrei)
         CALL ALLOC_FINE(ifine,ipar6_help,nparti*ndrei)
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         CALL METI(isp(ikomp_e),isp(ikomp_d),isp(ilnods),
     *             isp(icoord_num),isp(iknpar),
     *             isp(iparti),nparti,
     *             zer_zeig,ndat_max,
     *             isp(iint1_help),isp(iint2_help),isp(iint3_help),
     *             isp(igeb1_help),isp(igeb2_help),isp(igeb3_help),
     *             isp(igeb4_help),isp(igeb5_help),isp(igeb6_help),
     *             isp(ipar1_help),isp(ipar2_help),isp(ipar3_help),
     *             isp(ipar4_help),isp(ipar5_help),isp(ipar6_help))
      ENDIF

      CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************


c     *****************************************************************
c     BESTIMMUNG DER KERN-, HALO- UND DOPPLAPP-DATEN: 


      IF (parti_geo) THEN

         CALL ALLOC_FINE(ifine,iint1_help,npoin_max)
         CALL ALLOC_FINE(ifine,iint2_help,npoin_max)
         CALL ALLOC_FINE(ifine,iint3_help,npoin_max)
         CALL ALLOC_FINE(ifine,igeb1_help,ngebiet+1)
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         nkern_max=MAX(npoin_max,nelem_max)
         CALL ALLOC_ENDE(iende,ikern_kn,nkern_max)                
         CALL ALLOC_ENDE(iende,ikern_el,nkern_max)                
         CALL ALLOC_ENDE(iende,ikern_kn_adr,ngebiet+1)                
         CALL ALLOC_ENDE(iende,ikern_el_adr,ngebiet+1)                
         CALL ALLOC_ENDE(iende,ilapp_kn_adr,ngebiet+1)                
         CALL ALLOC_ENDE(iende,ilapp_el_adr,ngebiet+1)                
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         CALL KERN_DATA(isp(iknpar),isp(icoord_num),isp(ilnods),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ikern_el),isp(ikern_el_adr),nkern_max)

         nlapp_el=1
         CALL ALLOC_FINE(ifine,ilapp_el,nlapp_el)
         CALL ALLOC_FINE(ifine,ilapp_el_proz,nlapp_el) 
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         schreiben=.false.
         CALL HALO_ELEM(isp(iknpar),isp(ielpar),isp(icoord_num),
     *                  isp(ilnods),isp(ilnods_num),
     *                  isp(ikomp_e),isp(ikomp_d),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(ilapp_el_proz),
     *                  nkern_max,nlapp_el,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(igeb1_help),schreiben)

         CALL ALLOC_ENDE(iende,ilapp_el,nlapp_el)
         CALL ALLOC_ENDE(iende,ilapp_el_proz,nlapp_el) 
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         schreiben=.true.
         CALL HALO_ELEM(isp(iknpar),isp(ielpar),isp(icoord_num),
     *                  isp(ilnods),isp(ilnods_num),
     *                  isp(ikomp_e),isp(ikomp_d),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(ilapp_el_proz),
     *                  nkern_max,nlapp_el,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(igeb1_help),schreiben)

         nlapp_kn=1
         CALL ALLOC_FINE(ifine,ilapp_kn,nlapp_kn)
         CALL ALLOC_FINE(ifine,ilapp_kn_proz,nlapp_kn) 
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         schreiben=.false.
         CALL HALO_NODE(isp(iknpar),isp(ilnods),isp(ilnods_num),
     *                  isp(icoord_num),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(ilapp_kn_proz),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  nkern_max,nlapp_el,nlapp_kn,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(iint3_help),
     *                  schreiben)

         CALL ALLOC_ENDE(iende,ilapp_kn,nlapp_kn)
         CALL ALLOC_ENDE(iende,ilapp_kn_proz,nlapp_kn) 
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         schreiben=.true. 
         CALL HALO_NODE(isp(iknpar),isp(ilnods),isp(ilnods_num),
     *                  isp(icoord_num),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(ilapp_kn_proz),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  nkern_max,nlapp_el,nlapp_kn,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(iint3_help),
     *                  schreiben)

         if(dopplapp) then

           CALL ALLOC_ENDE(iende,idopp_kn_adr,ngebiet+1)                
           CALL ALLOC_ENDE(iende,idopp_el_adr,ngebiet+1)                
           ndopp_el=1
           CALL ALLOC_FINE(ifine,idopp_el,ndopp_el)
           CALL ALLOC_FINE(ifine,idopp_el_proz,ndopp_el) 
           CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)


           schreiben=.false.
           CALL DOPP_ELEM(isp(iknpar),isp(ielpar),isp(icoord_num),
     *                  isp(ilnods),isp(ilnods_num),
     *                  isp(ikomp_e),isp(ikomp_d),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(ilapp_el_proz),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  isp(idopp_el_proz),
     *                  nlapp_kn,nlapp_el,ndopp_el,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(iint3_help),
     *                  isp(igeb1_help),schreiben)

           CALL ALLOC_ENDE(iende,idopp_el,ndopp_el)
           CALL ALLOC_ENDE(iende,idopp_el_proz,ndopp_el) 
           CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

           schreiben=.true. 
           CALL DOPP_ELEM(isp(iknpar),isp(ielpar),isp(icoord_num),
     *                  isp(ilnods),isp(ilnods_num),
     *                  isp(ikomp_e),isp(ikomp_d),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(ilapp_el_proz),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  isp(idopp_el_proz),
     *                  nlapp_kn,nlapp_el,ndopp_el,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(iint3_help),
     *                  isp(igeb1_help),schreiben)
	   
           ndopp_kn=1
           CALL ALLOC_FINE(ifine,idopp_kn,ndopp_kn)
           CALL ALLOC_FINE(ifine,idopp_kn_proz,ndopp_kn) 
           CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

           schreiben=.false.
           CALL DOPP_NODE(isp(iknpar),isp(ilnods),isp(ilnods_num),
     *                  isp(icoord_num),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(idopp_kn),isp(idopp_kn_adr),
     *                  isp(idopp_kn_proz),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  nlapp_kn,nlapp_el,ndopp_el,ndopp_kn,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(iint3_help),
     *                  schreiben)

           CALL ALLOC_ENDE(iende,idopp_kn,ndopp_kn)
           CALL ALLOC_ENDE(iende,idopp_kn_proz,ndopp_kn) 
           CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

           schreiben=.true. 
           CALL DOPP_NODE(isp(iknpar),isp(ilnods),isp(ilnods_num),
     *                  isp(icoord_num),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(idopp_kn),isp(idopp_kn_adr),
     *                  isp(idopp_kn_proz),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  nlapp_kn,nlapp_el,ndopp_el,ndopp_kn,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(iint3_help),
     *                  schreiben)


         else
	 
	   ndopp_el=0
	   ndopp_kn=0
           CALL ALLOC_ENDE(iende,idopp_kn_adr,0)                
           CALL ALLOC_ENDE(iende,idopp_kn,0)                
           CALL ALLOC_ENDE(iende,idopp_kn_proz,0)                
           CALL ALLOC_ENDE(iende,idopp_el_adr,0)                
           CALL ALLOC_ENDE(iende,idopp_el,0)
           CALL ALLOC_ENDE(iende,idopp_el_proz,0) 

	 endif


c        Kontrolle der Freiheitsgrade:
         CALL FREE_CHECK(isp(icoord_num),isp(ilnods),
     *                   isp(ikomp_e),isp(ikomp_d),
     *                   isp(ikern_kn),isp(ikern_kn_adr),
     *                   isp(ikern_el),isp(ikern_el_adr),
     *                   isp(ilapp_el),isp(ilapp_el_adr),
     *                   nkern_max,nlapp_el,nlapp_kn,
     *                   isp(iint1_help),isp(iint2_help),
     *                   isp(igeb1_help),isp(iknpar))

c        Kontrolle der Elementnachbarn waere hier schoen,
c        kommt aber erst nach GRAPH_ELE

      ENDIF

      CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************


c     *****************************************************************
c     BESTIMMUNG DES GEBIETSGRAPHEN:

      CALL ALLOC_ENDE(iende,igemat_adr,ngebiet+1)
      CALL ALLOC_ENDE(iende,ifarb_geb,ngebiet+1)
      CALL ALLOC_ENDE(iende,ifarb_adr,ngebiet+1)
      CALL ALLOC_ENDE(iende,ifarb_per,ngebiet+1)

      CALL ALLOC_FINE(ifine,igeb1_help,ngebiet+1)
      CALL ALLOC_FINE(ifine,igeb1_help,ngebiet+1)
      CALL ALLOC_FINE(ifine,igeb2_help,ngebiet+1)
      CALL ALLOC_FINE(ifine,igeb3_help,ngebiet+1)
      CALL ALLOC_FINE(ifine,igeb4_help,ngebiet+1)
      CALL ALLOC_FINE(ifine,igeb5_help,ngebiet+1)

      nl_gemat=0
      CALL ALLOC_FINE(ifine,igemat,nl_gemat)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)
      
      schreiben=.false.
      CALL GRAPH_GEB(isp(igemat),isp(igemat_adr),nl_gemat,
     *               isp(ilapp_kn_proz),isp(ilapp_kn_adr),nlapp_kn,
     *               isp(idopp_kn_proz),isp(idopp_kn_adr),ndopp_kn,
     *               isp(igeb1_help),isp(igeb2_help),dopplapp,
     *               schreiben)

      CALL ALLOC_ENDE(iende,igemat,nl_gemat)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)
      
      schreiben=.true. 
      CALL GRAPH_GEB(isp(igemat),isp(igemat_adr),nl_gemat,
     *               isp(ilapp_kn_proz),isp(ilapp_kn_adr),nlapp_kn,
     *               isp(idopp_kn_proz),isp(idopp_kn_adr),ndopp_kn,
     *               isp(igeb1_help),isp(igeb2_help),dopplapp,
     *               schreiben)

      write(6,*)'Routine COLOR_GEB ist auskommentiert '
      write(6,*)'Routine COLOR_GEB ist auskommentiert '

c     CALL COLOR_GEB(isp(igemat),isp(igemat_adr),nl_gemat,
c    *               isp(igeb1_help),isp(ifarb_per),
c    *               isp(igeb2_help),isp(igeb3_help),
c    *               isp(ifarb_geb),isp(igeb4_help),
c    *               isp(igeb5_help),isp(ifarb_adr),nfarb_geb)
c    *               
      
      CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK DER GEOMETRIE:

      CALL ALLOC_ENDE(iende,igeom_dim,ngebiet*ndat_max)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)


      IF (print_all.and.parti_geo) THEN

         CALL ALLOC_FINE(ifine,iint1_help,npoin_max)
         CALL ALLOC_FINE(ifine,iint2_help,npoin_max)
         CALL ALLOC_FINE(ifine,igeb1_help,ngebiet)
         CALL ALLOC_FINE(ifine,igeb2_help,ngebiet)
         CALL ALLOC_FINE(ifine,cov_coord,npoin_max*ncd)
         CALL ALLOC_FINE(ifine,cov_lnods,nelem_max*nkd)
         CALL ALLOC_FINE(ifine,cov_lnods_num,nelem_max)
         CALL ALLOC_FINE(ifine,cov_lnods_proz,nelem_max)
         CALL ALLOC_FINE(ifine,cov_coord_num,npoin_max)
         CALL ALLOC_FINE(ifine,cov_coord_joi,npoin_max)
         CALL ALLOC_FINE(ifine,cov_lnods_joi,nelem_max)
         CALL ALLOC_FINE(ifine,cov_coord_mod,npoin_max)
         CALL ALLOC_FINE(ifine,cov_lnods_mod,nelem_max)
         CALL ALLOC_FINE(ifine,cov_coord_proz,npoin_max)
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

c        Ausdruck der Geometrie:
         if (covise_run.eq.0) then
            CALL PRINT_GEO(isp(ilnods),isp(ilnods_num),isp(ilnods_mod),
     *                  isp(icoord),isp(icoord_num),isp(icoord_mod),
     *                  isp(ielpar),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(ilapp_kn_proz),
     *                  isp(idopp_kn),isp(idopp_kn_adr),
     *                  isp(idopp_kn_proz),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(ilapp_el_proz),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  isp(idopp_el_proz),
     *                  nkern_max,nlapp_el,nlapp_kn,
     *                  ndopp_el,ndopp_kn,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(igeb1_help),isp(igeb2_help),
     *                  isp(igeom_dim),ndat_max,geo_pfad,geo_name,
     *                  dopplapp)
         else
            CALL PRINT_GEO_COV(isp(ilnods),isp(ilnods_num),
     *                  isp(ilnods_mod),
     *                  isp(icoord),isp(icoord_num),isp(icoord_mod),
     *                  isp(ielpar),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(ilapp_kn_proz),
     *                  isp(idopp_kn),isp(idopp_kn_adr),
     *                  isp(idopp_kn_proz),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(ilapp_el_proz),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  isp(idopp_el_proz),
     *                  nkern_max,nlapp_el,nlapp_kn,
     *                  ndopp_el,ndopp_kn,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(igeb1_help),isp(igeb2_help),
     *                  isp(igeom_dim),ndat_max,geo_pfad,geo_name,
     *                  dopplapp,
     *                  covise_run,
     *                  write_files,
     *                  isp(cov_coord), isp(cov_lnods),
     *                  isp(cov_lnods_num),
     *                  isp(cov_lnods_proz), isp(cov_coord_num),
     *                  isp(cov_coord_joi), isp(cov_lnods_joi),
     *                  isp(cov_coord_mod), isp(cov_lnods_mod),
     *                  isp(cov_coord_proz))		 
         endif
      ENDIF

      CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************

c     *****************************************************************
c     AUSDRUCK DER ELEMENTPARTITION:

c     ( Wichtig fuer die Colorierung von Oberflaechen mit
c       Programm MESH_GEB. Die Routine PRINT_ELPAR schreibt den
c       File ELE.PAR der von Programm MESH_GEB eingelesen wird. )

c     IF (parti_les) THEN
c        CALL ALLOC_FINE(ifine,iint1_help,npoin_max)
c        CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)
c
c        Ausdruck der Elementpartition:
c        CALL PRINT_ELPAR(isp(ilnods_num),
c    *                    isp(ilapp_el),isp(ilapp_el_adr),
c    *                    isp(ilapp_el_proz),nlapp_el,
c    *                    isp(iint1_help),isp(ifarb_geb),geo_name)
c
c     ENDIF
c
c     CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK DER RANDBEDINGUNGEN:

      CALL ALLOC_ENDE(iende,irand_dim,ngebiet*ndat_max)
      CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

      IF (print_all.and.parti_rbe) THEN

         CALL ALLOC_FINE(ifine,iint1_help,npoin_max)
         CALL ALLOC_FINE(ifine,iint2_help,npoin_max)
         CALL ALLOC_FINE(ifine,cov_displ_kn,nrbpoi)
         CALL ALLOC_FINE(ifine,cov_displ_wert,nrbpoi)
         CALL ALLOC_FINE(ifine,cov_displ_typ,nrbpoi)
         CALL ALLOC_FINE(ifine,cov_wand_kn,nwand*nrbknie)
         CALL ALLOC_FINE(ifine,cov_wand_el,nwand)
         CALL ALLOC_FINE(ifine,cov_wand_num,nwand)
         CALL ALLOC_FINE(ifine,cov_syme_kn,nsyme*nrbknie)
         CALL ALLOC_FINE(ifine,cov_syme_el,nsyme)
         CALL ALLOC_FINE(ifine,cov_pres_kn,npres*nrbknie)
         CALL ALLOC_FINE(ifine,cov_pres_el,npres)
         CALL ALLOC_FINE(ifine,cov_pres_num,npres)
         CALL ALLOC_FINE(ifine,cov_conv_kn,nconv*nrbknie)
         CALL ALLOC_FINE(ifine,cov_conv_el,nconv)
         CALL ALLOC_FINE(ifine,cov_conv_num,nconv)  
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         if (covise_run.eq.0) then
            CALL PRINT_RBE(isp(ielra_kno),isp(ielra_ele),isp(ielra_num),
     *                  isp(ielra_wer),isp(ielra_adr),
     *                  isp(iknra_kno),isp(iknra_mat),
     *                  isp(iknra_wer),isp(iknra_adr),
     *                  isp(ilnods_num),isp(icoord_num),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(idopp_kn),isp(idopp_kn_adr),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  nkern_max,nlapp_el,nlapp_kn,
     *                  ndopp_el,ndopp_kn,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(irand_dim),ndat_max,rbe_pfad,rbe_name,
     *                  dopplapp)
         else
            CALL PRINT_RBE_COV(isp(ielra_kno),isp(ielra_ele),
     *                  isp(ielra_num),
     *                  isp(ielra_wer),isp(ielra_adr),
     *                  isp(iknra_kno),isp(iknra_mat),
     *                  isp(iknra_wer),isp(iknra_adr),
     *                  isp(ilnods_num),isp(icoord_num),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(idopp_kn),isp(idopp_kn_adr),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  nkern_max,nlapp_el,nlapp_kn,
     *                  ndopp_el,ndopp_kn,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(irand_dim),ndat_max,rbe_pfad,rbe_name,
     *                  dopplapp,
     *                  covise_run,
     *                  write_files,
     *                  isp(cov_displ_wert),isp(cov_displ_kn),
     *                  isp(cov_displ_typ),
     *                  isp(cov_wand_el),isp(cov_wand_kn),
     *                  isp(cov_wand_num),
     *                  isp(cov_syme_el),isp(cov_syme_kn),
     *                  isp(cov_pres_el),isp(cov_pres_kn),
     *                  isp(cov_pres_num),isp(cov_conv_el),
     *                  isp(cov_conv_kn), isp(cov_conv_num))		 
		 endif
	 
      ENDIF

      CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK DER DIMENSIONEN:
      
c     ( Die Felder geom_dim und rand_dim sind nur belegt wenn die 
c       Daten geschrieben werden )

      if (print_all) then
         comment='Dimensionen der partitionierten Geometrie:'
         CALL PRINT_PAR(comment,isp(igeom_dim),isp(irand_dim),ndat_max)
      endif
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK DER ANFANGSNAEHERUNG:

      IF (print_all.and.parti_erg) THEN

		 CALL ERG_DIM(erg_name,nsp_kn_erg,nsp_el_erg,nsp_za_erg)
   
         CALL ALLOC_FINE(ifine,ierg_kn,npoin_max*nsp_kn_erg)
         CALL ALLOC_FINE(ifine,ierg_el,nelem_max*nsp_el_erg)
         CALL ALLOC_FINE(ifine,ierg_za,npoin_max*nsp_za_erg)
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

		 CALL ERG_LES(isp(icoord_num),isp(ilnods_num),
     *                isp(ierg_kn),nsp_kn_erg,
     *                isp(ierg_el),nsp_el_erg,
     *                isp(ierg_za),nsp_za_erg,
     *                transi_erg,erg_name)

         if (write_files.eq.1) then
            CALL PRINT_ERG(isp(ilnods_num),isp(icoord_num),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(idopp_kn),isp(idopp_kn_adr),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  nkern_max,nlapp_el,nlapp_kn,
     *                  ndopp_el,ndopp_kn,
     *                  isp(ierg_kn),nsp_kn_erg,
     *                  isp(ierg_el),nsp_el_erg,
     *                  isp(ierg_za),nsp_za_erg,
     *                  transi_erg,erg_pfad,erg_name,
     *                  dopplapp)
         endif

      ENDIF
      IF (print_all.and.parti_mit) THEN

		 CALL MIT_DIM(mit_name,nsp_kn_erg,nsp_el_erg,nsp_za_erg)

         CALL ALLOC_FINE(ifine,ierg_kn,npoin_max*nsp_kn_erg)
         CALL ALLOC_FINE(ifine,ierg_el,nelem_max*nsp_el_erg)
         CALL ALLOC_FINE(ifine,ierg_za,npoin_max*nsp_za_erg)
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)


            CALL MIT_LES(isp(icoord_num),isp(ilnods_num),
     *                  isp(ierg_kn),nsp_kn_erg,
     *                  isp(ierg_el),nsp_el_erg,
     *                  isp(ierg_za),nsp_za_erg,
     *                  transi_erg,mit_name)

         CALL PRINT_MIT(isp(ilnods_num),isp(icoord_num),
     *                  isp(ikern_kn),isp(ikern_kn_adr),
     *                  isp(ilapp_kn),isp(ilapp_kn_adr),
     *                  isp(idopp_kn),isp(idopp_kn_adr),
     *                  isp(ikern_el),isp(ikern_el_adr),
     *                  isp(ilapp_el),isp(ilapp_el_adr),
     *                  isp(idopp_el),isp(idopp_el_adr),
     *                  nkern_max,nlapp_el,nlapp_kn,
     *                  ndopp_el,ndopp_kn,
     *                  isp(ierg_kn),nsp_kn_erg,
     *                  isp(ierg_el),nsp_el_erg,
     *                  isp(ierg_za),nsp_za_erg,
     *                  transi_erg,mit_pfad,mit_name,
     *                  dopplapp,geo_name,geo_pfad,
     *                  rbe_name,rbe_pfad)

      ENDIF

      CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************


c     *****************************************************************
c     BERECHNUNG DES ELEMENT-GRAPHEN:

      IF (ober_geom.or.dopplapp) THEN

         graph_all=.true. 
         graph_fla=.false.
 
         if (myid.eq.0)write(6,*)'                                    '
         if (myid.eq.0)write(6,*)'Berechnung des Elementgraphen  '
 
         nl_kelem=nelem*nkd 

         CALL ALLOC_FINE(ifine,iint1_help,npoin)
         CALL ALLOC_FINE(ifine,iint2_help,npoin)
         CALL ALLOC_FINE(ifine,iint3_help,npoin)
         CALL ALLOC_FINE(ifine,ikelem,nl_kelem)
         CALL ALLOC_FINE(ifine,ikelem_adr,npoin+1)

         nl_elmat=0 
         CALL ALLOC_FINE(ifine,ielmat,nl_elmat)

         CALL ALLOC_FINE(ifine,ielmat_adr,nelem+1)
         CALL ALLOC_FINE(ifine,ielmat_stu,nelem+1)
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)
 
c        Bestimmung der an den Knoten beteiligten Elemente:
         CALL KELE(isp(ilnods),isp(ilnods_num),isp(icoord_num),
     *             nelem,nelem_max,nkd,
     *             isp(ikelem),isp(ikelem_adr),nl_kelem,
     *             isp(iint1_help),npoin,myid,parallel)
      
         schreiben=.false.
         CALL GRAPH_ELE(isp(ilnods),isp(ielmat),isp(ielmat_adr),
     *                  isp(ielmat_stu),nl_elmat,
     *                  isp(ikelem),isp(ikelem_adr),nl_kelem,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(iint3_help),
     *                  graph_all,graph_fla,schreiben)

         if (nl_elmat.gt.nl_kompakt) then
            CALL ALLOC_FINE(ifine,ielmat,nl_elmat)
         else
            ielmat=ikomp_e 
         endif
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         schreiben=.true.
         CALL GRAPH_ELE(isp(ilnods),isp(ielmat),isp(ielmat_adr),
     *                  isp(ielmat_stu),nl_elmat,
     *                  isp(ikelem),isp(ikelem_adr),nl_kelem,
     *                  isp(iint1_help),isp(iint2_help),
     *                  isp(iint3_help),
     *                  graph_all,graph_fla,schreiben)

         if (myid.eq.0)write(6,*)'Berechnung des Elementgraphen beendet'
	 
	 if (dopplapp) then
c        Haben alle Kernelemente alle Nachbarelemente in ihrem Gebiet?
           CALL DOPP_CHECK(isp(ilnods_num),isp(ilnods),
     *                   isp(ielmat),isp(ielmat_adr),nl_elmat,
     *                   isp(ikern_kn),isp(ikern_kn_adr),
     *                   isp(ikern_el),isp(ikern_el_adr),
     *                   isp(ilapp_el),isp(ilapp_el_adr),
     *                   isp(ilapp_el_proz),
     *                   isp(idopp_el),isp(idopp_el_adr),
     *                   isp(idopp_el_proz),
     *                   nkern_max,nlapp_el,ndopp_el,
     *                   isp(ielpar),
     *                   isp(iint1_help),isp(iint2_help),
     *                   isp(igeb1_help))

	 endif

      ENDIF


      CALL DEALLOC_ALLE(ifine,lmax)             
c     *****************************************************************

c     *****************************************************************
c     AUSDRUCK DER OBERFLAECHEN-GEOMETRIE:

      IF (ober_geom) THEN

         if (myid.eq.0) write(6,*)'                                '
         if (myid.eq.0) write(6,*)'OBERFLAECHEN-GEOMETRIE BERECHNEN'

         ober_name_geo='OBER.GEO'
         ober_name_ses='OBER.SES'

         CALL ALLOC_FINE(ifine,iint1_help,npoin)
         CALL ALLOC_FINE(ifine,igeb1_help,ngebiet+1)
         CALL SPEICH_CHECK(iende,ifine,lmax,speich_max)

         nfrei=ifine-iende
         CALL OBER_VIEW(isp(ielpar),isp(ilnods),isp(ilnods_num),
     *                  isp(icoord),isp(icoord_num),
     *                  isp(ielmat),isp(ielmat_adr),isp(ielmat_stu),
     *                  nl_elmat,
     *                  isp(igemat),isp(igemat_adr),nl_gemat,
     *                  isp(ifarb_geb),isp(ifarb_adr),isp(ifarb_per),
     *                  nfarb_geb,
     *                  isp(iint1_help),isp(igeb1_help),
     *                  ober_name_geo,ober_name_ses,
     *                  isp(iende),nfrei,speich_max_sub)

            speich_mom=speich_max_sub+iende+(lmax-ifine)
            speich_max=MAX(speich_max,speich_mom)

         if (myid.eq.0) write(6,*)'OBERFLAECHEN-GEOMETRIE BERECHNET'

      ENDIF
c     *****************************************************************



c     ****************************************************************
c     PROTOKOLL UND BILDSCHIRMAUSDRUCK:

       if (parallel) then
        CALL MPI_REDUCE(speich_max,speich_max_all,1,MPI_INTEGER,
     *                  MPI_MAX,0,MPI_COMM_WORLD,ierr)
       else
        speich_max_all=speich_max
       endif

      IF (myid.eq.0) THEN

         do 555 iii=1,2
           if (iii.eq.1) then
             lu=6
           else if (iii.eq.2) then
             lu=lupro
           endif
           write(lu,*)'Gesamtanzahl Knoten  :',npoin_ges
           write(lu,*)'Gesamtanzahl Elemente:',nelem_ges
           write(lu,*)'                                 '
           write(lu,*)'Maximal benoetigter Speicher:',speich_max_all
           write(lu,*)'Dimensioniert               :',lmax  
           write(lu,*)'                         '
 555     continue

         comment='File geschrieben:'
         call char_druck(comment,prot_file,6)
         close(lupro)

         write(6,*) '                         '
         write(6,*) '*******************************'
         write(6,*) '******* PROGRAMM-ENDE *********'
         write(6,*) '*******************************'
         write(6,*) '                         '


      ENDIF                  

 666  format(2(A23,1x,i8,2x))
c     ****************************************************************
      close(lupar)

      CALL MPI_FINALIZE(ierr)
c     stop
      end





