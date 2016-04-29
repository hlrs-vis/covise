C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE GRAPH_GEB(gemat,gemat_adr,nl_gemat, 
     *                     lapp_kn_proz,lapp_kn_adr,nlapp_kn,
     *                     dopp_kn_proz,dopp_kn_adr,ndopp_kn,
     *                     zeig,folg,dopplapp,schreiben)
  
      implicit none     

      include 'common.zer'

      integer   i,geb_num,igeb,nnn,luerr,
     *          icom,lentb,nanz

      character*80 comment,zeil_1,zeil_2

      logical   schreiben,dopplapp

      integer   gemat,gemat_adr,nl_gemat, 
     *          lapp_kn_proz,lapp_kn_adr,nlapp_kn,
     *          dopp_kn_proz,dopp_kn_adr,ndopp_kn,zeig,folg

      dimension gemat(nl_gemat),gemat_adr(ngebiet+1),
     *          zeig(ngebiet),folg(ngebiet),
     *          lapp_kn_proz(nlapp_kn),lapp_kn_adr(ngebiet+1),
     *          dopp_kn_proz(ndopp_kn),dopp_kn_adr(ngebiet+1)
c     *****************************************************************


c     *****************************************************************
c     BERECHNEN DES GEBIETSGRAPHEN:                                  

      do 701 i=1,ngebiet
         folg(i)=0
         zeig(i)=0
 701  continue

      gemat_adr(1)=1
      nanz=0
      do 700 igeb=1,ngebiet
         nnn=0 
        
c        nnn=1
c        folg(nnn)=igeb
c        zeig(igeb)=nnn 

         do 710 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
            geb_num=lapp_kn_proz(i)
            IF (geb_num.ne.igeb) THEN
               if (zeig(geb_num).eq.0) then
                  nnn=nnn+1 
                  folg(nnn)=geb_num
                  zeig(geb_num)=nnn
               endif
            ENDIF
 710     continue

         if (dopplapp) then
	    do 720 i=dopp_kn_adr(igeb),dopp_kn_adr(igeb+1)-1
               geb_num=dopp_kn_proz(i)
               IF (geb_num.ne.igeb) THEN
                  if (zeig(geb_num).eq.0) then
                     nnn=nnn+1 
                     folg(nnn)=geb_num
                     zeig(geb_num)=nnn
                  endif
               ELSE
                  call erro_init(myid,parallel,luerr)
                  write(luerr,*)'Fehler in Routine GRAPH_GEB'
                  write(luerr,*)'Nicht alle Dopplappknoten '
                  write(luerr,*)'liegen auf anderen Prozessoren '
                  write(luerr,*)'Gebiet: ',geb_num
c                  write(luerr,*)'Knoten ',coord_num(dopp_kn(i))
c                  write(luerr,*)'ist auf Prozessor ',igeb
c                  write(luerr,*)'Dopplappknoten und '
c                  write(luerr,*)'hat dopp_kn_proz(i)=',geb_num
                  call erro_ende(myid,parallel,luerr)
	       ENDIF
 720        continue
         endif

         gemat_adr(igeb+1)=gemat_adr(igeb)+nnn

c        Schreiben des Graphen:
         do 730 i=1,nnn
            nanz=nanz+1
            if (schreiben) then
               gemat(nanz)=folg(i)
            endif
            zeig(folg(i))=0
 730     continue

 700  continue
c     *****************************************************************


c     *****************************************************************
c     DIMENSIONSKONTROLLE:

      IF (schreiben) THEN
         nnn=gemat_adr(ngebiet+1)-gemat_adr(1)
         if (nnn.ne.nl_gemat.or.nanz.ne.nl_gemat) then   
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine GRAPH_GEB'
            write(luerr,*)'Die zuvor bestimmte Groesse des  '
            write(luerr,*)'Gebietsgraphen stimmt mit der    '
            write(luerr,*)'tatsaechlichen Dimension nicht ueberein.'
            write(luerr,*)'Bestimmte      Dimension:',nl_gemat
            write(luerr,*)'Tatsaechliche  Dimension:',nanz
            write(luerr,*)'nanz                    :',nanz
            call erro_ende(myid,parallel,luerr)
         endif
      ELSE
         nl_gemat=nanz
      ENDIF
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK DER GEBIETSGRAPHEN:

      IF (schreiben) THEN

         do 120 i=2,80 
            zeil_1(i-1:i)='*'
            zeil_2(i-1:i)='-'
 120     continue
 777     format(1x,A70)
 666     format(1x,A)
 555     format(1x,i3,3x,i2,5x,100(i3,1x))

         comment='Gebietsgraph der ausgewaehlten Zerlegung'
         icom=lentb(comment)
c        write(lupro,*)                   
c        write(lupro,777) zeil_1       
c        write(lupro,666) comment(1:icom)
c        write(lupro,666) zeil_2(1:icom)       
c
c        write(lupro,*)'Nr.   Anzahl                     Graph  '
c        do 600 igeb=1,ngebiet
c           anz=gemat_adr(igeb+1)-gemat_adr(igeb)
c           write(lupro,555) igeb,anz,(gemat(k),
c    *                      k=gemat_adr(igeb),gemat_adr(igeb+1)-1)
c600     continue
c
c        write(lupro,777) zeil_1       

      ENDIF
c     *****************************************************************

      return
      end
