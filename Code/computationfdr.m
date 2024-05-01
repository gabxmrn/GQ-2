function[fdr,fdrneg,fdrpos]=computationfdr(pvec,signpvec,pnul,threshold)
%pvec=vector of pvalues
%signpvec=sign of the coefficients associated with the pvalues (1 if
%positive and -1 if negative)
%pnul is the null proportion
%threshold is the rejection threshold
%computation of the fdr
n=size(pvec,1);
pr=sum(pvec<threshold)/n;
fdr=pnul*threshold/pr;

%computation of the fdr negative side
selecn=find(signpvec<0);
pvecneg=pvec(selecn,1);
prn=sum(pvecneg<threshold)/n;
fdrneg=(pnul*threshold/2)/prn;

%computation of the fdr positive side
selecp=find(signpvec>0);
pvecpos=pvec(selecp,1);
prp=sum(pvecneg<threshold)/n;
fdrpos=(pnul*threshold/2)/prp;
