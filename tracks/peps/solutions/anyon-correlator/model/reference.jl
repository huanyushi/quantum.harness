# arXiv:1012.1740v2 supplemental Eq.(1), h_y=0, J=1 (paper: J=1/2).
# Each tuple is (power_x,power_z,coefficient); analytic derivatives below.
const SERIES_TERMS = [
    (0,0,-1//1), (2,0,-1//4),(0,2,-1//4),
    (4,0,-15//64),(0,4,-15//64),(2,2,1//32),
    (6,0,-147//256),(0,6,-147//256),(4,2,113//1024),(2,4,113//1024),
    (8,0,-18003//8192),(0,8,-18003//8192),(6,2,6685//16384),(2,6,6685//16384),(4,4,20869//49152),
    (10,0,-5420775//524288),(0,10,-5420775//524288),
    (8,2,39524033//18874368),(2,8,39524033//18874368),
    (6,4,5650925//3538944),(4,6,5650925//3538944)]
function reference_values(hx,hz;order=10)
    e=0.0;mx=0.0;mz=0.0
    for (p,q,c) in SERIES_TERMS
        p+q>order && continue
        e+=c*hx^p*hz^q
        p>0 && (mx-=p*c*hx^(p-1)*hz^q)
        q>0 && (mz-=q*c*hx^p*hz^(q-1))
    end
    # This determines A+B at J_e=J_m=1; individual A,B only on axes or diagonal.
    sumab=-2(e+hx*mx+hz*mz)
    astar=hx==0 ? sumab-1 : hz==0 ? 1.0 : hx==hz ? sumab/2 : nothing
    bplaq=hz==0 ? sumab-1 : hx==0 ? 1.0 : hx==hz ? sumab/2 : nothing
    return (;energy=e,mx,mz,astar,bplaq)
end
