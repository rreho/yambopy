set style data dots
set nokey
set xrange [0: 4.73667]
set yrange [-20.19436 : 24.26138]
set arrow from  1.08988, -20.19436 to  1.08988,  24.26138 nohead
set arrow from  2.42470, -20.19436 to  2.42470,  24.26138 nohead
set arrow from  3.96601, -20.19436 to  3.96601,  24.26138 nohead
set xtics ("W"  0.00000,"L"  1.08988,"?~S"  2.42470,"X"  3.96601,"W"  4.73667)
 plot "LiF_band.dat"
