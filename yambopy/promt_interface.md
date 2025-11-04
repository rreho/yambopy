We must work on an interface between Yambo, wannier 90 and a third code written in c++.
I want to use yambopy, this library, as an interface between the two codes as I already have several routines in `yambopy/wannier/` that can be used for the IO of this purpose.

The main idea is to
1) Understand well the current library, especially the wannierio and the YamboWFDB and em1s db classes.
2) Implement the equations that I show to you
3) Always ask me for revision of routines and new objects
4) Write simple code without being pedantic on the tests for now. We will do it later.

The first step is to implement the following equation
$
\rho_{\mathbf{q}}^{n \mathbf{R}}(\mathbf{r})=\frac{1}{N_k} e^{-i \mathbf{q} \cdot \mathbf{R}} \sum_k \tilde{u}_{n \mathbf{k}}^*(\mathbf{r}) \tilde{u}_{n \mathbf{k}+\mathbf{q}}(\mathbf{r})
$

where

$
\tilde{u}_{n \mathbf{k}}(\mathbf{r})=\sum_m U_{m n}(\mathbf{k}) u_{m \mathbf{k}}(\mathbf{r})
$
wih U mean to be read from wannier90.
We will let wannier90 produce most of the files that are need to IO while we focus on the computation of equation for \rho.

Please note that I belive it's convenient to write the equation using simple kmesh grid. Have a look at how
YamboWFDB works when looking for k and k+q points outside the BZ. It should give us some ideas about how to proceed with the implementation. Optionally, we can also use the classes in `yambopy/wann_kgrids` but I would like to have a general function first and then have the possibility to pass the class with wann kpoints.

Once $\rho$ is computed we must compute the bare and screened potential according to this equations:
$
\begin{aligned}
& V_{\mathbf{q}}^{n \mathbf{R}}(\mathbf{r})=\int d^3 r^{\prime} f_{H x c}\left(\mathbf{r}, \mathbf{r}^{\prime}\right) \rho_{\mathbf{q}}^{n \mathbf{R}}\left(\mathbf{r}^{\prime}\right) \\
& V_{n m}(\mathbf{R})=\sum_{\mathbf{q}} \int d^3 r \rho_{\mathbf{q}}^{n \mathbf{R} *}(\mathbf{r}) V_{\mathbf{q}}^{m \mathbf{0}}(\mathbf{r})
\end{aligned}
$

$
\begin{gathered}
\Delta_{\mathbf{q}}^{n \mathbf{0}} \rho(\mathbf{r})=\int d^3 r^{\prime} \chi_{\mathbf{q}}\left(\mathbf{r}, \mathbf{r}^{\prime}\right) V_{\mathbf{q}}^{n \mathbf{0}}\left(\mathbf{r}^{\prime}\right) \\
W_{n m}(\mathbf{R})=V_{n m}(\mathbf{R})+\sum_{\mathbf{q}} \int d^3 r V_{\mathbf{q}}^{n \mathbf{R}^*}(\mathbf{r}) \Delta_{\mathbf{q}}^{n \mathbf{0}} \rho(\mathbf{r})
\end{gathered}
$

However, we must perform the integral in G-space and not in real space. Unless you suggest otherwise. For the bare potential we must substitute $f_{Hxc}$ with bare Coulomb potential in `ndb.em1s` and for the screened one we must substitute $f_{Hxc}$ with the screened Coulomb potential which should be also available in `ndb.em1s`. or it might be possible to compute it from there.
Anyway, for now the best approach is to implement a general subroutines and I'll come with the data.

For testing, files are in ~/workQE/Projects/Monolayers/MoS2/wannier90/wannier-6x6x1/mos2.save



We must make some edits to the WannierInputInterface