# Time Dependent Hartree
The time-dependent Hartree potential is defined as:

$
\begin{aligned} \label{eq:tdhpot}
V_{n_1 n_2}^H(t) & =\left\langle n_1\right| \int \mathrm{d}^3 r^{\prime} \frac{\rho\left(\mathbf{r}^{\prime} t\right)}{\left|\mathbf{r}-\mathbf{r}^{\prime}\right|}\left|n_2\right\rangle= \\
& =\int \mathrm{d}^3 r \mathrm{~d}^3 r^{\prime} \varphi_{n_1}^*(\mathbf{r}) \frac{\rho\left(\mathbf{r}^{\prime} t\right)}{\left|\mathbf{r}-\mathbf{r}^{\prime}\right|} \varphi_{n_2}(\mathbf{r})
\end{aligned}
$

We inser the expansion of the time-dependent density $\rho_{n m \mathbf{k}}(t)=\left\langle\psi_{n \mathbf{k}}^{K S}\right| \rho\left(r, r^{\prime} ; t\right)\left|\psi_{m \mathbf{k}}^{K S}\right\rangle$ and obtain

$
\begin{aligned} \label{eq:tdhpot2}
V_{n_1 n_2}^H(t) & =2 \sum_{l_1 l_2} \rho_{l_1 l_2}(t) \int \mathrm{d}^3 r \mathrm{~d}^3 r^{\prime} \varphi_{n_1}^*(\mathbf{r}) \varphi_{l_1}^*\left(\mathbf{r}^{\prime}\right) \varphi_{l_2}\left(\mathbf{r}^{\prime}\right) \varphi_{n_2}(\mathbf{r}) \frac{1}{\left|\mathbf{r}-\mathbf{r}^{\prime}\right|} \\
& \equiv 2 \sum_{l_1 l_2} \rho_{l_1 l_2}(t) V_{n_1 n_2}
\end{aligned}
$

The variation of the time dependent Hartree potential with respect to the density is:
$
\begin{aligned}
\frac{\delta V_{n_1 n_2}^H(t)}{\delta \rho_{m_3 m_4}(\bar{t})}=2\left[V^{q_v=0}\right]_{\substack{n_1 n_2 \\ m_3 m_4}} \delta(t-\bar{t})
\end{aligned}
$

# Time Dependent Exchange Potential
The time-dependent exchange potential is defined as:

$
\begin{aligned}
\Sigma^{\mathrm{SEX}}\left(\mathbf{r} t, \mathbf{r}^{\prime} t\right) & =\mathrm{i} G^0\left(\mathbf{r} t, \mathbf{r}^{\prime} t\right) W\left(\mathbf{r}, \mathbf{r}^{\prime}\right) \\
& =-\rho\left(\mathbf{r r}^{\prime}, t\right) W\left(\mathbf{r}, \mathbf{r}^{\prime}\right)
\end{aligned}
$

$
\Sigma_{n_1 n_2}^{\mathrm{SEX}}(t)=-\sum_{l_1 l_2} \rho_{l_1 l_2}(t) \int \mathrm{d}^3 r \mathrm{~d}^3 r^{\prime} \varphi_{l_1}^*\left(\mathbf{r}^{\prime}\right) \varphi_{l_2}(\mathbf{r}) \varphi_{n_1}^*(\mathbf{r}) \varphi_{n_2}\left(\mathbf{r}^{\prime}\right) W\left(\mathbf{r}, \mathbf{r}^{\prime}\right)
$

$
\Sigma_{n_1 n_2}^{\mathrm{SEX}}(t)=-\sum_{l_1 l_2} \rho_{l_1 l_2}(t) W_{\substack{n_1 l_2 \\ l_1 n_2}}
$

And the variation with respect to the density is given by:

$
\frac{\delta \sum_{n_1 n_2}^{\mathrm{SEX}}(t)}{\delta \rho_{m_3 m_4}(\bar{t})}=-W_{m_1 n_2}^{n_1 m_2} \delta(t-\bar{t})
$
where $\delta W/\delta \rho$ is neglected.