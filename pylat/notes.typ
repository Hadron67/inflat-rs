#set math.equation(numbering: "(1)")

#let pd(a, b) = $ frac(partial #a, partial #b) $
#let vd(a, b) = $ frac(delta #a, delta #b) $

= Proca-Axion action
The Lagrangian is given by
$
    cal(L) = R - 1/2 nabla_mu phi.alt nabla^mu phi.alt - 1/2 m^2 A_mu A^mu - 1/4 F_(mu nu)F^(mu nu) - V(phi.alt) - 1/2 xi(phi.alt) F^(mu nu)(star F)_(mu nu),
$
where
$
    (star F)_(mu nu) = 1/2 epsilon_(mu nu rho sigma)F^(rho sigma) .
$
On FRW metric we have
$
    L & = integral dif^3x sqrt(|g|) cal(L) = integral dif^3x [-6 a dot(a)^2 + 1/2 a^3 dot(phi.alt)^2 - 1/2 a partial_i phi.alt partial^i phi.alt \
       &  - 1/(4a)F_(i j)F^(i j) - 1/2 m^2 a A_i A^i + 1/2 a dot(A)_i dot(A)^i - a dot(A)_i partial^i A_0 + 1/2 a partial_i A_0 partial^i A_0 \
       & + a^3 (1/2 m^2 A_0^2 - V(phi.alt)) - 2 F_(i j) epsilon^(i j k) dot(A)_k + 2 F_(i j) epsilon^(i j k) partial_k A_0 xi(phi.alt) ].
$
We have dropped some divergence terms. Notice that the first term is constant and integrate over it gives the space volume. We proceed to compute the canonical momenta
$
    pi_a = pd(L, dot(a)) = -12 V a dot(a),
    quad pi_phi.alt = vd(L, dot(phi.alt)) = a^3 dot(phi.alt), \
    quad Pi_0 = vd(L, dot(A)_0) = 0,
    quad Pi^i = vd(L, dot(A)_i) = a dot(A)^i - a partial^i A_0 - 2epsilon^(i j k)F_(j k) .
$
We see that there's one primary constrain $phi.alt_0 = Pi_0 = 0$. The Hamitonian is then
$
    H & = -1/(24V) pi_a^2 / a + integral dif^3x [ pi_phi.alt^2 / (2a^3) + 1/2 a partial_i phi.alt partial^i phi.alt + (Pi_i Pi^i) / (2a) - partial_i Pi^i A_0 + 1/2 m^2 a A_i A^i \
        & quad a^3 (-1/2 m^2 A_0^2 + V(phi.alt)) + 2/a xi (phi.alt) F_(i j)epsilon^(i j k) Pi_k + (1 + 16xi (phi.alt)^2) / (4a) F_(i j) F^(i j)] + lambda Pi_0 .
$ <eq:hamitonian0>
We compute
$
    dot(phi.alt)_0 = { Pi_0, H } = partial_i Pi^i + m^2 a^3 A_0 equiv phi.alt_1 .
$
Note that
$
    {partial_i Pi^i, integral dif^3 x F_(j k) C^(j k)} = partial_i partial_j C^(j i) = 0 .
$
So we have
$
    dot(phi.alt)_1 = m^2 a ( lambda a^2 - partial_i Pi^i) = 0 arrow.double lambda = (partial_i Pi^i) / (a^2) .
$
So that
$
    {phi.alt_0 (arrow(x)), phi.alt_1 (arrow(y))} = -m^2 a^3 delta (arrow(x) - arrow(y)) != 0 ,
$
this means $phi.alt_0, phi.alt_1$ are first-class constraints. We compute the constrain matrix
$
    C(arrow(x), arrow(y)) = mat(0, m^2a^3; -m^2 a^3, 0) delta(arrow(x) - arrow(y)), quad
    C^(-1)(arrow(x), arrow(y)) = mat(0, 1/(m^2a^3); -1 / (m^2 a^3), 0) delta(arrow(x) - arrow(y)) .
$
So that ${phi.alt_i, Pi^j} = 0$, so the Dirac bracket ${A_i, Pi^j}^ast$ is the same as Poisson bracket ${A_i, Pi^j}$.

Now we can impose the constraints on the Hamitonian directly. Substitute
$
    Pi_0 = 0, quad
    A_0 = -1/(m^2 a^3)partial_i Pi^i
$
into @eq:hamitonian0 we have
$
    H & = -1/(24V) pi_a^2 / a + integral dif^3x [ pi_phi.alt^2 / (2a^3) + 1/2 a partial_i phi.alt partial^i phi.alt + (Pi_i Pi^i) / (2a) + (partial_i Pi^i)^2 / (2m^2a^3) + 1/2 m^2 a A_i A^i \
        & quad + a^3 V(phi.alt) + 2/a xi (phi.alt) F_(i j)epsilon^(i j k) Pi_k + (1 + 16xi (phi.alt)^2) / (4a) F_(i j) F^(i j)] .
$

To numerically solve the equation of motion using Yoshida method, we need to ensure that each term in $H$ should not envolve the canonical coordinate and momentum of the same coordinate. The term $F_(i j)epsilon^(i j k)Pi_k$ appears to contain $A_i$ and $Pi^i$ simutaneously, but $epsilon^(i j k)$ ensures that each component is separated. We only need to deal with the first term $pi_a^2 / a$. Introduce a new set of variables $(b, pi_b)$ so that
$
    pi_a^2 / a = k^2 pi_b^2 .
$
Using the symplectic condition $dif a and dif pi_a = dif b and dif pi_b$, we have
$
    pi_b = pi_a / (k sqrt(a)), quad b = 2/3 k a^(3/2) .
$
For convenience, we can set $k = 3/2$.

We can then split the Hamitionian as follows
$
    H & = K_1 + K_2 + K_3 + K_4 + K_5, quad
    K_1 = - 3 / (32V)pi_b^2, quad
    K_2 = integral dif^3x b^(-2) / 2 pi_phi.alt^2, \
    K_3 & = integral dif^3x [ b^(-2/3) / 2 Pi_i Pi^i + b^(-2) / (2m^2) (partial_i Pi^i)^2 ], \
    K_4 & = integral dif^3x[ 1/2 m^2 b^(2/3) A_i A^i + b^2 V(phi.alt) + 1/2 b^(2/3) partial_i phi.alt partial^i phi.alt + b^(-2/3) / 4 (1 + 16 xi (phi.alt)^2) F_(i j) F^(i j) ], \
    K_5 & = integral dif^3x 2 b^(-2/3) xi (phi.alt) F_(i j)epsilon^(i j k)Pi_k .
$
We compute
