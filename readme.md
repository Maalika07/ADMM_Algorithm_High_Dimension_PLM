# Application of the ADMM Algorithm for a High-Dimensional Partially Linear Model
## **Team Members:** 
1. Kiruthika (CB.SC.U4AIE24329) 
2. Mithul Pranav (CB.SC.U4AIE24331)
3. Maalika P (CB.SC.U4AIE24332)
4. Rithan S (CB.SC.U4AIE24348)

## Introduction

Partially linear models are useful when some variables act in a roughly linear way and other variables have curved effects. They keep part of the model easy to interpret while still allowing enough flexibility to fit nonlinear patterns. This makes them a good middle ground between a fully linear model and a fully non-parametric model.

In this project, the unknown smooth part is written with B-spline basis functions, and the full estimation problem is written as a constrained least-squares problem. The Alternating Direction Method of Multipliers (ADMM) is then used to break that problem into smaller update steps that can be solved directly. This makes the method easier to understand and easier to implement.

The MATLAB implementation covers four model classes:

- constrained partially linear regression,
- constrained linear regression,
- constrained nonlinear regression, and
- unconstrained linear regression.

The project uses the CPS 1985 wage dataset and the Wine Quality datasets to study how model form, spline terms, and equality constraints affect prediction quality.

## Objective

The main objectives of the project are:

- to write partially linear regression as a constrained least-squares problem,
- to represent unknown smooth effects with B-spline basis functions,
- to derive explicit ADMM update equations for all implemented model cases,
- to identify which predictors should be treated as nonlinear by using a degree-of-nonlinearity measure, and
- to compare linear, nonlinear, and partially linear models on test data.

## Problem Statement

Let $Y \in \mathbb{R}^n$ denote the response vector, let $X \in \mathbb{R}^{n \times p}$ denote the matrix of linear predictors, and let $Z$ denote a predictor whose effect is assumed to be smooth but unknown. If the nonlinear effect is approximated through a spline basis matrix $B \in \mathbb{R}^{n \times q}$, then the partially linear model takes the matrix form

$$Y = X\beta + B\gamma + \varepsilon,$$

where:

- $\beta \in \mathbb{R}^p$ is the vector of linear regression coefficients,
- $\gamma \in \mathbb{R}^q$ is the vector of spline coefficients,
- $\varepsilon \in \mathbb{R}^n$ is the random error vector.

In addition to ordinary least-squares fitting, prior structural information is imposed through linear equality constraints

$$R\beta = d,$$

where:

- $R \in \mathbb{R}^{k \times p}$ is a known constraint matrix,
- $d \in \mathbb{R}^k$ is a known vector,
- $k$ is the number of imposed restrictions.

The estimation problem is therefore

$$\min_{\beta,\gamma}\ \frac{1}{2}\|Y - X\beta - B\gamma\|_2^2 \quad \text{subject to} \quad R\beta = d.$$

In the implementation, the matrix $R$ is constructed from consecutive-difference rows so that selected groups of coefficients are forced to be equal. This transforms the regression task into a constrained convex optimization problem, for which ADMM provides a convenient decomposition into smaller subproblems with closed-form updates.

## Repository Files

| File | Purpose |
| --- | --- |
| `ADMM_Algorithm.m` | ADMM solver for the constrained partially linear model |
| `ADMM_Algorithm_Linear.m` | ADMM solver for the constrained linear-only model |
| `ADMM_Algorithm_NonLinear.m` | ADMM solver for the constrained nonlinear-only model |
| `ADMM_Algorithm_Linear_Uncons.m` | Unconstrained linear least-squares estimator |
| `bspline_basismatrix.m` | Construction of B-spline basis matrices |
| `degreeNL.m` | Degree-of-nonlinearity computation |
| `MFC_P1.mlx` | Main live script containing derivations, experiments, and outputs |
| `Determinants of Wages Data (CPS 1985).csv` | CPS wage dataset |
| `winequality-red.csv` | Red wine quality dataset |
| `winequality-white.csv` | White wine quality dataset |

## Dataset Description

### 1. CPS 1985 Wage Dataset

The CPS 1985 dataset contains 534 observations describing workers' wages and personal or occupational characteristics. In the implementation, the response variable is transformed as

$$Y = \log(\text{wage}),$$

which reduces scale imbalance, moderates the influence of extreme wage values, and is consistent with standard econometric practice for wage modeling.

The predictors used in the project are:

- `education`,
- `experience`,
- `age`,
- `gender`,
- `ethnicity`,
- `region`,
- `occupation`,
- `sector`,
- `union`,
- `married`.

The categorical variables are converted to dummy variables inside the live script. Depending on the experimental case:

- `experience` is treated as the nonlinear variable in the main partially linear experiment,
- `age` and `experience` are both treated as nonlinear in an alternative semiparametric experiment,
- `age` alone is treated as nonlinear in another comparison,
- all available predictors are treated as linear in the constrained linear case,
- education, age, and experience are all expanded by spline bases in the nonlinear-only case.

This dataset is particularly appropriate for a partially linear formulation because earnings are often influenced by demographic and structural variables in approximately linear ways, while the effect of accumulated experience or age is usually nonlinear.

### 2. Wine Quality Datasets

The project also includes the UCI Wine Quality datasets:

- `winequality-red.csv`: 1599 samples
- `winequality-white.csv`: 4898 samples

For both datasets, each observation contains 11 physicochemical measurements:

- `fixed acidity`
- `volatile acidity`
- `citric acid`
- `residual sugar`
- `chlorides`
- `free sulfur dioxide`
- `total sulfur dioxide`
- `density`
- `pH`
- `sulphates`
- `alcohol`

The response variable is `quality`, an expert score assigned to each wine sample. In this project, these datasets are used in the unconstrained linear case.

## Methodology

### 1. Mathematical Notation

The notation below is aligned with the mathematical formulation used in the reference paper, while also matching the matrix objects used in the MATLAB code.

Let the observed sample be

$$\{(Y_i, X_i, Z_i)\}_{i=1}^n,$$

where:

- $Y_i \in \mathbb{R}$ is the response for the $i$th observation,
- $X_i = (X_{i1}, X_{i2}, \ldots, X_{ip})^\top \in \mathbb{R}^p$ is the vector of linear covariates,
- $Z_i \in \mathbb{R}$ is the covariate entering the model through an unknown smooth function.

The stacked response vector is

$$Y = (Y_1,\ldots,Y_n)^\top \in \mathbb{R}^n,$$

the linear design matrix is

$$X = \begin{bmatrix} X_1^\top \\ X_2^\top \\ \vdots \\ X_n^\top \end{bmatrix} \in \mathbb{R}^{n \times p},$$

and the parametric coefficient vector is

$$\beta = (\beta_1,\ldots,\beta_p)^\top \in \mathbb{R}^p.$$

For the non-parametric part, the base paper writes the smooth component in terms of spline basis functions. If $q$ basis functions are used, then for each observation $Z_i$ we define the basis-evaluation vector

$$b(Z_i) = \bigl(B_1(Z_i),\ldots,B_q(Z_i)\bigr)^\top \in \mathbb{R}^q.$$

Stacking these vectors row-wise gives the basis matrix used in the code:

$$B = \begin{bmatrix} B_1(Z_1) & \cdots & B_q(Z_1) \\ B_1(Z_2) & \cdots & B_q(Z_2) \\ \vdots & \ddots & \vdots \\ B_1(Z_n) & \cdots & B_q(Z_n) \end{bmatrix} \in \mathbb{R}^{n \times q},$$

and the spline coefficient vector is

$$\gamma = (\gamma_1,\ldots,\gamma_q)^\top \in \mathbb{R}^q.$$

The random error vector is

$$\varepsilon = (\varepsilon_1,\ldots,\varepsilon_n)^\top \in \mathbb{R}^n.$$

The equality-constrained formulation also introduces:

- $R \in \mathbb{R}^{k \times p}$ or $R \in \mathbb{R}^{k \times q}$ as a constraint matrix,
- $d \in \mathbb{R}^k$ as the right-hand-side constraint vector,
- $\lambda \in \mathbb{R}^k$ as the Lagrange multiplier vector,
- $\rho > 0$ as the ADMM penalty parameter,
- `tol` as the stopping tolerance,
- `max_iter` as the maximum number of iterations.

With this notation, the fitted response is

$$\hat{Y} = X\beta + B\gamma,$$

and the residual vector is

$$r = Y - X\beta - B\gamma.$$

### 2. Partially Linear Model

Following the base paper, the partially linear model is written as

$$Y_i = X_i^\top \beta + m(Z_i) + \varepsilon_i, \qquad i = 1,\ldots,n,$$

where:

- $X_i^\top \beta$ is the parametric linear component,
- $m(Z_i)$ is an unknown smooth function,
- $\varepsilon_i$ is the random error term.

This model is called semiparametric because one part of it is linear and one part of it is a smooth curve. The advantage is that it does not force every predictor to behave in the same way.

To make the model solvable in practice, the unknown smooth function is approximated by a B-spline expansion:

$$m(Z_i) \approx \sum_{j=1}^{q} \gamma_j B_j(Z_i).$$

Hence,

$$Y_i \approx X_i^\top \beta + \sum_{j=1}^{q} \gamma_j B_j(Z_i) + \varepsilon_i.$$

Collecting all observations gives

$$Y = X\beta + B\gamma + \varepsilon.$$

This is the matrix form used by the solver `ADMM_Algorithm.m`.

### 3. B-Spline Approximation

The nonlinear component is represented through B-spline basis functions. A spline is a function made of polynomial pieces joined at selected values called knots. Instead of fitting one global polynomial over the whole predictor range, a spline fits low-degree polynomials on neighboring intervals and joins them smoothly. This makes spline models flexible enough to capture curved behavior without suffering from the instability often associated with high-degree global polynomials.

B-splines are a particularly convenient spline basis because:

- each basis function is nonzero only on a limited interval, so the representation has local support,
- the resulting basis matrix is numerically stable,
- the degree of smoothness is controlled by the polynomial degree and knot multiplicity,
- the approximation can be incorporated naturally into least-squares estimation.

In simple terms, the unknown smooth function is not estimated directly as one complicated expression. Instead, it is written as a weighted sum of basis functions, and the data are used to estimate only the weights. The shape of the fitted curve is then determined by those estimated weights together with the chosen knot sequence.

Let the knot sequence be

$$t_1 \le t_2 \le \cdots \le t_{q+r+1},$$

where $r$ denotes the spline degree. If $r=3$, the basis functions are cubic B-splines. The number of basis functions is determined by the knot vector through

$$q = (\text{number of knots}) - r - 1.$$

The basis functions are defined recursively. For degree $0$,

$$N_{i,0}(x) = \begin{cases} 1, & t_i \le x < t_{i+1}, \\ 0, & \text{otherwise}. \end{cases}$$

For degree $r \ge 1$,

$$N_{i,r}(x) = \frac{x - t_i}{t_{i+r} - t_i}\,N_{i,r-1}(x) + \frac{t_{i+r+1} - x}{t_{i+r+1} - t_{i+1}}\,N_{i+1,r-1}(x)$$

where any term with zero denominator is interpreted as zero. This recursive definition is the Cox-de Boor recursion implemented in `bspline_basismatrix.m`.

The smooth component is then approximated by

$$m(x) \approx \sum_{j=1}^{q}\gamma_j N_{j,r}(x).$$

Each coefficient $\gamma_j$ determines how strongly the corresponding basis function contributes to the final smooth curve. The fitted nonlinear effect is therefore linear in the unknown coefficients even though it is nonlinear as a function of the predictor. This is an important reason why spline-based semiparametric estimation can still be solved by linear-algebraic optimization methods.

For the $n$ observed values of the nonlinear predictor, the basis matrix is

$$B_{ij} = N_{j,r}(Z_i), \qquad i=1,\ldots,n,\ j=1,\ldots,q.$$

Thus, the $i$th row of $B$ contains all basis evaluations at $Z_i$, and the vector $B\gamma$ gives the smooth fitted contribution across the full sample.

The function `bspline_basismatrix.m` constructs exactly this matrix. In the live script:

- cubic splines are used, so the degree is $r=3$,
- repeated boundary knots are employed in the main regression experiments,
- quantile-based interior knots are used inside `degreeNL.m` for nonlinearity assessment.

In the main CPS regression sections, the knot vectors are chosen with repeated boundary knots only:

$$[\,\min(Z),\min(Z),\min(Z),\max(Z),\max(Z),\max(Z)\,].$$

With degree $r=3$, the number of basis functions is

$$q = 6 - 3 - 1 = 2.$$

Hence, in the main CPS regression experiments, each nonlinear variable is represented by a compact two-column spline basis. In `degreeNL.m`, the knot sequence is richer because interior quantiles are added, allowing a more expressive spline fit for nonlinearity diagnosis.

### 4. Constrained Least-Squares Formulation

Once the smooth term is written as a spline expansion, estimation is based on least squares. The unconstrained objective is

$$\min_{\beta,\gamma}\ \frac{1}{2}\|Y - X\beta - B\gamma\|_2^2.$$

The base paper and this project incorporate prior information through linear equality constraints:

$$R\beta = d.$$

Thus the constrained partially linear estimation problem becomes

$$\min_{\beta,\gamma}\ \frac{1}{2}\|Y - X\beta - B\gamma\|_2^2 \quad \text{subject to} \quad R\beta = d.$$

In the code, the rows of $R$ are typically chosen in the form $(0,\ldots,0,1,-1,0,\ldots,0)$, which enforces $\beta_j - \beta_{j+1} = 0$. This relation does not constrain only one pair in isolation. Because several such rows are stacked together, they form a chain of equalities. For example, if the constraint matrix contains the rows corresponding to

$$\beta_s - \beta_{s+1}=0,\qquad \beta_{s+1} - \beta_{s+2}=0,\qquad \ldots,\qquad \beta_{p-1} - \beta_p = 0,$$

then all coefficients from index $s$ onward must satisfy

$$\beta_s = \beta_{s+1} = \cdots = \beta_p.$$

This is the precise sense in which a block of coefficients is forced to be equal.

In the semiparametric CPS experiments, the starting index is chosen so that the first few coefficients remain free, while the remaining linear coefficients are tied together through these chained difference constraints. Hence:

- the early coefficients correspond to variables that are allowed to keep separate effects,
- the later coefficients are treated as a common-effect block.

In the nonlinear-only case implemented in `ADMM_Algorithm_NonLinear.m`, the same chained-difference idea is used, but it is applied to selected spline-model coefficients collected in $\gamma$ rather than to the linear coefficients $\beta$. In the corresponding live-script construction, the spline basis columns for education, age, and experience are placed first, and the dummy-coded categorical block is appended afterward. The constraint matrix starts at the first categorical column. Therefore:

- the spline coefficients associated with the smooth terms remain unconstrained,
- the categorical coefficients at the end of the basis matrix are linked by consecutive equalities,
- the imposed restriction is not "all spline coefficients are equal", but rather "the chosen trailing coefficient block is equalized".

This distinction is important for understanding what the code actually estimates.

### 5. Augmented Lagrangian and ADMM

For a general constrained problem

$$\min_{x,z}\ f(x) + g(z) \quad \text{subject to} \quad Ax + Cz = c,$$

the augmented Lagrangian is

$$\mathcal{L}_\rho(x,z,\lambda) = f(x) + g(z) + \lambda^\top(Ax + Cz - c) + \frac{\rho}{2}\|Ax + Cz - c\|_2^2.$$

ADMM alternates between minimization with respect to one block variable, then the other, and finally updates the dual variable:

$$x^{k+1} = \underset{x}{\mathrm{arg\,min}}\ \mathcal{L}_\rho(x,z^k,\lambda^k),$$

$$z^{k+1} = \underset{z}{\mathrm{arg\,min}}\ \mathcal{L}_\rho(x^{k+1},z,\lambda^k),$$

$$\lambda^{k+1} = \lambda^k + \rho(Ax^{k+1} + Cz^{k+1} - c).$$

In this project, ADMM is useful because it breaks one large constrained problem into smaller steps with direct least-squares updates.

### 6. ADMM Formulations for Different Cases

#### Case A. Constrained Partially Linear Model

This is the model implemented in `ADMM_Algorithm.m`:

$$\min_{\beta,\gamma}\ \frac{1}{2}\|Y - X\beta - B\gamma\|_2^2 \quad \text{subject to} \quad R\beta = d.$$

The augmented Lagrangian is

$$\begin{aligned} \mathcal{L}_\rho(\beta,\gamma,\lambda) &= \frac{1}{2}\|Y - X\beta - B\gamma\|_2^2 \\ &\quad + \lambda^\top(R\beta - d) + \frac{\rho}{2}\|R\beta - d\|_2^2. \end{aligned}$$

For fixed $(\beta^k,\lambda^k)$, the $\gamma$-subproblem is

$$\gamma^{k+1} = \underset{\gamma}{\mathrm{arg\,min}}\ \frac{1}{2}\|Y - X\beta^k - B\gamma\|_2^2.$$

Taking the derivative with respect to $\gamma$ gives

$$-B^\top(Y - X\beta^k - B\gamma) = 0.$$

Therefore,

$$B^\top B\,\gamma = B^\top(Y - X\beta^k),$$

and hence

$$\gamma^{k+1} = (B^\top B)^\dagger\, B^\top(Y - X\beta^k).$$

For fixed $(\gamma^{k+1},\lambda^k)$, the $\beta$-subproblem is

$$\begin{aligned} \beta^{k+1} = \underset{\beta}{\mathrm{arg\,min}} \Bigl[&\frac{1}{2}\|Y - X\beta - B\gamma^{k+1}\|_2^2 \\ &+ (\lambda^k)^\top(R\beta - d) + \frac{\rho}{2}\|R\beta - d\|_2^2 \Bigr]. \end{aligned}$$

Differentiating with respect to $\beta$,

$$-X^\top(Y - X\beta - B\gamma^{k+1}) + R^\top\lambda^k + \rho R^\top(R\beta - d) = 0.$$

Rearranging,

$$(X^\top X + \rho R^\top R)\beta = X^\top(Y - B\gamma^{k+1}) + R^\top(\rho d - \lambda^k).$$

Thus,

$$\beta^{k+1} = (X^\top X + \rho R^\top R)^\dagger \bigl[X^\top(Y - B\gamma^{k+1}) + R^\top(\rho d - \lambda^k)\bigr].$$

The dual update is

$$\lambda^{k+1} = \lambda^k + \rho(R\beta^{k+1} - d).$$

Therefore, the final ADMM iteration is

$$\boxed{\begin{aligned} \gamma^{k+1} &= (B^\top B)^\dagger\, B^\top(Y - X\beta^k), \\ \beta^{k+1} &= (X^\top X + \rho R^\top R)^\dagger \bigl[X^\top(Y - B\gamma^{k+1}) + R^\top(\rho d - \lambda^k)\bigr], \\ \lambda^{k+1} &= \lambda^k + \rho(R\beta^{k+1} - d). \end{aligned}}$$

#### Case B. Constrained Linear Model

This is the model implemented in `ADMM_Algorithm_Linear.m`:

$$\min_{\beta}\ \frac{1}{2}\|Y - X\beta\|_2^2 \quad \text{subject to} \quad R\beta = d.$$

The augmented Lagrangian is

$$\begin{aligned} \mathcal{L}_\rho(\beta,\lambda) &= \frac{1}{2}\|Y - X\beta\|_2^2 \\ &\quad + \lambda^\top(R\beta - d) + \frac{\rho}{2}\|R\beta - d\|_2^2. \end{aligned}$$

The $\beta$-update solves

$$\begin{aligned} \beta^{k+1} = \underset{\beta}{\mathrm{arg\,min}} \Bigl[&\frac{1}{2}\|Y - X\beta\|_2^2 \\ &+ (\lambda^k)^\top(R\beta - d) + \frac{\rho}{2}\|R\beta - d\|_2^2 \Bigr]. \end{aligned}$$

Differentiating and setting the gradient to zero:

$$-X^\top(Y - X\beta) + R^\top\lambda^k + \rho R^\top(R\beta - d) = 0.$$

So

$$(X^\top X + \rho R^\top R)\beta = X^\top Y + R^\top(\rho d - \lambda^k).$$

Hence,

$$\beta^{k+1} = (X^\top X + \rho R^\top R)^\dagger \bigl[X^\top Y + R^\top(\rho d - \lambda^k)\bigr].$$

The dual step is

$$\lambda^{k+1} = \lambda^k + \rho(R\beta^{k+1} - d).$$

The MATLAB implementation uses an equivalent rearrangement:

$$\beta^{k+1} = (X^\top X + \rho R^\top R)^\dagger \left[X^\top Y + \rho R^\top\\left(d - \frac{\lambda^k}{\rho}\right)\right].$$

#### Case C. Constrained Nonlinear-Only Model

This is the model implemented in `ADMM_Algorithm_NonLinear.m`:

$$\min_{\gamma}\ \frac{1}{2}\|Y - B\gamma\|_2^2 \quad \text{subject to} \quad R\gamma = d.$$

The augmented Lagrangian is

$$\begin{aligned} \mathcal{L}_\rho(\gamma,\lambda) &= \frac{1}{2}\|Y - B\gamma\|_2^2 \\ &\quad + \lambda^\top(R\gamma - d) + \frac{\rho}{2}\|R\gamma - d\|_2^2. \end{aligned}$$

The $\gamma$-subproblem is

$$\begin{aligned} \gamma^{k+1} = \underset{\gamma}{\mathrm{arg\,min}} \Bigl[&\frac{1}{2}\|Y - B\gamma\|_2^2 \\ &+ (\lambda^k)^\top(R\gamma - d) + \frac{\rho}{2}\|R\gamma - d\|_2^2 \Bigr]. \end{aligned}$$

Taking the derivative:

$$-B^\top(Y - B\gamma) + R^\top\lambda^k + \rho R^\top(R\gamma - d) = 0.$$

Therefore,

$$(B^\top B + \rho R^\top R)\gamma = B^\top Y + R^\top(\rho d - \lambda^k),$$

and

$$\gamma^{k+1} = (B^\top B + \rho R^\top R)^\dagger \bigl[B^\top Y + R^\top(\rho d - \lambda^k)\bigr].$$

The dual update is

$$\lambda^{k+1} = \lambda^k + \rho(R\gamma^{k+1} - d).$$

#### Case D. Unconstrained Linear Model

This is the model implemented in `ADMM_Algorithm_Linear_Uncons.m`:

$$\min_{\beta}\ \frac{1}{2}\|Y - X\beta\|_2^2.$$

Since there is no constraint, the first-order optimality condition is simply

$$-X^\top(Y - X\beta) = 0.$$

Hence,

$$X^\top X\,\beta = X^\top Y,$$

and the closed-form solution is

$$\boxed{\beta^\star = (X^\top X)^\dagger X^\top Y}$$

This is ordinary least squares computed by the Moore-Penrose pseudoinverse.

### 7. Toy Example: How the Program Works End-to-End

This section shows the full logic of the program on a very small artificial dataset. The aim is not to obtain realistic estimates, but to make the role of each matrix completely clear.

#### Step 1. Raw toy data

Suppose we have six observations. For each observation, we record:

- a response $Y$,
- two variables that we want to keep linear: `education` and `age`,
- one variable that we want to model nonlinearly: `experience`,
- three dummy variables: `gender`, `union`, and `married`.

The toy data are:

| Observation | $Y$ | Education | Age | Experience | Gender | Union | Married |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.10 | 10 | 22 | 0  | 0 | 0 | 0 |
| 2 | 2.25 | 11 | 24 | 2  | 1 | 0 | 0 |
| 3 | 2.40 | 12 | 27 | 5  | 0 | 1 | 0 |
| 4 | 2.58 | 13 | 31 | 9  | 1 | 1 | 0 |
| 5 | 2.73 | 14 | 36 | 14 | 0 | 1 | 1 |
| 6 | 2.90 | 15 | 41 | 20 | 1 | 1 | 1 |

Thus the response vector is

$$Y = \begin{bmatrix} 2.10 \\ 2.25 \\ 2.40 \\ 2.58 \\ 2.73 \\ 2.90 \end{bmatrix}.$$

#### Step 2. Decide which variables are linear and which are nonlinear

The first modeling choice in the program is:

- variables treated as linear are placed in $X$,
- variables treated as nonlinear are expanded into a spline basis and placed in $B$.

For this toy example, choose:

- linear variables: `education`, `age`, `gender`, `union`, `married`,
- nonlinear variable: `experience`.

This means:

- `education` and `age` will each receive one ordinary regression coefficient,
- each dummy variable will also receive one ordinary regression coefficient,
- `experience` will not receive a single slope coefficient; instead, it will be represented through spline basis columns.

#### Step 3. Construct the linear design matrix $X$

Using the linear variables, the design matrix is

$$X = \begin{bmatrix} 10 & 22 & 0 & 0 & 0 \\ 11 & 24 & 1 & 0 & 0 \\ 12 & 27 & 0 & 1 & 0 \\ 13 & 31 & 1 & 1 & 0 \\ 14 & 36 & 0 & 1 & 1 \\ 15 & 41 & 1 & 1 & 1 \end{bmatrix},$$

with column order $[\text{education},\ \text{age},\ \text{gender},\ \text{union},\ \text{married}]$. So the coefficient vector is

$$\beta = \begin{bmatrix} \beta_1 \\ \beta_2 \\ \beta_3 \\ \beta_4 \\ \beta_5 \end{bmatrix}.$$

Each row of $X$ contains the linear information for one observation. For example, the fourth row $[13,\ 31,\ 1,\ 1,\ 0]$ means education = 13, age = 31, gender = 1, union = 1, married = 0. Its linear contribution to the fitted value is therefore

$$13\beta_1 + 31\beta_2 + \beta_3 + \beta_4.$$

#### Step 4. Construct the nonlinear variable vector and spline basis matrix $B$

The nonlinear predictor is

$$Z = \begin{bmatrix} 0 \\ 2 \\ 5 \\ 9 \\ 14 \\ 20 \end{bmatrix},$$

which represents experience. The code then chooses $r=3$ for a cubic spline, and the knot vector

$$t = [\min(Z),\min(Z),\min(Z),\max(Z),\max(Z),\max(Z)] = [0,0,0,20,20,20].$$

Since there are 6 knots and degree 3, the basis size is $q = 6 - 3 - 1 = 2$. So `bspline_basismatrix.m` creates a matrix with two spline columns:

$$B = \begin{bmatrix} B_1(0) & B_2(0) \\ B_1(2) & B_2(2) \\ B_1(5) & B_2(5) \\ B_1(9) & B_2(9) \\ B_1(14) & B_2(14) \\ B_1(20) & B_2(20) \end{bmatrix}.$$

Using the current spline routine, the interior rows are approximately

$$B \approx \begin{bmatrix} 0 & 0 \\ 0.2430 & 0.0270 \\ 0.4219 & 0.1406 \\ 0.4084 & 0.3341 \\ 0.1890 & 0.4410 \\ 0 & 0 \end{bmatrix}.$$

Hence the spline coefficient vector is $\gamma = (\gamma_1,\ \gamma_2)^\top$, and the nonlinear contribution for the fourth observation is approximately $0.4084\,\gamma_1 + 0.3341\,\gamma_2$. So for observation 4, the full fitted value in the partially linear model is

$$\hat{Y}_4 = 13\beta_1 + 31\beta_2 + \beta_3 + \beta_4 + 0.4084\,\gamma_1 + 0.3341\,\gamma_2.$$

This single equation shows exactly what the program is doing: one part of the effect is linear and another part is carried by the spline basis.

#### Step 5. Construct the constraint matrix $R$

Suppose we want the coefficients of `gender`, `union`, and `married` to be equal. In terms of $\beta$, this means $\beta_3 = \beta_4 = \beta_5$. The code writes the equivalent chained differences $\beta_3 - \beta_4 = 0$ and $\beta_4 - \beta_5 = 0$, giving

$$R = \begin{bmatrix} 0 & 0 & 1 & -1 & 0 \\ 0 & 0 & 0 & 1 & -1 \end{bmatrix}, \qquad d = \begin{bmatrix} 0 \\ 0 \end{bmatrix}.$$

Therefore $R\beta = d \;\Longleftrightarrow\; \beta_3 = \beta_4 = \beta_5$. This is exactly how the real code builds blocks of equal coefficients.

#### Step 6. Case A: constrained partially linear model

Now all ingredients for `ADMM_Algorithm.m` are ready: response vector $Y$, linear matrix $X$, spline matrix $B$, and constraint objects $R$ and $d$. The model is

$$Y \approx X\beta + B\gamma, \qquad R\beta = d.$$

This means the effects of `education`, `age`, `gender`, `union`, and `married` are linear, the effect of `experience` is nonlinear, and the last three linear coefficients are forced to be equal. This is the most important case in the project because it combines interpretability, smooth nonlinear modeling, and equality constraints.

#### Step 7. Case B: constrained linear-only model

Suppose we decide that `experience` should not be modeled through a spline. Then it is moved directly into the linear matrix $X_{\text{lin}} = \bigl[X \;\; Z\bigr]$. Now there is no spline matrix, so the model becomes

$$Y \approx X_{\text{lin}}\beta, \qquad R\beta = d.$$

This is the case solved by `ADMM_Algorithm_Linear.m`. This case asks: what if every predictor is treated as linear?

#### Step 8. Case C: constrained nonlinear-only model

Now consider the opposite direction. Suppose we collect everything into one large design matrix:

$$B_{\text{full}} = \bigl[ B^{(\text{education})} \; B^{(\text{age})} \; B^{(\text{experience})} \; C \bigr],$$

where $B^{(\text{education})}$, $B^{(\text{age})}$, $B^{(\text{experience})}$ are spline bases and $C$ is the dummy-variable block. Then the model is

$$Y \approx B_{\text{full}}\gamma, \qquad R\gamma = d.$$

Here there is no $\beta$ vector at all. The constraint starts at the first categorical column, so the spline coefficients remain free while the trailing categorical coefficients are tied together by chained equalities. This is the case solved by `ADMM_Algorithm_NonLinear.m`.

#### Step 9. Case D: unconstrained linear-only model

Finally, remove both the spline part and the constraints. Then the model is simply $Y \approx X\beta$, and the solution is ordinary least squares:

$$\beta^\star = (X^\top X)^\dagger X^\top Y.$$

This is the case solved by `ADMM_Algorithm_Linear_Uncons.m`, and it is the same idea used for the wine experiments.

#### Step 10. What the program is really changing across all cases

The whole project can be understood through three modeling decisions:

1. Which variables are placed in the linear matrix $X$?
2. Which variables are expanded into the spline basis matrix $B$?
3. Is a block of coefficients constrained through $R\theta = d$?

Once these choices are made, the correct solver is selected automatically:

| Modeling choice | Solver |
| --- | --- |
| Linear + spline + equality constraint on linear block | `ADMM_Algorithm.m` |
| Linear only + equality constraint | `ADMM_Algorithm_Linear.m` |
| Combined nonlinear-design matrix + equality constraint on selected coefficient block | `ADMM_Algorithm_NonLinear.m` |
| Linear only + no equality constraint | `ADMM_Algorithm_Linear_Uncons.m` |

So the toy example is simply a small version of the exact same workflow used in the real program: build matrices, choose the model structure, call the corresponding solver, and then evaluate the predictions.

### 8. Degree of Nonlinearity

The function `degreeNL.m` estimates how strongly a predictor departs from linear behavior. For a predictor $x$ and response $y$, the procedure is:

1. Standardize the predictor and center the response:

$$\tilde{x}_i = \frac{x_i - \bar{x}}{\mathrm{std}(x)}, \qquad \tilde{y}_i = y_i - \bar{y}.$$

2. Fit the straight-line model $\tilde{y} \approx a + b\tilde{x}$. If

$$L = \begin{bmatrix} 1 & \tilde{x}_1 \\  \\ 1 & \tilde{x}_n \end{bmatrix},$$

then the least-squares coefficient is $\hat{\beta}_L = (L^\top L)^\dagger L^\top \tilde{y}$, and the corresponding mean squared error is

$$\mathrm{MSE}_L = \frac{1}{n}\sum_{i=1}^n\bigl(\tilde{y}_i - \hat{y}^{(L)}_i\bigr)^2.$$

3. Fit a cubic-spline model $\tilde{y} \approx S\hat{\gamma}$, where $S$ is the spline basis matrix generated from quantile-based knots. Then $\hat{\gamma} = (S^\top S)^\dagger S^\top \tilde{y}$, and

$$\mathrm{MSE}_S = \frac{1}{n}\sum_{i=1}^n\bigl(\tilde{y}_i - \hat{y}^{(S)}_i\bigr)^2.$$

4. Define the degree of nonlinearity by

$$d_{\mathrm{NL}} = \max\\left(0,\\frac{\mathrm{MSE}_L - \mathrm{MSE}_S}{\mathrm{MSE}_L}\right).$$

If this value is close to zero, a linear fit is already adequate. Larger values indicate that the spline fit explains additional structure not captured by a straight line.

For the CPS notebook run, the recorded values are:

| Variable | Degree of nonlinearity |
| --- | ---: |
| Age | 0.07987 |
| Experience | 0.08122 |
| Education | 0.01781 |

These values support the use of age and experience as nonlinear candidates, while education behaves much more like a linear predictor.

### 9. Experimental Procedure

The computational pipeline used in the live script is:

1. load the dataset,
2. define the response vector,
3. construct the linear design matrix $X$,
4. construct the spline basis matrix $B$ when nonlinear terms are included,
5. build the constraint matrix $R$ and vector $d$,
6. split the sample into training and test sets,
7. estimate parameters using the appropriate solver,
8. evaluate the fitted model on the test set.

The project reports the following metrics:

$$\mathrm{MAE} = \underset{i}{\mathrm{median}}\,|y_i - \hat{y}_i|,$$

$$\mathrm{RMSE} = \sqrt{\frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}}(y_i - \hat{y}_i)^2}.$$

The MAE provides a robust measure of predictive error, while the RMSE penalizes larger deviations more strongly.

## Results

### 1. CPS 1985 Wage Data

Only the experiments whose printed outputs are actually stored in `MFC_P1.mlx` are reported numerically below. This avoids introducing values that are not explicitly present in the saved code outputs.

| Experiment | Nonlinear part | Solver used | MAE | RMSE |
| --- | --- | --- | ---: | ---: |
| Semiparametric PLM | Experience | `ADMM_Algorithm.m` | 0.2977 | 0.4989 |
| Constrained linear-only | None | `ADMM_Algorithm_Linear.m` | 0.4304 | 0.5553 |
| Constrained nonlinear-only | Education + Age + Experience splines | `ADMM_Algorithm_NonLinear.m` | 0.3314 | 0.5363 |
| Semiparametric PLM | Age + Experience | `ADMM_Algorithm.m` | 0.3209 | 0.5035 |
| Semiparametric PLM | Age | `ADMM_Algorithm.m` | 0.2822 | 0.4398 |

These results show that the semiparametric formulations perform better than the constrained linear-only formulation. In particular, the model that treats age as the nonlinear component attains the lowest recorded MAE and RMSE among the stored CPS runs. This is consistent with the nonlinearity analysis, which indicates that age and experience both contain stronger nonlinear structure than education.

### 2. Wine Quality Data

The live script contains code cells for the red-wine and white-wine linear regression experiments, both using `ADMM_Algorithm_Linear_Uncons.m`. However, the current saved output of `MFC_P1.mlx` does not include printed MAE or RMSE values for those cells. To keep the README fully faithful to the stored code outputs, no unsupported numerical wine results are reported here.

## Conclusion

This project studies constrained regression by using partially linear models and solving the resulting optimization problems with ADMM. The main idea is simple: keep some variables linear, model selected variables with B-splines, and estimate all unknown coefficients under equality constraints.

The derivations show that each implemented case leads to direct update formulas. The stored CPS results show that partially linear models perform better than the constrained linear-only model, especially when age or experience is allowed to have a nonlinear effect. This matches the degree-of-nonlinearity values reported by the code. The wine sections apply the same least-squares idea to another real dataset, even though the current saved live-script output does not include final MAE or RMSE values for those cells.

Overall, the project shows that ADMM is a practical way to solve constrained partially linear models when both simple interpretation and nonlinear fitting are needed.

## Running the Project

1. Open MATLAB.
2. Set the working folder to `MFC3_D7_ADMM_for_HD_PLM/src`.
3. Add the `src` folder to the MATLAB path if needed.
4. Open and run `MFC_P1.mlx` section by section.
5. If the notebook cannot find datasets, replace hard-coded absolute paths with paths valid on your machine.

## References

1. Aifen Feng, Xiaogai Chang, Youlin Shang, and Jingya Fan. *Application of the ADMM Algorithm for a High-Dimensional Partially Linear Model*. Mathematics, 2022, 10(24), 4767.
2. Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. *Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers*. Foundations and Trends in Machine Learning, 3(1):1-122, 2011.
3. E. R. Berndt. *The Practice of Econometrics: Classic and Contemporary*. Addison-Wesley, 1991.
4. Paulo Cortez, Antonio Cerdeira, Fernando Almeida, Telmo Matos, and Jose Reis. *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4):547-553, 2009.
