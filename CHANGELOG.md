# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.18.0] - 2024-07-15

### Changed
- Fix objective value computation if variable are complex.
- If variable are from Fourier transform with real input and hermitian property
  (see np.fft.rfftn), can use custom `hdot` function.

## [0.17.0] - 2024-07-13

### Changed
- Add GY and GR sup when available.
- Add sup for Huber.

## [0.16.0] - 2024-07-12

### Changed
- Add `val_grad` methods to objective.

## [0.15.0] - 2024-07-04

### Changed
- Update python version compatibility.

## [0.14.1] - 2023-06-06

### Fixed
- Remove a `print("here")`.
- Small docstring typo fix.

### Changed
- Rename `calc_fun` to `calc_objv`.
- Add `calc_objv` argument to lcg.
- Fix `lcg` second member update.

## [0.13.1] - 2022-09-12

### Fixed
- Fix `pcg` exposition.

### Changed
- Add docstring for pcg.

## [0.13.0] - 2022-09-12

### Changed
- Remove `calc_fun` for `lcg` since the overhead is very small

### Added
- Add `pcg`, similar to `lcg` with a differente API.

## [0.12.1] - 2022-06-20

### Changed
- Docstring and documentation fixes.
- Update to maximum 3.11 python verion.

## [0.12.0] - 2022-03-21

### Added
- The list of data feature has been factorized in a 'Stacked' class.
- QuadObjective computes lazily the second term and the constant.

### Changed
- Docstring and documentation fixes.

## [0.11.0] - 2021-12-10

### Added
- Add the Geman and Yang coefficients

### Changed
- Small fix and refactoring

## [0.10.0] - 2021-12-08

### Changed
- Docstring corrections.
- Inverse covariance fix.
- Constant computation fix in QuadObjective.

## [0.9.0] - 2021-11-30

### Changed
- Fix `fun` calc in linear CG (lcg)
- Fix `fun` attr in OptimizeResult
- Clean OptimizeResult to remove unused fields
- Fix step for real to complex operators 
- Use abs() before square to avoid complex casting
- Fix sectioning in documentation

## [0.8.0] - 2021-11-06

### Changed
- Refactor '_vect' as 'vect' and add 'vectorize' decorator.
- Refactor 'metric' array to 'invcov' as callable.
- Update demo and image illustration.
- Fix pyproject.toml for pylint.

## [0.7.0] - 2021-04-29

### Added
- Add preconditioner to 3MG.

### Changed
- Better typing and some typo correction.

## [0.5.0] - 2021-04-10

### Added
- Add *MixedObjective* class, a list-like object that represent the sum of
  objectives. Support "+" operartor.
- *Objective* support "+" operator and returns a *MixedObjective*.
- *BaseObjective* have *lastv* attribute that return the last computed objective
  value.
- *BaseObjective* have a *name* string attribute.
- Add a CHANGELOG.md file.

### Changed 
- rename *lastv* to *lastgv*.
- rename *calc_objv* to *calc_fun* 
- rename *init* to *x0* like in scipy.

## [0.4.0] - 2021-04-07

### Added
- add *lastv* attribut to *BaseObjv* that equals to the objective value after
  last gradient evaluation.
- add *calc_objv* flag to compute criterion value with low overhead.
- add a *callback* function to optimization algorithms.

### Changed 
- rename *Criterion* to *Objective*.
- rename *Potential* to *Loss*.
