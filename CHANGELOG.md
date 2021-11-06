# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
