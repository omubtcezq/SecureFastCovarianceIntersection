# BNPackages

**Version [1.0.0] - 2019-02-17**

## Todos

- [ ] BNTikzDefinitions should be in separate package for Beamer.

## Useful commands for submodules

### Add submodule to project

```
git submodule add https://github.com/ben-no/BNPackages.git BNPackages
```

### How to push changes in submodule
Submodules are organized differently, as noted in the [git documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Changes could accidently be overwritten.
```
cd ./BNPackages
git commit -a -m "new message"
git push origin HEAD:master
```

### Notes on checkout of superproject
If the superproject is checked out on new system, submodule will first be empty. The following commands will update the submodule.
```
git submodule init
git submodule update
```
