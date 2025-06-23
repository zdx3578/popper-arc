# popper-arc

This project explores using the [Popper](https://github.com/logic-and-learning-lab/Popper) ILP system on ARC tasks.

## Installing SWI-Prolog
Popper depends on the SWI-Prolog runtime. Version **9.2.0** or newer is required.
On Ubuntu based systems you can install it from the official PPA:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:swi-prolog/stable
sudo apt update
sudo apt install swi-prolog
```

Check the installation with:

```bash
swipl --version
```

After SWI-Prolog is installed you can install Popper from source:

```bash
pip install git+https://github.com/logic-and-learning-lab/Popper@main
```

## Basic usage
Once Popper is installed it can be used from Python:

```python
from popper.util import Settings
from popper.loop import learn_solution

settings = Settings(kbpath='input_dir')
prog, score, stats = learn_solution(settings)
if prog is not None:
    Settings.print_prog_score(prog, score)
```

This repository contains utilities in `bkbias/objattr.py` to generate the BK, bias and example files from an ARC task and run Popper programmatically.

### Example: run Popper on a task

```
python -m bkbias.objattr oldcodeforref/arcMrule/popper/05a7bcf2.json --out tmpkb
```

The command above writes `bk.pl`, `bias.pl` and `exs.pl` under `tmpkb/` and then invokes Popper on them. If Popper finds a hypothesis it will be printed to the console.
