# DSC-LSP-Code

This is the author implementation of the article "Deep spectral clustering by integrating local structure and prior
information" publised in Knowledge-Based Systems.

## Requirements
```
  python == 3.10.9
  numpy == 1.25.0
  torch == 2.0.1
  scipy == 1.11.4
```

## Run
Use the command `python main.py` to run the code. If you want to change the dataset, you can change it in the `data.py` and `main.py` files (in the load_mydata function in `data.py` and at line 77 in the `main.py` file).

## Citation

If you found this useful, please consider cite it:
```
@article{MengZL25,
  author       = {Hua Meng and
                  Yueyi Zhang and
                  Zhiguo Long},
  title        = {Deep spectral clustering by integrating local structure and prior
                  information},
  journal      = {Knowledge-Based Systems},
  volume       = {308},
  pages        = {112743},
  year         = {2025}
}
```
