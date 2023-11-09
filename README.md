## Folder Structure
     
    .
    ├── data                   # datasets
    │   ├── tless              # TLESS datasets
    │   │   ├── train_pbr                # PBR-BlenderProc4BOP training images
    │   │   │   ├── 000000
    │   │   │   ├── 000001
    │   │   │   └── ......
    │   │   ├── train_primesense         # Real training images of isolated objects
    │   │   ├── train_render_reconst     # Synt. training images of isolated objects
    │   │   ├── test_primesense          # Test images
    │   │   ├── camera_primesense.json
    │   │   └── dataset_info.md
    │   ├── tutorial           # tutorial datasets
    │   └── ...                
    ├── lib                    # libraries
    ├── src                    # Source files 
    ├── thesis                 # Masterthesis text
    ├── tutorial               # pytorch tutorial
    └── README.md

Datasets are downloaded from: https://bop.felk.cvut.cz/datasets/

## Training
```shell
python ./src/train.py --run=Test_Name
```

## wandb Logger problem: numpy has no Attribute bool
Solution --> downgrade numpy
```shell
python -m pip uninstall numpy
python -m pip install numpy==1.23.1
```