export SRC=https://bop.felk.cvut.cz/media/data/bop_datasets
wget $SRC/tless_base.zip         # Base archive with dataset info, camera parameters, etc.
wget $SRC/tless_models.zip       # 3D object models.
wget $SRC/tless_test_all.zip     # All test images ("_bop19" for a subset used in the BOP Challenge 2019/2020).
wget $SRC/tless_train_pbr.zip    # PBR training images (rendered with BlenderProc4BOP).

unzip tless_base.zip             # Contains folder "lm".
unzip tless_models.zip -d tless     # Unpacks to "lm".
unzip tless_test_all.zip -d tless   # Unpacks to "lm".
unzip tless_train_pbr.zip -d tless  # Unpacks to "lm".
