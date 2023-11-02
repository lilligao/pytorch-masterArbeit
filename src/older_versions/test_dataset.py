import sys
import os
from pathlib import Path


bop_toolkit_path = Path('./lib/bop_toolkit')
# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(bop_toolkit_path))

# Do the import
import bop_toolkit_lib.dataset.bop_scenewise as bop_dataloader

scene_dir = Path('./data/tless/test_primesense/000001')
scene_data = bop_dataloader.load_scene_data(scene_dir)

