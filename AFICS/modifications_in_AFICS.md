In the `rmsd_CN.py` file, the authors transitioned from a script that filters out "incorrect" coordination numbers to one that dynamically handles all of them. This variability is handled primarily through the interaction of three methods: **`getThresholdAtoms`**, **`getIdealGeos`**, and **`getRMSDs`**.

### 1. Inclusion of All Frames (`getThresholdAtoms`)

In the original `rmsd.py`, frames were only added to the analysis list if their coordination number (CN) matched a single user-specified value (`self.coorNum`). In the augmented version, this restriction is removed:

* **Storage Change:** The script now calculates the number of coordinating atoms for every frame and stores both the **count** and the **coordinates** in `self.thresholdAtoms`.
* **No Filtering:** Unlike the original code (which is commented out in this version), there is no `if (len(frameAtoms) == self.coorNum)` check. Every single frame is processed regardless of its CN.

### 2. Multi-CN Geometry Generation (`getIdealGeos`)

Instead of generating one set of ideal geometries for a single CN, the script now prepares comparison sets for every unique CN found in the trajectory:

* **Unique CN Identification:** The script iterates through `self.thresholdAtoms` to build `liste_cn`, a list of all unique coordination numbers encountered during the simulation.
* **Dynamic Generation:** It then loops through this list and calls `geometries.idealGeometries(val_cn, self.dist)` for **every** value in `liste_cn`, storing these reference shapes in `self.idealGeos`.

### 3. Conditional Comparison Logic (`getRMSDs`)

When calculating the final RMSD values, the script uses conditional logic to ensure each frame is only compared against physically relevant ideal shapes:

* **CN Matching:** For each frame, the code checks its specific coordination number (`frame[0]`) against the CN of the ideal geometry set (`geopos[0]` or `ideal[0]`).
* **Specific Comparison:** The RMSD calculation (`kabschRMSD`) is only triggered if the frame's CN and the ideal geometry's CN match. This allows the script to process a CN=7 frame using 7-coordinate ideal shapes and a CN=8 frame using 8-coordinate shapes within the same run.

### 4. Data Tracking and Matrix Output (`printRMSDs`)

The variability is finally handled in the output by creating a comprehensive matrix:

* **Geometry Mapping:** The `printRMSDs` method identifies every unique geometry name encountered across all coordination numbers and sorts them.
* **Sparse Matrix:** It then builds a CSV where the columns represent all possible geometries. If a frame did not have a matching CN for a specific geometry, the script leaves that cell empty or handles it via a dictionary lookup (`rmsd_dict.get(geometry)`), ensuring the data remains aligned even as the CN changes between frames.