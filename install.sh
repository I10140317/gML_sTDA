#!/bin/bash
cd ./molden_parser/src
  python3 setup.py build_ext --inplace
cd ../../

#python3 - <<'PY'
#from parse_molden import MoldenData
#m = MoldenData("h2o.molden")
#m.summary()
#PY

cd ./stda_overlap/src
  make
cd ../../

mv ./molden_parser/src/*.so ./example
mv ./stda_overlap/src/stda_overlap_v2 ./example
cp ./gML_sTDA/src/* ./example

