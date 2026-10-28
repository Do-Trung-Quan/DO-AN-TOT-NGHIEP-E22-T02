set -e
export PYTHONIOENCODING=utf-8
python -W ignore v2_p01_after.py
python -W ignore v2_baselines.py
python -W ignore v2_run.py PG --search
python -W ignore v2_run.py T0 --search
python -W ignore v2_run.py T2 --search
for n in PG-noedge PG-complete PG-shuffle PG-global PG-noMD PG-noclin PG-noctx PG-RGCN PG-nobnn PG-enc-b PG-enc-c; do python -W ignore v2_run.py $n; done
echo CHAIN_V2_DONE
