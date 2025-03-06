root=
output=vis/V2A
cd SyncFormer
python gen_feat.py --root $root

cd ../CodePredictor
python generate_code.py --root $root

cd ..
python inference.py --root $root --output_dir $output