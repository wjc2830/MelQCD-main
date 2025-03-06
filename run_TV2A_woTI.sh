root=
output=vis/TV2A_woTI
prompt=examples/text_info.json
cd SyncFormer
python gen_feat.py --root $root

cd ../CodePredictor
python generate_code.py --root $root

cd ..
python inference.py --root $root --output_dir $output --prompt_dir $prompt --woTI True