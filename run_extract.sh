mkdir tutorials/codex_dataset/multiplex_images/
cp -r /nfs/turbo/umms-drjieliu/proj/HPAP-Spatial/CODEX/hpapdata/HPAP-007 tutorials/codex_dataset/multiplex_images/
cp -r /nfs/turbo/umms-drjieliu/proj/HPAP-Spatial/CODEX/hpapdata/HPAP-009 tutorials/codex_dataset/multiplex_images/

python tutorials/utils/patch_extractor.py --input_dir tutorials/codex_dataset/multiplex_images --output_dir tutorials/codex_dataset/patches